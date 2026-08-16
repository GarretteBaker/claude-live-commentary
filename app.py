"""Live LLM commentary on a room conversation.

Pipeline: audio (mic or file) -> faster-whisper rolling transcription
-> Claude "silent commentator" -> projected web page (SSE).

Run:  uv run app.py                     # live mic (parecord / PipeWire)
      uv run app.py --wav lecture.mp3   # simulate from a recording
      uv run app.py --wav "https://www.youtube.com/watch?v=..."   # as if live
      uv run app.py --mock              # no API key needed, canned comments
Then project http://localhost:8710 on the screen.
"""

import argparse
import asyncio
import importlib.util
import os
import sys
import json
import queue
import re
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response, StreamingResponse

SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2  # s16le mono

COMMENTATOR_SYSTEM = """\
You are Marginalia: you write notes in the margin of a live lecture transcript, the way a sharp reader annotates a textbook. The room sees the transcript as body text on a projected page; your notes appear handwritten in the margin, each one anchored to the words it responds to, which get underlined.
You see a rolling speech-recognition transcript; it contains transcription errors — read through them and never comment on transcription quality.
Lines carry anonymous speaker letters assigned by voice. Lowercase letters ("a:", "b:", "speaker:") are provisional live attribution; when the revision lane is running, those lines get replaced within a minute by higher-accuracy versions with settled UPPERCASE letters ("A:", "B:") — the two are separate namespaces (live "a" is not necessarily settled "A"). The letters carry no roles: infer who is lecturing and who is asking from behavior. Attribution can be wrong, especially lowercase lines, short remarks, and overlapping speech — treat letters as hints, and if the room uses names, map names to letters yourself. Lines like [silence 12s] mark long pauses; settled lines may carry sound tags like [laughter] or [applause] — use them to catch jokes, irony, and the room's reactions.

Margin notes are read asynchronously: people glance at the margin when they have a spare moment, sometimes minutes after you wrote. So never narrate the moment ("now she's arguing…") — write annotations of the argument that stay worth reading later.
Each time you see the transcript, reply with either the single token PASS or one note.

Reply PASS unless your note is all of:
- anchored to something said in the last minute or two;
- still worth reading a few minutes from now — not a reaction that expires with the moment;
- not a summary or paraphrase of what was said;
- understandable on its own, without extra context;
- at most 25 words, in a margin-note register: compact and pointed; fragments, "NB:", "cf.", "?!" are all at home;
- clearly different from your previous notes (shown to you).

Useful interventions:
- identifying a hidden assumption;
- connecting two earlier remarks;
- distinguishing two senses of an ambiguous term;
- stating the central unresolved question;
- pointing out an apparent tension;
- proposing a compact example or counterexample;
- supplying a crisp relevant fact, standard term, or canonical reference.

Two refinements:
- Every note must open with a quote line "> their words" giving 3-8 consecutive words copied verbatim from the transcript (strip the timestamp and speaker label, keep the words exactly as transcribed, even if garbled). These exact words are what gets underlined in the body text, with your note drawn beside them — if you paraphrase, the anchor fails and your note floats loose. The quote does not count toward the word limit.
- If the room addresses you directly or explicitly poses a question for you to answer, answer it — this outranks the PASS criteria, and the answer may run to 80 words. Your name in the room is "Marginalia"; treat "Marginalia", "Claude", "chat" (Twitch-style: "chat, is this real?"), "the screen", or "the commentary" all as direct address (transcription may garble the name — read generously).

Each transcript line is prefixed with the wall-clock time it was transcribed, e.g. [14:03:52]. Use it to judge how recent a remark is and how fast the discussion is moving; never include timestamps in notes.

The room can vote on your notes; previous notes may carry tallies like [2↑ 1↓] and private voter notes explaining the vote. The notes are visible only to you — never quote, mention, or respond to them on screen. Use them to calibrate what this audience values, and raise the PASS bar after downvotes.

You always see the full session transcript. If the room explicitly asks you to search or look something up ("Marginalia, search for…", "chat, look up…"), reply exactly "SEARCH | <what to find>" instead of a note: a background web-search agent will look it up and its report will appear in your next turn under "Reports from your web-search agent". The room sees nothing while it runs, so deliver the answer as a note on a later turn using the report. Only search when explicitly asked to.

People in the room can write their own margin notes on the page (shown to you as "Reader margin notes" with ids like m2). To respond to one, reply "REPLY m2 | <your response, at most 40 words>" — this opens a side-conversation ("aside") attached to their note. Respond when you can add something real (an answer, a correction, a sharpened version of their point); stay silent otherwise. Readers can also open asides on YOUR notes; a parallel copy of you carries those conversations, and recent thread messages appear in your context — never use REPLY on a thread that already has messages.

You may also reply "ASK | <question, at most 15 words>": it is projected to the room as a question from you, and someone may answer aloud (the answer reaches you through the transcript). Use your judgement about when — the best questions are high-entropy: you are genuinely uncertain of the answer, and the answer would change what you write next. A garbled key term, an ambiguous referent, suspected mis-attribution, a fork in the argument you can't resolve — all fair game. Don't ask what context already determines.

Most turns should be PASS. When you decline, reply "PASS | <ten-word reason>" so the operator can tune the system; the reason is never projected. Otherwise output only the note text (with its quote line) — no preamble or markdown. LaTeX math with $...$ or $$...$$ delimiters renders on the page; use it for formulas."""

CHATTINESS_ADDENDA = {
    "strict": "",
    "chatty": """

Demo mode: lower the bar. Comment whenever there is any substantive claim, question, or disagreement you can engage with — roughly every other opportunity. Reserve PASS for pure logistics, small talk, or unintelligible audio.""",
    "eager": """

Eager mode: comment at nearly every opportunity — any claim, question, definition, or example is fair game. PASS only when nothing new was said or the audio is unintelligible.""",
}

# mutable at runtime: the operator page (/?ops) can retune chattiness live
config = {"chattiness": "strict", "context": ""}


# ---------------------------------------------------------------- gpu setup

def ensure_cuda_libs():
    """ctranslate2 needs cublas/cudnn on LD_LIBRARY_PATH, which the dynamic
    loader reads only at process start - so point it at the pip-installed
    nvidia wheels and re-exec once."""
    if os.environ.get("_CUDA_LIBS_PATCHED"):
        return
    lib_dirs = []
    for mod in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(mod)
        if spec and spec.submodule_search_locations:
            lib_dirs.append(str(Path(spec.submodule_search_locations[0]) / "lib"))
    if not lib_dirs:
        return
    prior = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([prior] if prior else []))
    os.environ["_CUDA_LIBS_PATCHED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    import ctranslate2
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


# ---------------------------------------------------------------- audio

def mic_pcm_blocks(block_sec: float, sources: list[str] | None = None):
    """One parecord per source, mixed to mono. PipeWire clocks every source
    to the same graph, so parallel captures stay sample-synced — no drift
    between two USB interfaces."""
    cmd = ["parecord", "--raw", "--format=s16le",
           f"--rate={SAMPLE_RATE}", "--channels=1"]
    procs = [subprocess.Popen(cmd + ([f"--device={s}"] if s else []),
                              stdout=subprocess.PIPE)
             for s in (sources or [None])]
    n = int(BYTES_PER_SEC * block_sec)
    frame_n = SAMPLE_RATE * ENERGY_FRAME_MS // 1000
    while True:
        bufs = [p.stdout.read(n) for p in procs]
        if any(not b for b in bufs):
            return
        if sources:
            # per-channel energy timeline, 50 ms frames, for mic-per-person
            # speaker attribution
            m = min(len(b) for b in bufs)
            chans = [np.frombuffer(b[:m], np.int16).astype(np.float32)
                     for b in bufs]
            with energy_lock:
                for f in range(0, m // 2, frame_n):
                    energy_frames.append(np.array(
                        [float(np.sqrt(np.mean(c[f:f + frame_n] ** 2)))
                         for c in chans]))
        if len(bufs) == 1:
            yield bufs[0]
            continue
        mix = np.sum([c.astype(np.int32) for c in
                      (np.frombuffer(b[:m], np.int16) for b in bufs)], axis=0)
        yield np.clip(mix, -32768, 32767).astype(np.int16).tobytes()


def file_pcm_blocks(source: str, speed: float, block_sec: float):
    """Decode a local file — or, via yt-dlp, a YouTube/etc. URL — to PCM and
    pace it like a live feed."""
    ffmpeg_cmd = ["ffmpeg", "-loglevel", "error", "-i", "pipe:0",
                  "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    if source.startswith(("http://", "https://")):
        fetcher = subprocess.Popen(
            [sys.executable, "-m", "yt_dlp", "-q", "-f", "bestaudio", "-o", "-", source],
            stdout=subprocess.PIPE,
        )
        proc = subprocess.Popen(ffmpeg_cmd, stdin=fetcher.stdout,
                                stdout=subprocess.PIPE)
    else:
        proc = subprocess.Popen(ffmpeg_cmd[:4] + [source] + ffmpeg_cmd[5:],
                                stdout=subprocess.PIPE)
    n = int(BYTES_PER_SEC * block_sec)
    while True:
        t0 = time.monotonic()
        data = proc.stdout.read(n)
        if not data:
            return
        yield data
        budget = block_sec / speed - (time.monotonic() - t0)
        if budget > 0:
            time.sleep(budget)


# ---------------------------------------------------------------- state

class Transcript:
    """Ordered segments, each optionally carrying an audio span (ms) and a
    line id, so the batch revision lane can replace the provisional streaming
    tail in place. Everything below `stable_count()` never changes again —
    that is the invariant the prompt-cache chunking relies on."""

    def __init__(self):
        self._lock = threading.Lock()
        self._segments: list[str] = []
        self._spans: list[tuple[float, float] | None] = []
        self._ids: list[int] = []
        self._next_id = 0
        self._stable = 0
        self.revision_active = False  # set by the revision thread at startup

    def append(self, text: str, span: tuple[float, float] | None = None) -> int:
        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._segments.append(text)
            self._spans.append(span)
            self._ids.append(sid)
            return sid

    def revise(self, t0_ms: float, t1_ms: float,
               lines: list[tuple[str, tuple[float, float]]]):
        """Replace all segments whose span lies inside [t0_ms, t1_ms] with the
        revised lines. Returns (replaced_ids, new_ids)."""
        with self._lock:
            idx = [i for i, sp in enumerate(self._spans)
                   if sp is not None and sp[0] >= t0_ms - 1 and sp[1] <= t1_ms + 1]
            lo, hi = (idx[0], idx[-1] + 1) if idx else (len(self._segments),) * 2
            replaced = self._ids[lo:hi]
            new_ids = list(range(self._next_id, self._next_id + len(lines)))
            self._next_id += len(lines)
            self._segments[lo:hi] = [t for t, _ in lines]
            self._spans[lo:hi] = [s for _, s in lines]
            self._ids[lo:hi] = new_ids
            self._stable = lo + len(lines)
            return replaced, new_ids

    def mark_stable(self, t1_ms: float):
        """Advance the stable boundary without replacing (empty revision)."""
        with self._lock:
            for i in range(len(self._segments) - 1, self._stable - 1, -1):
                if self._spans[i] is not None and self._spans[i][1] <= t1_ms + 1:
                    self._stable = i + 1
                    return

    def stable_count(self) -> int:
        with self._lock:
            return self._stable if self.revision_active else len(self._segments)

    def stable_end_ms(self) -> float:
        """Audio position the revision lane has settled up to — lets a
        relaunched revisor resume instead of re-revising from zero."""
        with self._lock:
            ends = [sp[1] for sp in self._spans[:self._stable] if sp is not None]
            return max(ends) if ends else 0.0

    def text(self) -> str:
        with self._lock:
            return "\n".join(self._segments)

    def tail(self, n: int) -> list[str]:
        with self._lock:
            return self._segments[-n:]

    def tail_with_ids(self, n: int) -> list[tuple[int, str]]:
        with self._lock:
            return list(zip(self._ids[-n:], self._segments[-n:]))

    def word_count(self) -> int:
        return len(self.text().split())


class SessionLog:
    """One JSONL file per run under sessions/: every transcript line, every
    Claude reply (comments and PASSes, with timing), config changes, grades.
    Lazy start() so the pre-re-exec process (ensure_cuda_libs) creates nothing."""

    def __init__(self):
        self._lock = threading.Lock()
        self.path: Path | None = None

    def start(self, args):
        d = Path(__file__).parent / "sessions"
        d.mkdir(exist_ok=True)
        self.path = d / (time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
        self.log("session_start", source=args.wav or "mic", speed=args.speed,
                 model=args.claude_model, effort=args.effort, fast=args.fast,
                 chattiness=config["chattiness"], asr=args.asr,
                 revise=args.revise, whisper=args.whisper_model)

    def log(self, type_: str, **fields):
        if self.path is None:
            return
        entry = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": type_, **fields}
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class Broadcaster:
    """Threads publish events; async SSE clients each get an asyncio.Queue."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self._clients: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=200)
        self._clients.append(q)
        return q

    def publish(self, event: dict):
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._publish_now, event)

    def _publish_now(self, event: dict):
        # a full queue means the client stopped reading; drop it
        self._clients[:] = [q for q in self._clients if not q.full()]
        for q in self._clients:
            q.put_nowait(event)


transcript = Transcript()

# raw PCM retained for the batch revision lane (~230 MB per 2 h; fine)
audio_store = bytearray()
audio_lock = threading.Lock()
audio_meta = {"wall0": None, "speed": 1.0}  # wall0 = wall time of sample 0

# mic-per-person mode: each capture channel belongs to one person, so speaker
# attribution is hardware truth — argmax of per-channel energy over the word's
# time span (your mic hears you louder than anyone else's mic does)
ENERGY_FRAME_MS = 50
energy_frames: list[np.ndarray] = []   # one per-channel RMS vector per frame
energy_lock = threading.Lock()
mic_map = {"names": []}                # non-empty => hardware attribution on


# voice-mode fusion: AssemblyAI runs as a pure speaker timeline ("who spoke
# when"), and Scribe realtime's words get labels by timestamp overlap
speaker_timeline: list[tuple[float, float, str]] = []   # (a_ms, b_ms, letter)
timeline_lock = threading.Lock()


def speaker_for_span(a_ms: float, b_ms: float) -> str | None:
    best: dict[str, float] = {}
    with timeline_lock:
        recent = speaker_timeline[-500:]
    for x, y, s in recent:
        ov = min(b_ms, y) - max(a_ms, x)
        if ov > 0:
            best[s] = best.get(s, 0.0) + ov
    return max(best, key=best.get) if best else None


def channel_for_span(a_ms: float, b_ms: float) -> int:
    f0 = int(a_ms // ENERGY_FRAME_MS)
    f1 = max(f0 + 1, int(b_ms // ENERGY_FRAME_MS))
    with energy_lock:
        window = energy_frames[f0:f1]
    if not window:   # startup edge: no frames yet
        return 0
    return int(np.mean(window, axis=0).argmax())


def ts_for_ms(ms: float) -> str:
    """Wall-clock [HH:MM:SS] for an audio offset, honoring --speed replays."""
    wall = audio_meta["wall0"] + (ms / 1000.0) / audio_meta["speed"]
    return time.strftime("%H:%M:%S", time.localtime(wall))
broadcaster = Broadcaster()
session_log = SessionLog()
search_reports: list[dict] = []  # results from spawned search agents
comments: list[dict] = []  # {"id", "text", "ts", "context"}
margins: list[dict] = []   # reader-written margin notes {"id","quote","text","ts"}
threads: list[dict] = []   # endnote threads {"id","root_kind","root_id","messages"}
thread_agent_lock = threading.Lock()  # one thread-agent call at a time
# handshake for pulling the partially-filled audio chunk through whisper
# right before a Claude call, so the prompt includes the freshest words
flush_req = threading.Event()
flush_done = threading.Event()
# populated at startup: youtube_* lets the display page embed the video
# synced to the audio feed; url is the LAN address phones should open
meta = {"youtube_id": None, "speed": 1.0, "started_at": None, "url": None}


def lan_ip() -> str:
    """The address other devices on the network reach this machine at."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # routing lookup only; no packets sent
    ip = s.getsockname()[0]
    s.close()
    return ip


# ---------------------------------------------------------------- speakers

class SpeakerLabeler:
    """Greedy online clustering of ECAPA speaker embeddings. Clusters get
    anonymous letters A, B, C… in order of first appearance — no role
    inference; Claude works out who's who from behavior."""

    SIM_THRESHOLD = 0.45   # cosine; far-field audio compresses the margin
    MIN_SEC = 0.8          # segments shorter than this inherit the last label

    def __init__(self):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self._torch = torch
        # CPU on purpose: torch's bundled cudnn clashes with the cudnn we put
        # on LD_LIBRARY_PATH for ctranslate2, and CPU is ~ms per segment anyway
        self._enc = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"},
        )
        self._centroids: list[np.ndarray] = []   # unit vectors
        self._durations: list[float] = []        # accumulated speech per cluster
        self._last = "a"

    def label(self, audio: np.ndarray, start: float, end: float) -> str:
        dur = end - start
        if dur < self.MIN_SEC:
            return self._last
        seg = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        with self._torch.no_grad():
            emb = self._enc.encode_batch(self._torch.from_numpy(seg)[None])
        v = emb.squeeze().cpu().numpy()
        v = v / np.linalg.norm(v)

        if self._centroids:
            sims = [float(v @ c) for c in self._centroids]
            best = int(np.argmax(sims))
        if not self._centroids or sims[best] < self.SIM_THRESHOLD:
            self._centroids.append(v)
            self._durations.append(dur)
            best = len(self._centroids) - 1
        else:
            # EMA keeps the centroid tracking slow drift (position, loudness)
            c = 0.9 * self._centroids[best] + 0.1 * v
            self._centroids[best] = c / np.linalg.norm(c)
            self._durations[best] += dur

        self._last = chr(97 + best) if best < 26 else f"s{best}"
        return self._last


# ---------------------------------------------------------------- transcription

def transcriber_thread(audio_q: queue.Queue, args):
    from faster_whisper import WhisperModel

    print(f"[whisper] loading {args.whisper_model} on {args.device} …")
    model = WhisperModel(args.whisper_model, device=args.device,
                         compute_type="int8" if args.device == "cpu" else "float16")
    labeler = None
    if not args.no_speakers:
        print("[speakers] loading ECAPA embedder …")
        labeler = SpeakerLabeler()
    print("[whisper] ready")
    broadcaster.publish({"type": "status", "text": "listening"})

    buf = bytearray()
    chunk_bytes = int(BYTES_PER_SEC * args.chunk_sec)
    prev_plain = ""  # label-free tail for whisper's initial_prompt

    def process(raw: bytes):
        nonlocal prev_plain
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if np.sqrt(np.mean(audio**2)) < 0.0015:  # silence gate
            return
        t0 = time.monotonic()
        segments, _info = model.transcribe(
            audio, language=args.language, vad_filter=True,
            initial_prompt=prev_plain[-200:] or None, beam_size=1,
        )
        lines: list[tuple[str, str]] = []  # (speaker label, text)
        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            who = labeler.label(audio, s.start, s.end) if labeler else ""
            if lines and lines[-1][0] == who:
                lines[-1] = (who, lines[-1][1] + " " + text)
            else:
                lines.append((who, text))
        dt = time.monotonic() - t0
        if not lines:
            return
        prev_plain = " ".join(text for _who, text in lines)
        for who, text in lines:
            line = f"{who}: {text}" if who else text
            transcript.append(f"[{time.strftime('%H:%M:%S')}] {line}")
            print(f"[asr] ({dt:.2f}s) {line}")
            broadcaster.publish({"type": "transcript", "text": line,
                                 "ts": time.strftime("%H:%M:%S")})
            session_log.log("transcript", text=line)

    while True:
        buf.extend(audio_q.get())
        if flush_req.is_set():
            flush_req.clear()
            if len(buf) >= BYTES_PER_SEC:  # ≥1 s pending: transcribe it early
                process(bytes(buf))
                buf.clear()
            flush_done.set()
        if len(buf) >= chunk_bytes:
            process(bytes(buf))
            buf.clear()


def deepgram_transcriber_thread(audio_q: queue.Queue, args):
    """Streaming ASR via Deepgram nova-3: word-level diarization, smart
    formatting, filler words. Used when --asr deepgram (or auto + key set)."""
    import websocket

    params = ("model=nova-3&encoding=linear16&sample_rate=16000&channels=1"
              "&diarize=true&smart_format=true&interim_results=false&filler_words=true")
    ws = websocket.create_connection(
        f"wss://api.deepgram.com/v1/listen?{params}",
        header=[f"Authorization: Token {os.environ['DEEPGRAM_API_KEY']}"],
    )
    print("[deepgram] connected: nova-3, streaming diarization on")
    broadcaster.publish({"type": "status", "text": "listening"})

    def label(spk) -> str:
        if spk is None:
            return "speaker"
        return chr(97 + spk) if spk < 26 else f"s{spk}"

    def sender():
        while True:
            ws.send_binary(audio_q.get())

    threading.Thread(target=sender, name="deepgram-sender", daemon=True).start()

    while True:
        msg = json.loads(ws.recv())
        if msg.get("type") != "Results":
            continue
        words = msg["channel"]["alternatives"][0]["words"]
        if not words:
            continue
        # group consecutive words by speaker into lines
        lines: list[tuple[int | None, list[str]]] = []
        for w in words:
            spk = w.get("speaker")
            token = w.get("punctuated_word", w["word"])
            if lines and lines[-1][0] == spk:
                lines[-1][1].append(token)
            else:
                lines.append((spk, [token]))
        for spk, tokens in lines:
            line = f"{label(spk)}: {' '.join(tokens)}"
            transcript.append(f"[{time.strftime('%H:%M:%S')}] {line}")
            print(f"[asr] {line}")
            broadcaster.publish({"type": "transcript", "text": line,
                                 "ts": time.strftime("%H:%M:%S")})
            session_log.log("transcript", text=line)


def assemblyai_transcriber_thread(audio_q: queue.Queue, args):
    """Streaming ASR via AssemblyAI Universal-3.5 Pro: best-in-class streaming
    accuracy with live speaker diarization. Speakers keep AssemblyAI's raw
    letters (A, B, C…). Used when --asr assemblyai (or auto + key set)."""
    import urllib.parse
    import websocket

    params = {"speech_model": "universal-3-5-pro", "encoding": "pcm_s16le",
              "sample_rate": "16000", "format_turns": "true",
              "speaker_labels": "true"}
    if args.keyterms:
        # priority-ordered file: first 100 non-comment lines are used (API cap)
        terms = [t.strip() for t in Path(args.keyterms).read_text().splitlines()
                 if t.strip() and not t.startswith("#")][:100]
        params["keyterms_prompt"] = json.dumps(terms)
        print(f"[assemblyai] boosting {len(terms)} keyterms")
    if config["context"]:
        # scenario context steers recognition toward the domain's vocabulary
        params["prompt"] = config["context"][:500]
    ws = websocket.create_connection(
        "wss://streaming.assemblyai.com/v3/ws?" + urllib.parse.urlencode(params),
        header=[f"Authorization: {os.environ['ASSEMBLYAI_API_KEY']}"],
    )
    print("[assemblyai] connected: universal-3-5-pro, streaming diarization on")
    broadcaster.publish({"type": "status", "text": "listening"})

    def sender():
        while True:
            ws.send_binary(audio_q.get())

    threading.Thread(target=sender, name="assemblyai-sender", daemon=True).start()

    def emit(line: str, span: tuple[float, float]):
        sid = transcript.append(f"[{time.strftime('%H:%M:%S')}] {line}", span)
        print(f"[asr] {line}")
        broadcaster.publish({"type": "transcript", "text": line, "id": sid,
                             "ts": time.strftime("%H:%M:%S")})
        session_log.log("transcript", text=line)

    last_end_ms = None  # end of the previous turn's last word
    while True:
        msg = json.loads(ws.recv())
        # each turn arrives once more with formatting applied; emit only that
        # final version so lines never duplicate
        if msg.get("type") != "Turn" or not msg.get("end_of_turn") \
                or not msg.get("turn_is_formatted"):
            continue
        words = msg.get("words", [])
        if not words:
            continue
        # long pauses are informative (thinking, discomfort, topic boundary)
        gap_ms = words[0]["start"] - last_end_ms if last_end_ms is not None else 0
        if gap_ms > 2500:
            emit(f"[silence {round(gap_ms / 1000)}s]",
                 (last_end_ms, words[0]["start"]))
        last_end_ms = words[-1]["end"]
        # per-word speaker: hardware truth when each mic is one person, else
        # AssemblyAI's voice diarization. lowercase = provisional (the
        # revision lane rewrites these lines); PENDING/quiet = neutral
        def who_of(w) -> str:
            if mic_map["names"]:
                return mic_map["names"][channel_for_span(w["start"], w["end"])].lower()
            spk = w.get("speaker")
            return ("speaker" if spk in (None, "UNKNOWN", "PENDING")
                    else spk.lower())

        lines: list[list] = []  # [who, tokens, start_ms, end_ms]
        for w in words:
            who = who_of(w)
            if lines and lines[-1][0] == who:
                lines[-1][1].append(w["text"])
                lines[-1][3] = w["end"]
            else:
                lines.append([who, [w["text"]], w["start"], w["end"]])
        for who, tokens, a_ms, b_ms in lines:
            emit(f"{who}: {' '.join(tokens)}", (a_ms, b_ms))


def assemblyai_diarizer_thread(audio_q: queue.Queue, args):
    """AssemblyAI reduced to a speaker timeline: same websocket, but instead of
    emitting transcript lines it records (start_ms, end_ms, letter) intervals.
    Scribe realtime's words then get speakers by timestamp overlap."""
    import urllib.parse
    import websocket

    params = {"speech_model": "universal-3-5-pro", "encoding": "pcm_s16le",
              "sample_rate": "16000", "speaker_labels": "true"}
    ws = websocket.create_connection(
        "wss://streaming.assemblyai.com/v3/ws?" + urllib.parse.urlencode(params),
        header=[f"Authorization: {os.environ['ASSEMBLYAI_API_KEY']}"],
    )
    print("[diarizer] assemblyai speaker timeline connected")

    def sender():
        while True:
            ws.send_binary(audio_q.get())

    threading.Thread(target=sender, name="diarizer-sender", daemon=True).start()

    while True:
        msg = json.loads(ws.recv())
        if msg.get("type") != "Turn" or not msg.get("end_of_turn"):
            continue
        with timeline_lock:
            for w in msg.get("words", []):
                spk = w.get("speaker")
                if spk not in (None, "UNKNOWN", "PENDING"):
                    speaker_timeline.append((w["start"], w["end"], spk.lower()))


def scribe_realtime_transcriber_thread(audio_q: queue.Queue, args):
    """Fast lane on ElevenLabs Scribe v2 Realtime (best streaming WER, no
    native diarization): word timestamps + either mic-channel attribution
    (per-person mics) or the AssemblyAI speaker timeline (voice mode, with a
    short hold so the timeline is populated before lines publish)."""
    import base64
    import urllib.parse
    import websocket

    params = {"audio_format": "pcm_16000", "include_timestamps": "true",
              "commit_strategy": "vad", "vad_silence_threshold_secs": "0.5",
              "language_code": args.language}
    if args.keyterms:
        # realtime caps keyterms at 50 of ≤20 chars each (batch: 1000 of ≤50)
        # and wants repeated query params, not a JSON array
        terms = [t.strip() for t in Path(args.keyterms).read_text().splitlines()
                 if t.strip() and not t.startswith("#") and len(t.strip()) <= 20][:50]
        params["keyterms"] = terms
        print(f"[scribe-rt] biasing {len(terms)} keyterms (≤20 chars each)")
    ws = websocket.create_connection(
        "wss://api.elevenlabs.io/v1/speech-to-text/realtime?"
        + urllib.parse.urlencode(params, doseq=True),
        header=[f"xi-api-key: {os.environ['ELEVENLABS_API_KEY']}"],
    )
    print("[scribe-rt] connected: realtime, word timestamps on")
    broadcaster.publish({"type": "status", "text": "listening"})

    # dense speech can outrun VAD commits; force one if 12 s pass without —
    # otherwise lines (and Claude's view) lag arbitrarily far behind the room
    last_commit = [time.monotonic()]

    def sender():
        while True:
            block = audio_q.get()
            force = time.monotonic() - last_commit[0] > 12
            if force:
                last_commit[0] = time.monotonic()
            ws.send(json.dumps({"message_type": "input_audio_chunk",
                                "audio_base_64": base64.b64encode(block).decode(),
                                "sample_rate": 16000,
                                **({"commit": True} if force else {})}))

    threading.Thread(target=sender, name="scribe-rt-sender", daemon=True).start()

    def emit(line: str, span: tuple[float, float]):
        sid = transcript.append(f"[{time.strftime('%H:%M:%S')}] {line}", span)
        print(f"[asr] {line}")
        broadcaster.publish({"type": "transcript", "text": line, "id": sid,
                             "ts": time.strftime("%H:%M:%S")})
        session_log.log("transcript", text=line)

    last_end_ms = None

    def process(words: list[dict]):
        nonlocal last_end_ms
        gap_ms = words[0]["start"] * 1000 - last_end_ms if last_end_ms is not None else 0
        if gap_ms > 2500:
            emit(f"[silence {round(gap_ms / 1000)}s]",
                 (last_end_ms, words[0]["start"] * 1000))
        last_end_ms = words[-1]["end"] * 1000

        def who_of(w) -> str:
            a, b = w["start"] * 1000, w["end"] * 1000
            if mic_map["names"]:
                return mic_map["names"][channel_for_span(a, b)].lower()
            return speaker_for_span(a, b) or "speaker"

        lines: list[list] = []  # [who, tokens, a_ms, b_ms]
        for w in words:
            who = who_of(w)
            if lines and lines[-1][0] == who:
                lines[-1][1].append(w["text"])
                lines[-1][3] = w["end"] * 1000
            else:
                lines.append([who, [w["text"]], w["start"] * 1000, w["end"] * 1000])
        for who, tokens, a_ms, b_ms in lines:
            emit(f"{who}: {' '.join(tokens)}", (a_ms, b_ms))

    # voice-mode fusion lag: AAI finalizes turns ~seconds after Scribe commits,
    # so hold each batch briefly before attributing; mic attribution is instant
    delay = 0.0 if mic_map["names"] else args.fuse_delay
    batches: queue.Queue = queue.Queue()

    def emitter():
        while True:
            ready, words = batches.get()
            wait = ready - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            process(words)

    threading.Thread(target=emitter, name="scribe-rt-emitter", daemon=True).start()

    seen_types: set[str] = set()
    while True:
        raw = ws.recv()
        if not raw:   # server closed the socket (e.g. idle after stream end)
            raise ConnectionError("scribe realtime: socket closed")
        msg = json.loads(raw)
        mt = msg.get("message_type")
        if mt not in seen_types:   # one shape sample per type, for API drift
            seen_types.add(mt)
            print(f"[scribe-rt] first {mt!r}: {str(msg)[:220]}")
        if mt and ("error" in mt or mt in ("rate_limited", "invalid_request",
                                           "quota_exceeded")):
            raise RuntimeError(f"scribe realtime: {msg}")
        if mt == "committed_transcript_with_timestamps":
            last_commit[0] = time.monotonic()
            words = [w for w in msg.get("words", []) if w.get("type") == "word"]
            if words:
                batches.put((time.monotonic() + delay, words))


# ---------------------------------------------------------------- revision lane

class SpeakerStitcher:
    """Scribe's speaker ids reset every request, so batch diarization needs
    identity stitched across chunks: embed each chunk-speaker's voice (ECAPA,
    CPU) and greedy-match against global centroids. Settled speakers get
    UPPERCASE letters — a separate namespace from the provisional lowercase
    streaming letters."""

    SIM_THRESHOLD = 0.40

    def __init__(self):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self._torch = torch
        self._enc = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
        self._centroids: list[np.ndarray] = []

    def letter(self, pcm: np.ndarray) -> str:
        if len(pcm) < SAMPLE_RATE // 2:   # <0.5 s of voice: don't trust it
            return "SPEAKER"
        with self._torch.no_grad():
            emb = self._enc.encode_batch(self._torch.from_numpy(pcm)[None])
        v = emb.squeeze().cpu().numpy()
        v = v / np.linalg.norm(v)
        if self._centroids:
            sims = [float(v @ c) for c in self._centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.SIM_THRESHOLD:
                c = 0.9 * self._centroids[best] + 0.1 * v
                self._centroids[best] = c / np.linalg.norm(c)
                return chr(65 + best)
        self._centroids.append(v)
        return chr(65 + len(self._centroids) - 1)


def scribe_revision_thread(args):
    """Every ~--revise-sec, re-transcribe the provisional audio tail with
    ElevenLabs Scribe v2 batch (top accuracy, full diarization, [laughter]
    tags) and rewrite the streaming lines in place. Windows end at streaming
    turn boundaries, so no word is ever cut mid-utterance."""
    import httpx
    import io
    import wave

    terms = []
    if args.keyterms:
        terms = [t.strip() for t in Path(args.keyterms).read_text().splitlines()
                 if t.strip() and not t.startswith("#")][:1000]
    # mic-per-person mode needs no voice stitching — attribution is hardware
    stitcher = SpeakerStitcher() if not mic_map["names"] else None
    transcript.revision_active = True
    print(f"[scribe] revision lane on: scribe_v2 every {args.revise_sec:.0f}s"
          + (f", {len(terms)} keyterms" if terms else ""))

    revised_until = transcript.stable_end_ms()  # resume point after a relaunch
    if revised_until:
        print(f"[scribe] resuming from {revised_until / 1000:.0f}s")

    def pcm_slice(a_ms: float, b_ms: float) -> bytes:
        with audio_lock:
            return bytes(audio_store[int(a_ms / 1000 * SAMPLE_RATE) * 2:
                                     int(b_ms / 1000 * SAMPLE_RATE) * 2])

    while True:
        time.sleep(5.0)
        with audio_lock:
            have_ms = len(audio_store) / 2 / SAMPLE_RATE * 1000
        if (have_ms - revised_until < args.revise_sec * 1000
                and not audio_meta.get("ended")):   # flush the tail at stream end
            continue
        # end the window at the newest streaming turn boundary
        spans = [sp for sp in transcript._spans if sp is not None]
        if not spans or spans[-1][1] <= revised_until:
            continue
        t0, t1 = revised_until, spans[-1][1]

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm_slice(t0, t1))
        data = {"model_id": "scribe_v2", "diarize": "true",
                "tag_audio_events": "true", "language_code": args.language}
        if terms:
            data["keyterms"] = terms   # repeated multipart fields, not JSON
        t_req = time.monotonic()
        resp = httpx.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            data=data,
            files={"file": ("chunk.wav", buf.getvalue(), "audio/wav")},
            timeout=90.0)
        resp.raise_for_status()
        words = [w for w in resp.json()["words"] if w["type"] != "spacing"]

        if not words:
            transcript.mark_stable(t1)
            revised_until = t1
            continue

        if mic_map["names"]:
            # hardware attribution: whoever's mic was loudest owns the words
            # (including audio events — the laugher's mic hears it loudest)
            def settled_who(w) -> str:
                ch = channel_for_span(t0 + w["start"] * 1000, t0 + w["end"] * 1000)
                return mic_map["names"][ch].upper()
        else:
            # per-speaker voice samples for identity stitching (≤8 s each)
            voice: dict[str, list[bytes]] = {}
            for w in words:
                if w["type"] != "word" or not w.get("speaker_id"):
                    continue
                clips = voice.setdefault(w["speaker_id"], [])
                if sum(len(c) for c in clips) < SAMPLE_RATE * 2 * 8:
                    clips.append(pcm_slice(t0 + w["start"] * 1000,
                                           t0 + w["end"] * 1000))
            letters = {
                spk: stitcher.letter(
                    np.frombuffer(b"".join(clips), np.int16).astype(np.float32) / 32768)
                for spk, clips in voice.items()}

            def settled_who(w) -> str:
                return letters.get(w.get("speaker_id"), "SPEAKER")

        # group words into lines on speaker change or a long gap
        lines: list[tuple[str, tuple[float, float]]] = []
        cur_who, cur_tokens, cur_a, cur_b = None, [], None, None
        prev_end = None

        def flush():
            nonlocal cur_tokens
            if cur_tokens:
                lines.append((f"[{ts_for_ms(t0 + cur_a * 1000)}] {cur_who}: "
                              f"{' '.join(cur_tokens)}",
                              (t0 + cur_a * 1000, t0 + cur_b * 1000)))
            cur_tokens = []

        for w in words:
            gap = w["start"] - prev_end if prev_end is not None else 0
            if gap > 2.5:
                flush()
                lines.append((f"[{ts_for_ms(t0 + w['start'] * 1000)}] "
                              f"[silence {round(gap)}s]",
                              (t0 + prev_end * 1000, t0 + w["start"] * 1000)))
                cur_who = None
            token = w["text"]
            if w["type"] == "audio_event":
                token = "[" + w["text"].strip("()") + "]"
            who = settled_who(w)
            if cur_tokens and who != cur_who:
                flush()
            if not cur_tokens:
                cur_who, cur_a = who, w["start"]
            cur_tokens.append(token)
            cur_b = w["end"]
            prev_end = w["end"]
        flush()

        replaced, new_ids = transcript.revise(t0, t1, lines)
        dt = time.monotonic() - t_req
        print(f"[scribe] revised {t0/1000:.0f}s–{t1/1000:.0f}s: "
              f"{len(replaced)} lines -> {len(lines)} ({dt:.1f}s)")
        ev_lines = []
        for sid, (text, _sp) in zip(new_ids, lines):
            ts, _, body = text.partition("] ")
            ev_lines.append({"id": sid, "text": body, "ts": ts.lstrip("[")})
        broadcaster.publish({"type": "revision", "replaced_ids": replaced,
                             "lines": ev_lines})
        session_log.log("revision", from_ms=round(t0), to_ms=round(t1),
                        replaced=len(replaced), lines=[l["text"] for l in ev_lines],
                        dt=round(dt, 1))
        revised_until = t1


# ---------------------------------------------------------------- commentary

TRANSCRIPT_CHUNK = 40  # lines per immutable cache block


def build_user_content() -> list[dict]:
    """Full transcript first, chunked into immutable blocks with a cache
    breakpoint on the last completed chunk. Caching is an exact-byte prefix
    match per block, so appending into one growing block would miss every
    time; completed chunks never change, so each call reuses the cached
    prefix and pays full price only for new speech + the volatile tail."""

    def fmt(c: dict) -> str:
        tally = f"[{c['up']}↑ {c['down']}↓] " if c["up"] or c["down"] else ""
        ask = "[you asked] " if c.get("kind") == "ask" else ""
        line = f"- {tally}{ask}{c['text']}"
        if c["notes"]:
            joined = "; ".join(f'({n["grade"]}) "{n["note"]}"' for n in c["notes"][-5:])
            line += f"\n  private voter notes: {joined}"
        return line

    prev = "\n".join(fmt(c) for c in comments[-10:]) or "(none)"
    reader_block = ""
    if margins:
        mm = "\n".join(
            f"- m{m['id']} on \"{m['quote'][:80]}\": {m['text']}"
            + (f"  (already in thread #{m['thread']})" if m.get("thread") is not None else "")
            for m in margins[-12:])
        reader_block = f"Reader margin notes (written by people in the room):\n{mm}\n\n"
    threads_block = ""
    if threads:
        tt = []
        for t in threads[-5:]:
            msgs = "; ".join(f"{m['who']}: {m['text'][:100]}" for m in t["messages"][-4:])
            tt.append(f"#{t['id']} (on {t['root_kind']} {t['root_id']}): {msgs}")
        threads_block = ("Open asides (a parallel you answers these — "
                        "context only):\n" + "\n".join(tt) + "\n\n")
    # margin backpressure: when notes come faster than the margin can absorb
    # them, say so — the PASS bar rises at the source instead of the display
    # drowning in ink
    now = time.time()
    recent = sum(1 for c in comments[-12:]
                 if now - time.mktime(time.strptime(
                     time.strftime("%Y-%m-%d ") + c["ts"], "%Y-%m-%d %H:%M:%S")) < 120)
    pressure = (f"You have already written {recent} notes in the last two minutes; "
                "the margin is crowded. PASS unless a note would be exceptional.\n\n"
                if recent >= 4 else "")
    reports = "\n".join(
        f"- [{r['ts']}] you asked \"{r['query']}\" → {r['report']}"
        for r in search_reports[-3:])
    reports_block = f"Reports from your web-search agent:\n{reports}\n\n" if reports else ""
    segs = transcript.tail(10**9)
    header = "Full session transcript (most recent speech last):\n"
    blocks: list[dict] = []
    # cache chunks may only contain lines that will never change again: with
    # the revision lane on, that is the revised prefix — the provisional
    # streaming tail stays in the volatile block
    n_full = min(len(segs), transcript.stable_count()) // TRANSCRIPT_CHUNK
    for i in range(n_full):
        chunk = segs[i * TRANSCRIPT_CHUNK:(i + 1) * TRANSCRIPT_CHUNK]
        blocks.append({"type": "text",
                       "text": (header if i == 0 else "") + "\n".join(chunk) + "\n"})
    if blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    tail_lines = segs[n_full * TRANSCRIPT_CHUNK:]
    blocks.append({"type": "text",
                   "text": ((header if not blocks else "")
                            + ("\n".join(tail_lines) if tail_lines else "(nothing yet)")
                            + "\n")})
    blocks.append({"type": "text",
                   "text": (f"\nYour previous comments:\n{prev}\n\n{reader_block}"
                            f"{threads_block}{reports_block}{pressure}"
                            f"Reply with PASS, SEARCH | <query>, ASK | <question>, "
                            f"REPLY m<id> | <response>, or one note.")})
    return blocks


def build_system_prompt() -> str:
    system = COMMENTATOR_SYSTEM + CHATTINESS_ADDENDA[config["chattiness"]]
    if mic_map["names"]:
        system += (
            "\n\nThis session uses one microphone per person, so speaker labels "
            "are hardware-attributed and reliable — and unlike the default "
            "described above, the lowercase and UPPERCASE forms of a label are "
            "the SAME person (lowercase only means the text has not been revised "
            "yet). Speakers in mic order: " + ", ".join(mic_map["names"]) + ". "
            "Speech from anyone without a mic is attributed to whichever mic "
            "heard it loudest — read attributions of clearly-different voices "
            "with that in mind.")
    if config["context"]:
        system += ("\n\nBackground provided by the operator (abstract, "
                   "curriculum, notes) — use it to sharpen comments, never "
                   "comment on it directly:\n" + config["context"])
    return system


# ------------------------------------------------------------- endnote threads

def open_thread(root_kind: str, root_id: int) -> dict:
    """Find or create the endnote thread attached to a note or reader margin."""
    for t in threads:
        if t["root_kind"] == root_kind and t["root_id"] == root_id:
            return t
    t = {"id": len(threads) + 1, "root_kind": root_kind, "root_id": root_id,
         "messages": []}
    threads.append(t)
    (margins if root_kind == "margin" else comments)[root_id]["thread"] = t["id"]
    broadcaster.publish({"type": "thread", "tid": t["id"],
                         "root_kind": root_kind, "root_id": root_id})
    session_log.log("thread", tid=t["id"], root_kind=root_kind, root_id=root_id)
    return t


def thread_post(t: dict, who: str, text: str):
    msg = {"who": who, "text": text, "ts": time.strftime("%H:%M:%S")}
    t["messages"].append(msg)
    broadcaster.publish({"type": "thread_msg", "tid": t["id"], **msg})
    session_log.log("thread_msg", tid=t["id"], who=who, text=text)


def thread_root_desc(t: dict) -> str:
    if t["root_kind"] == "margin":
        m = margins[t["root_id"]]
        return f"the reader's margin note on \"{m['quote'][:80]}\": {m['text']}"
    return f"your note: {comments[t['root_id']]['text']}"


def thread_agent_thread(args, tid: int):
    """A parallel Marginalia carries the endnote conversation: same system
    prompt, same full-transcript context (cache-shared with the main loop),
    plus the thread so far. Serialized so replies to one thread stay ordered."""
    import anthropic
    t = threads[tid - 1]
    with thread_agent_lock:
        msgs = "\n".join(f"{m['who']}: {m['text']}" for m in t["messages"])
        content = build_user_content()
        content.append({"type": "text", "text": (
            f"\n[Aside #{t["id"]}] You are continuing a side "
            f"conversation attached to {thread_root_desc(t)}\n"
            f"The thread so far:\n{msgs}\n\n"
            "Reply with only your next message in this thread: conversational "
            "margin voice, at most 60 words, no quote line, never PASS. "
            "LaTeX renders.")})
        client = anthropic.Anthropic(max_retries=4)
        resp = client.beta.messages.create(
            model=args.claude_model, max_tokens=500,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": args.effort},
            system=build_system_prompt(),
            messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
    thread_post(t, "marginalia", text)
    print(f"[thread #{tid}] marginalia: {text}")


SEARCH_AGENT_SYSTEM = """\
You are the web-search agent for a live-lecture commentary system. You get one \
query from the live commentator plus a little discussion context. Search the \
web and return a compact factual report: the answer, key numbers or dates, and \
brief source attributions (site or paper names, no raw URLs). Under 150 words, \
plain text. If the web doesn't settle it, say what you found and what remains \
uncertain."""


def search_agent_thread(client, args, query: str):
    """Spawned on demand so the fast commentary loop never blocks on it."""
    t0 = time.monotonic()
    messages = [{"role": "user", "content":
                 f"Query: {query}\n\nRecent discussion context:\n{transcript.text()[-2000:]}"}]
    for _hop in range(5):  # server-side tool loop can pause; resume it
        response = client.beta.messages.create(
            model=args.claude_model,
            max_tokens=1500,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            system=SEARCH_AGENT_SYSTEM,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=messages,
        )
        if response.stop_reason != "pause_turn":
            break
        messages = [messages[0], {"role": "assistant", "content": response.content}]
    if response.stop_reason == "refusal":
        report = "(search refused)"
    else:
        report = " ".join(b.text for b in response.content if b.type == "text").strip()
    dt = time.monotonic() - t0
    search_reports.append({"query": query, "report": report,
                           "ts": time.strftime("%H:%M:%S")})
    print(f"[search] ({dt:.1f}s) {query!r} → {report[:100]}")
    session_log.log("search", query=query, report=report, dt=round(dt, 1))
    broadcaster.publish({"type": "search_done", "query": query,
                         "ts": time.strftime("%H:%M:%S"), "dt": round(dt, 1)})


def stream_claude(client, args, comment_id: int) -> tuple[str, float | None]:
    """Stream a reply, publishing deltas to the display as soon as it is
    clear the reply is a comment rather than a PASS.
    Returns (full text, seconds to first displayed words or None)."""
    t0 = time.monotonic()
    first = None
    shown = ""
    kwargs = {}
    if args.fast:  # ~2.5x output speed at 2x price; thinking tokens too, so
        kwargs = {"speed": "fast"}  # it cuts time-to-first-words directly
    with client.beta.messages.stream(
        model=args.claude_model,
        max_tokens=500,
        betas=["server-side-fallback-2026-07-01"]
              + (["fast-mode-2026-02-01"] if args.fast else []),
        fallbacks="default",
        output_config={"effort": args.effort},
        system=build_system_prompt(),
        messages=[{"role": "user", "content": build_user_content()}],
        **kwargs,
    ) as stream:
        text = ""
        for delta in stream.text_stream:
            text += delta
            clean = text.lstrip()
            # withhold until we can rule out PASS, SEARCH and ASK — none of
            # these stream to the screen (ASK lands whole via comment_done)
            if (len(clean) < 7 or clean[:4].upper() == "PASS"
                    or clean[:6].upper() == "SEARCH"
                    or clean[:5].upper() == "REPLY"
                    or clean[:5].upper() == "ASK |" or clean[:4].upper() == "ASK|"):
                continue
            if first is None:
                first = time.monotonic() - t0
                broadcaster.publish({"type": "comment_start", "id": comment_id,
                                     "ts": time.strftime("%H:%M:%S")})
            if clean != shown:
                shown = clean
                broadcaster.publish({"type": "comment_delta", "id": comment_id,
                                     "text": clean})
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        return "PASS", first
    return text.strip(), first


def stream_mock(comment_id: int) -> tuple[str, float | None]:
    reply = next(MOCK_COMMENTS, "PASS | out of canned comments")
    if reply.upper().startswith("PASS"):
        return reply, None
    broadcaster.publish({"type": "comment_start", "id": comment_id,
                         "ts": time.strftime("%H:%M:%S")})
    shown = ""
    for word in reply.split(" "):
        shown = (shown + " " + word).strip()
        time.sleep(0.08)
        broadcaster.publish({"type": "comment_delta", "id": comment_id, "text": shown})
    return reply, 0.08


MOCK_COMMENTS = iter([
    "PASS | warming up, nothing substantive yet",
    "> group discussion quality\nHidden assumption: quality is measured against a solo-LLM baseline rather than the group's own counterfactual.",
    "PASS | mock pass to exercise the operator pane",
    "Two senses of 'better than an LLM': content coverage vs. shared attention. The disagreement may only concern the first.",
])


def commentator_thread(args):
    if args.mock:
        client = None
    else:
        import anthropic
        client = anthropic.Anthropic(max_retries=4)  # ride out short network blips

    seen_words = 0
    last_fire = time.monotonic() - args.call_gap  # allow an early first call
    while True:
        time.sleep(1.0)
        now = time.monotonic()
        n = transcript.word_count()
        if n < seen_words:   # a revision shortened the transcript
            seen_words = n
        # Speculative firing: call as soon as enough new speech accumulated,
        # even mid-sentence — the reply streams to the screen as it is written.
        wait = max(0, round(args.call_gap - (now - last_fire)))
        broadcaster.publish({"type": "tick", "new_words": n - seen_words,
                             "need_words": args.min_new_words, "wait": wait})
        if wait > 0 or n - seen_words < args.min_new_words:
            continue

        # pull the partially-filled audio chunk through whisper first, so
        # Claude sees the freshest words rather than a chunk-boundary-stale
        # view (cloud ASR streams continuously; nothing to flush)
        if args.asr == "whisper":
            flush_done.clear()
            flush_req.set()
            flush_done.wait(timeout=3.0)

        last_fire = time.monotonic()
        seen_words = transcript.word_count()
        comment_id = len(comments)

        broadcaster.publish({"type": "stage", "text": "thinking…",
                             "ts": time.strftime("%H:%M:%S")})
        t0 = time.monotonic()
        if args.mock:
            reply, ttfw = stream_mock(comment_id)
        else:
            reply, ttfw = stream_claude(client, args, comment_id)
        dt = time.monotonic() - t0
        broadcaster.publish({"type": "stage", "text": "listening"})

        timing = (f"{ttfw:.1f}s to first words, " if ttfw is not None else "") + f"{dt:.1f}s total"
        print(f"[claude] ({timing}) {reply}")
        if reply.strip().upper().startswith("SEARCH"):
            query = reply.split("|", 1)[1].strip() if "|" in reply else reply.strip()[6:].strip()
            print(f"[claude] spawning search agent: {query!r}")
            broadcaster.publish({"type": "search_spawn", "text": query,
                                 "ts": time.strftime("%H:%M:%S")})
            threading.Thread(target=lambda: search_agent_thread(client, args, query),
                             name="search-agent", daemon=True).start()
            continue
        m_reply = re.match(r"REPLY\s+m?(\d+)\s*\|\s*(.*)", reply.strip(),
                           re.IGNORECASE | re.DOTALL)
        if m_reply:
            mid, text = int(m_reply.group(1)), m_reply.group(2).strip()
            if mid < len(margins) and text:
                t = open_thread("margin", mid)
                thread_post(t, "marginalia", text)
                print(f"[claude] replied to margin m{mid} (aside #{t["id"]})")
            continue
        if reply.strip().upper().startswith("PASS"):
            reason = reply.split("|", 1)[1].strip() if "|" in reply else ""
            broadcaster.publish({"type": "pass", "text": reason,
                                 "ts": time.strftime("%H:%M:%S"), "dt": round(dt, 1)})
            session_log.log("pass", reason=reason, dt=round(dt, 1))
            continue
        # "ASK | q" projects a clarification question; the room answers aloud
        # and the answer reaches Claude through the transcript
        kind = "note"
        if re.match(r"ASK\s*\|", reply.strip(), re.IGNORECASE):
            kind = "ask"
            reply = reply.split("|", 1)[1].strip()
        comment = {"id": comment_id, "text": reply, "kind": kind,
                   "ts": time.strftime("%H:%M:%S"),
                   "up": 0, "down": 0, "notes": [],
                   "context": transcript.text()[-1500:]}
        comments.append(comment)
        broadcaster.publish({"type": "comment_done", "id": comment["id"],
                             "text": comment["text"], "kind": kind,
                             "ts": comment["ts"], "dt": round(dt, 1)})
        session_log.log("comment", id=comment["id"], text=comment["text"],
                        ttfw=round(ttfw, 1) if ttfw is not None else None,
                        dt=round(dt, 1))


# ---------------------------------------------------------------- web

def make_app(args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        broadcaster.loop = asyncio.get_running_loop()

        audio_q: queue.Queue = queue.Queue()
        if args.wav:
            blocks = file_pcm_blocks(args.wav, args.speed, 0.5)
            yt = re.search(r"(?:v=|youtu\.be/|/shorts/|/live/)([\w-]{11})", args.wav)
            if yt:
                meta.update(youtube_id=yt.group(1), speed=args.speed,
                            started_at=time.time())
        else:
            blocks = mic_pcm_blocks(0.5, args.mic)

        audio_meta["speed"] = args.speed if args.wav else 1.0

        # scribe fast lane in voice mode runs AAI in parallel as a pure
        # speaker timeline — both need the audio, so tee it
        diarize_q: queue.Queue | None = (
            queue.Queue() if args.asr == "scribe" and not mic_map["names"]
            and os.environ.get("ASSEMBLYAI_API_KEY") else None)

        def reader():
            for b in blocks:
                if audio_meta["wall0"] is None:
                    audio_meta["wall0"] = time.time()
                with audio_lock:
                    audio_store.extend(b)   # retained for the revision lane
                audio_q.put(b)
                if diarize_q is not None:
                    diarize_q.put(b)
            audio_meta["ended"] = True   # lets the revisor flush the tail
            print("[audio] stream ended")

        transcriber = {"deepgram": deepgram_transcriber_thread,
                       "assemblyai": assemblyai_transcriber_thread,
                       "scribe": scribe_realtime_transcriber_thread,
                       }.get(args.asr, transcriber_thread)
        targets = {
            "audio": reader,
            "transcriber": lambda: transcriber(audio_q, args),
            "commentator": lambda: commentator_thread(args),
        }
        if diarize_q is not None:
            targets["diarizer"] = lambda: assemblyai_diarizer_thread(diarize_q, args)
        if args.revise:
            targets["revisor"] = lambda: scribe_revision_thread(args)

        def spawn(name: str):
            threading.Thread(target=targets[name], name=name, daemon=True).start()

        def on_thread_crash(exc):
            import traceback
            tb = "".join(traceback.format_exception(
                exc.exc_type, exc.exc_value, exc.exc_traceback))
            print(f"\n[FATAL] {exc.thread.name} thread crashed:\n{tb}", flush=True)
            session_log.log("error", thread=exc.thread.name, traceback=tb)
            # A transient network/API failure shouldn't end commentary for the
            # whole session: relaunch the commentator. Anything else (bad key,
            # 4xx, bugs) stays dead and visible.
            import anthropic
            transient = (anthropic.APIConnectionError, anthropic.RateLimitError,
                         anthropic.InternalServerError)
            # a 429 saying "0 fast mode tokens" means the org has no fast-mode
            # allocation at all — permanent, don't relaunch into a crash loop
            if "0 fast mode" in str(exc.exc_value):
                print("[commentator] this org has no fast-mode quota — "
                      "rerun without --fast", flush=True)
                broadcaster.publish({"type": "status",
                                     "text": "commentator crashed — see terminal"})
            elif exc.thread.name in ("transcriber", "diarizer") and args.asr in (
                    "deepgram", "assemblyai", "scribe") and (
                    issubclass(exc.exc_type, (OSError, ConnectionError))
                    or "WebSocket" in exc.exc_type.__name__) \
                    and not audio_meta.get("ended"):
                name = exc.thread.name
                print(f"[{name}] connection lost — reconnecting in 5s", flush=True)
                broadcaster.publish({"type": "status", "text": "reconnecting ASR…"})

                def relaunch_asr():
                    time.sleep(5)
                    broadcaster.publish({"type": "status", "text": "live"})
                    spawn(name)

                threading.Thread(target=relaunch_asr, name="asr-relauncher",
                                 daemon=True).start()
            elif exc.thread.name == "revisor":
                import httpx
                # TransportError covers ConnectError (DNS), timeouts, resets —
                # everything transient; HTTP 4xx (bad key etc.) stays dead
                if issubclass(exc.exc_type,
                              (OSError, ConnectionError, httpx.TransportError)):
                    print("[scribe] revision call failed — retrying in 10s", flush=True)

                    def relaunch_revisor():
                        time.sleep(10)
                        spawn("revisor")

                    threading.Thread(target=relaunch_revisor,
                                     name="revisor-relauncher", daemon=True).start()
            elif exc.thread.name == "commentator" and issubclass(exc.exc_type, transient):
                print("[commentator] transient failure — restarting in 15s", flush=True)
                broadcaster.publish({"type": "status", "text": "reconnecting to Claude…"})

                def relaunch():
                    time.sleep(15)
                    broadcaster.publish({"type": "status", "text": "live"})
                    spawn("commentator")

                threading.Thread(target=relaunch, name="relauncher", daemon=True).start()
            else:
                broadcaster.publish({"type": "status",
                                     "text": f"{exc.thread.name} crashed — see terminal"})

        threading.excepthook = on_thread_crash
        for name in targets:
            spawn(name)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "display.html")

    @app.get("/margin")
    async def margin():
        # the margin-notes experiment: transcript as textbook body text,
        # comments handwritten in the margin (see margin.html)
        return FileResponse(Path(__file__).parent / "margin.html")

    @app.get("/meta")
    async def get_meta():
        return meta

    @app.get("/qr.svg")
    async def qr_svg():
        import io
        import qrcode
        import qrcode.image.svg
        # phones land on the grading view, so votes (and private notes) are
        # one tap away for everyone who scans
        img = qrcode.make(meta["url"] + "/?grade",
                          image_factory=qrcode.image.svg.SvgPathImage)
        buf = io.BytesIO()
        img.save(buf)
        return Response(buf.getvalue(), media_type="image/svg+xml")

    @app.get("/config")
    async def get_config():
        return {"chattiness": config["chattiness"]}

    @app.post("/config")
    async def set_config(body: dict):
        if body.get("chattiness") in CHATTINESS_ADDENDA:
            config["chattiness"] = body["chattiness"]
            print(f"[config] chattiness -> {config['chattiness']}")
            broadcaster.publish({"type": "config", "chattiness": config["chattiness"]})
            session_log.log("config", chattiness=config["chattiness"])
        return {"chattiness": config["chattiness"]}

    @app.post("/grade")
    async def grade(body: dict):
        c = comments[int(body["id"])]
        if body["grade"] in ("up", "down"):
            c[body["grade"]] += 1
            broadcaster.publish({"type": "grade", "id": c["id"],
                                 "up": c["up"], "down": c["down"]})
        session_log.log("grade", id=c["id"], comment=c["text"],
                        grade=body["grade"], note=body.get("note", ""))
        print(f"[grade] #{c['id']} {body['grade']} (now {c['up']}↑ {c['down']}↓)")
        return {"ok": True}

    @app.post("/grade_note")
    async def grade_note(body: dict):
        """A voter's private explanation — fed to the commentator, never shown."""
        c = comments[int(body["id"])]
        note = body.get("note", "").strip()[:300]
        if note:
            c["notes"].append({"grade": body.get("grade", ""), "note": note})
            session_log.log("grade_note", id=c["id"], grade=body.get("grade", ""),
                            note=note)
            print(f"[grade] #{c['id']} note: {note}")
        return {"ok": True}

    @app.post("/margin")
    async def add_margin(body: dict):
        """A reader highlighted page text and wrote their own margin note."""
        m = {"id": len(margins), "quote": body.get("quote", "").strip()[:300],
             "text": body.get("text", "").strip()[:500],
             "ts": time.strftime("%H:%M:%S"), "thread": None}
        if not m["text"]:
            return {"ok": False}
        margins.append(m)
        broadcaster.publish({"type": "user_margin", **m})
        session_log.log("user_margin", id=m["id"], quote=m["quote"], text=m["text"])
        print(f"[margin] m{m['id']} on {m['quote'][:50]!r}: {m['text']}")
        return {"ok": True, "id": m["id"]}

    @app.post("/reply")
    async def reply_endpoint(body: dict):
        """A reader message into (or opening) an endnote thread; a parallel
        Marginalia answers without blocking the main loop."""
        kind, rid = body.get("root_kind"), int(body.get("root_id", -1))
        text = body.get("text", "").strip()[:500]
        pool = margins if kind == "margin" else comments
        if not text or kind not in ("note", "margin") or not 0 <= rid < len(pool):
            return {"ok": False}
        t = open_thread(kind, rid)
        thread_post(t, "reader", text)
        print(f"[thread #{t['id']}] reader: {text}")
        if args.mock:
            thread_post(t, "marginalia",
                        "(mock) Say more — which sense do you mean?")
        else:
            threading.Thread(target=lambda: thread_agent_thread(args, t["id"]),
                             name=f"thread-agent-{t['id']}", daemon=True).start()
        return {"ok": True, "tid": t["id"]}

    @app.get("/events")
    async def events():
        async def gen():
            q = broadcaster.subscribe()
            # replay the recent conversation for late joiners, interleaved in
            # time order; stored transcript lines are "[HH:MM:SS] text"
            replay = []
            for sid, seg in transcript.tail_with_ids(25):
                ts, _, text = seg.partition("] ")
                ts = ts.lstrip("[")
                replay.append((ts, {"type": "transcript", "text": text,
                                    "ts": ts, "id": sid}))
            for c in comments[-8:]:
                replay.append((c["ts"], {"type": "comment", "id": c["id"],
                                         "text": c["text"], "ts": c["ts"],
                                         "kind": c.get("kind", "note"),
                                         "up": c["up"], "down": c["down"]}))
            for m in margins[-10:]:
                replay.append((m["ts"], {"type": "user_margin", **m}))
            for t in threads:
                if not t["messages"]:
                    continue
                t0 = t["messages"][0]["ts"]
                replay.append((t0, {"type": "thread", "tid": t["id"],
                                    "root_kind": t["root_kind"],
                                    "root_id": t["root_id"]}))
                for msg in t["messages"][-8:]:
                    replay.append((msg["ts"], {"type": "thread_msg",
                                               "tid": t["id"], **msg}))
            for _ts, ev in sorted(replay, key=lambda p: p[0]):
                yield f"data: {json.dumps(ev)}\n\n"
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


# ---------------------------------------------------------------- main

def main():
    sys.stdout.reconfigure(line_buffering=True)  # logs stay live when piped to a file
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav", "--input", dest="wav", metavar="FILE_OR_URL",
                   help="audio/video file or YouTube/etc. URL to simulate a live feed from (mic if omitted)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed for --wav")
    p.add_argument("--mic", action="append", metavar="SOURCE",
                   help="PulseAudio/PipeWire source name; repeat the flag to mix "
                        "several (e.g. two USB interfaces) into one mono feed. "
                        "Default: the system default source. List sources with: "
                        "pactl list sources short")
    p.add_argument("--mic-names", metavar="NAME,NAME,...",
                   help="one name per --mic source, in order: each mic belongs to "
                        "that person and speaker attribution becomes hardware "
                        "truth (loudest mic owns the words). Defaults to letters "
                        "when several mics are captured")
    p.add_argument("--mock", action="store_true", help="canned comments instead of the Claude API")
    p.add_argument("--asr", default="auto",
                   choices=["auto", "whisper", "deepgram", "assemblyai", "scribe"],
                   help="auto = scribe realtime (ElevenLabs, best streaming WER; "
                        "diarization from per-person mics, or from a parallel "
                        "AssemblyAI speaker timeline in voice mode) when "
                        "ELEVENLABS_API_KEY is set, else assemblyai, else deepgram, "
                        "else local whisper")
    p.add_argument("--fuse-delay", type=float, default=2.5,
                   help="scribe voice mode: seconds to hold committed lines so the "
                        "AssemblyAI speaker timeline can label them")
    p.add_argument("--whisper-model", default="distil-large-v3",
                   help="faster-whisper model; small.en is a lighter fallback")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--language", default="en")
    p.add_argument("--no-speakers", action="store_true",
                   help="disable speaker labeling (skips the ECAPA model)")
    p.add_argument("--chunk-sec", type=float, default=7.0, help="transcription chunk length")
    p.add_argument("--claude-model", default="claude-opus-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high"],
                   help="Claude reasoning effort; low is the main latency lever")
    p.add_argument("--fast", action="store_true",
                   help="Opus fast mode: ~2.5x generation speed at 2x price "
                        "(research preview; claude-opus-5 / claude-opus-4-8 only)")
    p.add_argument("--chattiness", choices=list(CHATTINESS_ADDENDA), default="strict",
                   help="how low the commentary bar starts (retunable live from /?ops)")
    p.add_argument("--chatty", action="store_true",
                   help="shorthand for --chattiness chatty")
    p.add_argument("--context", metavar="FILE",
                   help="text file with background for the commentator (abstract, curriculum, notes)")
    p.add_argument("--keyterms", metavar="FILE",
                   help="file with one name/term per line (first 100 used for AssemblyAI, "
                        "up to 1000 for Scribe) — boosted in recognition; the tokens "
                        "Claude can't repair from context are the ones worth listing")
    p.add_argument("--revise", choices=["auto", "on", "off"], default="auto",
                   help="batch revision lane: rewrite the provisional streaming tail with "
                        "ElevenLabs Scribe v2 (full diarization, [laughter] tags, top WER). "
                        "auto = on when ELEVENLABS_API_KEY is set and --asr is assemblyai")
    p.add_argument("--revise-sec", type=float, default=30.0,
                   help="minimum seconds of provisional audio before a revision pass")
    p.add_argument("--call-gap", type=float, default=1.0,
                   help="min seconds between Claude calls (calls never overlap, so the "
                        "effective cadence during speech is bounded by API latency)")
    p.add_argument("--min-new-words", type=int, default=1,
                   help="skip the Claude call unless this many new words arrived")
    p.add_argument("--port", type=int, default=8710)
    args = p.parse_args()

    if args.asr == "auto":
        args.asr = ("scribe" if os.environ.get("ELEVENLABS_API_KEY")
                    else "assemblyai" if os.environ.get("ASSEMBLYAI_API_KEY")
                    else "deepgram" if os.environ.get("DEEPGRAM_API_KEY")
                    else "whisper")
    default_keyterms = Path(__file__).parent / "keyterms.txt"
    if not args.keyterms and default_keyterms.exists():
        args.keyterms = str(default_keyterms)
    if args.revise == "on" and not (os.environ.get("ELEVENLABS_API_KEY")
                                    and args.asr in ("assemblyai", "scribe")):
        sys.exit("--revise on needs ELEVENLABS_API_KEY and --asr assemblyai|scribe "
                 "(revision aligns to the streaming lane's word timestamps)")
    args.revise = (args.revise == "on"
                   or (args.revise == "auto" and args.asr in ("assemblyai", "scribe")
                       and bool(os.environ.get("ELEVENLABS_API_KEY"))))
    for backend, envvar in (("deepgram", "DEEPGRAM_API_KEY"),
                            ("assemblyai", "ASSEMBLYAI_API_KEY"),
                            ("scribe", "ELEVENLABS_API_KEY")):
        if args.asr == backend and not os.environ.get(envvar):
            sys.exit(f"--asr {backend} needs {envvar} in the environment")
    ensure_cuda_libs()
    args.device = resolve_device(args.device)
    config["chattiness"] = "chatty" if args.chatty and args.chattiness == "strict" else args.chattiness
    if args.context:
        config["context"] = Path(args.context).read_text()[:8000]
        print(f"[app] context:  {args.context} ({len(config['context'])} chars)")

    session_log.start(args)
    meta["url"] = f"http://{lan_ip()}:{args.port}"
    print(f"[app] project:  {Path(__file__).resolve().parent}")
    print(f"[app] display:  http://localhost:{args.port}  (project this)")
    print(f"[app] margin:   http://localhost:{args.port}/margin  (textbook + margin-notes view)")
    print(f"[app] phones:   {meta['url']}  (QR in the corner of the display)")
    scribe_diar = ("per-person mics" if mic_map["names"]
                   else "assemblyai timeline" if os.environ.get("ASSEMBLYAI_API_KEY")
                   else "NONE until revision — set ASSEMBLYAI_API_KEY")
    asr_desc = {"deepgram": "deepgram nova-3 (streaming diarization)",
                "assemblyai": "assemblyai universal-3-5-pro (streaming diarization)",
                "scribe": f"elevenlabs scribe realtime (diarization: {scribe_diar})",
                }.get(args.asr, f"whisper {args.whisper_model} on {args.device}")
    if not args.mic and not args.wav:
        # zero-config hardware: capture every plugged-in audio-interface mic
        # input (e.g. both channels of each Volt 2) and mix them
        out = subprocess.run(["pactl", "list", "sources", "short"],
                             capture_output=True, text=True).stdout
        args.mic = [l.split("\t")[1] for l in out.splitlines()
                    if "alsa_input.usb-" in l] or None
        if args.mic:
            print(f"[app] mics:     auto-detected {len(args.mic)} USB mic inputs")
    if args.mic and not args.wav and (args.mic_names or len(args.mic) >= 2):
        # each mic = one person: hardware speaker attribution
        names = ([n.strip() for n in args.mic_names.split(",")]
                 if args.mic_names else
                 [chr(65 + i) for i in range(len(args.mic))])
        if len(names) != len(args.mic):
            sys.exit(f"--mic-names has {len(names)} names for {len(args.mic)} mics")
        mic_map["names"] = names
    if args.mic:
        for i, m in enumerate(args.mic):
            owner = f" -> {mic_map['names'][i]}" if mic_map["names"] else ""
            print(f"[app]   mic:    {m}{owner}")
    print(f"[app] asr:      {asr_desc}")
    print(f"[app] revise:   " + (f"scribe_v2 every {args.revise_sec:.0f}s "
                                 "(settled UPPERCASE letters, [laughter] tags)"
                                 if args.revise else "off (provisional ASR only)"))
    print(f"[app] log:      {session_log.path}")
    uvicorn.run(make_app(args), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
