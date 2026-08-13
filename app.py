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
Lines carry heuristic speaker labels: LECTURER is the speaker with the most airtime, AUDIENCE-n are others. Labels can be wrong, especially early on or for short remarks — treat them as hints, useful for telling audience questions apart from the main thread.

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

def mic_pcm_blocks(block_sec: float):
    proc = subprocess.Popen(
        ["parecord", "--raw", "--format=s16le",
         f"--rate={SAMPLE_RATE}", "--channels=1"],
        stdout=subprocess.PIPE,
    )
    n = int(BYTES_PER_SEC * block_sec)
    while True:
        data = proc.stdout.read(n)
        if not data:
            return
        yield data


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
    def __init__(self):
        self._lock = threading.Lock()
        self._segments: list[str] = []

    def append(self, text: str):
        with self._lock:
            self._segments.append(text)

    def text(self) -> str:
        with self._lock:
            return "\n".join(self._segments)

    def tail(self, n: int) -> list[str]:
        with self._lock:
            return self._segments[-n:]

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
                 chattiness=config["chattiness"], whisper=args.whisper_model)

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
broadcaster = Broadcaster()
session_log = SessionLog()
search_reports: list[dict] = []  # results from spawned search agents
comments: list[dict] = []  # {"id", "text", "ts", "context"}
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
    """Greedy online clustering of ECAPA speaker embeddings. The cluster with
    the most accumulated speech duration is the LECTURER; everyone else is
    AUDIENCE-n. No enrollment needed - label semantics come from airtime."""

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
        self._last = "LECTURER"

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

        lecturer = int(np.argmax(self._durations))
        self._last = ("LECTURER" if best == lecturer
                      else f"AUDIENCE-{best if best < lecturer else best - 1}")
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

    durations: dict[int, float] = {}  # speaker index -> accumulated airtime

    def label(spk) -> str:
        if spk is None:
            return "SPEAKER"
        lecturer = max(durations, key=durations.get)
        return ("LECTURER" if spk == lecturer
                else f"AUDIENCE-{spk if spk < lecturer else spk - 1}")

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
            durations[spk] = durations.get(spk, 0.0) + (w["end"] - w["start"])
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
    accuracy with live speaker diarization. Used when --asr assemblyai
    (or auto + ASSEMBLYAI_API_KEY set)."""
    import websocket

    params = ("speech_model=universal-3-5-pro&encoding=pcm_s16le"
              "&sample_rate=16000&format_turns=true&speaker_labels=true")
    ws = websocket.create_connection(
        f"wss://streaming.assemblyai.com/v3/ws?{params}",
        header=[f"Authorization: {os.environ['ASSEMBLYAI_API_KEY']}"],
    )
    print("[assemblyai] connected: universal-3-5-pro, streaming diarization on")
    broadcaster.publish({"type": "status", "text": "listening"})

    durations: dict[str, float] = {}  # speaker letter -> accumulated airtime

    def label(spk) -> str:
        if spk in (None, "UNKNOWN"):
            return "SPEAKER"
        lecturer = max(durations, key=durations.get)
        if spk == lecturer:
            return "LECTURER"
        others = sorted(k for k in durations if k != lecturer)
        return f"AUDIENCE-{others.index(spk)}"

    def sender():
        while True:
            ws.send_binary(audio_q.get())

    threading.Thread(target=sender, name="assemblyai-sender", daemon=True).start()

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
        # group consecutive words by speaker into lines (a turn is usually one
        # speaker, but word-level attribution catches quick interjections)
        lines: list[tuple[str | None, list[str]]] = []
        for w in words:
            spk = w.get("speaker")
            if spk not in (None, "UNKNOWN"):
                durations[spk] = durations.get(spk, 0.0) + (w["end"] - w["start"])
            if lines and lines[-1][0] == spk:
                lines[-1][1].append(w["text"])
            else:
                lines.append((spk, [w["text"]]))
        for spk, tokens in lines:
            line = f"{label(spk)}: {' '.join(tokens)}"
            transcript.append(f"[{time.strftime('%H:%M:%S')}] {line}")
            print(f"[asr] {line}")
            broadcaster.publish({"type": "transcript", "text": line,
                                 "ts": time.strftime("%H:%M:%S")})
            session_log.log("transcript", text=line)


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
        line = f"- {tally}{c['text']}"
        if c["notes"]:
            joined = "; ".join(f'({n["grade"]}) "{n["note"]}"' for n in c["notes"][-5:])
            line += f"\n  private voter notes: {joined}"
        return line

    prev = "\n".join(fmt(c) for c in comments[-10:]) or "(none)"
    reports = "\n".join(
        f"- [{r['ts']}] you asked \"{r['query']}\" → {r['report']}"
        for r in search_reports[-3:])
    reports_block = f"Reports from your web-search agent:\n{reports}\n\n" if reports else ""
    segs = transcript.tail(10**9)
    header = "Full session transcript (most recent speech last):\n"
    blocks: list[dict] = []
    n_full = len(segs) // TRANSCRIPT_CHUNK
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
                   "text": (f"\nYour previous comments:\n{prev}\n\n{reports_block}"
                            f"Reply with PASS, SEARCH | <query>, or one comment.")})
    return blocks


def build_system_prompt() -> str:
    system = COMMENTATOR_SYSTEM + CHATTINESS_ADDENDA[config["chattiness"]]
    if config["context"]:
        system += ("\n\nBackground provided by the operator (abstract, "
                   "curriculum, notes) — use it to sharpen comments, never "
                   "comment on it directly:\n" + config["context"])
    return system


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
            # withhold until we can rule out PASS and SEARCH — neither displays
            if (len(clean) < 7 or clean[:4].upper() == "PASS"
                    or clean[:6].upper() == "SEARCH"):
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
        if reply.strip().upper().startswith("PASS"):
            reason = reply.split("|", 1)[1].strip() if "|" in reply else ""
            broadcaster.publish({"type": "pass", "text": reason,
                                 "ts": time.strftime("%H:%M:%S"), "dt": round(dt, 1)})
            session_log.log("pass", reason=reason, dt=round(dt, 1))
            continue
        comment = {"id": comment_id, "text": reply,
                   "ts": time.strftime("%H:%M:%S"),
                   "up": 0, "down": 0, "notes": [],
                   "context": transcript.text()[-1500:]}
        comments.append(comment)
        broadcaster.publish({"type": "comment_done", "id": comment["id"],
                             "text": comment["text"], "ts": comment["ts"],
                             "dt": round(dt, 1)})
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
            blocks = mic_pcm_blocks(0.5)

        def reader():
            for b in blocks:
                audio_q.put(b)
            print("[audio] stream ended")

        transcriber = {"deepgram": deepgram_transcriber_thread,
                       "assemblyai": assemblyai_transcriber_thread,
                       }.get(args.asr, transcriber_thread)
        targets = {
            "audio": reader,
            "transcriber": lambda: transcriber(audio_q, args),
            "commentator": lambda: commentator_thread(args),
        }

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
            elif exc.thread.name == "transcriber" and args.asr in ("deepgram", "assemblyai") and (
                    issubclass(exc.exc_type, (OSError, ConnectionError))
                    or "WebSocket" in exc.exc_type.__name__):
                print(f"[{args.asr}] connection lost — reconnecting in 5s", flush=True)
                broadcaster.publish({"type": "status", "text": "reconnecting ASR…"})

                def relaunch_asr():
                    time.sleep(5)
                    broadcaster.publish({"type": "status", "text": "live"})
                    spawn("transcriber")

                threading.Thread(target=relaunch_asr, name="asr-relauncher",
                                 daemon=True).start()
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

    @app.get("/events")
    async def events():
        async def gen():
            q = broadcaster.subscribe()
            # replay the recent conversation for late joiners, interleaved in
            # time order; stored transcript lines are "[HH:MM:SS] text"
            replay = []
            for seg in transcript.tail(25):
                ts, _, text = seg.partition("] ")
                ts = ts.lstrip("[")
                replay.append((ts, {"type": "transcript", "text": text, "ts": ts}))
            for c in comments[-8:]:
                replay.append((c["ts"], {"type": "comment", "id": c["id"],
                                         "text": c["text"], "ts": c["ts"],
                                         "up": c["up"], "down": c["down"]}))
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
    p.add_argument("--mock", action="store_true", help="canned comments instead of the Claude API")
    p.add_argument("--asr", default="auto",
                   choices=["auto", "whisper", "deepgram", "assemblyai"],
                   help="auto = assemblyai streaming (universal-3-5-pro, best accuracy) "
                        "when ASSEMBLYAI_API_KEY is set, else deepgram when "
                        "DEEPGRAM_API_KEY is set, else local whisper")
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
    p.add_argument("--call-gap", type=float, default=1.0,
                   help="min seconds between Claude calls (calls never overlap, so the "
                        "effective cadence during speech is bounded by API latency)")
    p.add_argument("--min-new-words", type=int, default=1,
                   help="skip the Claude call unless this many new words arrived")
    p.add_argument("--port", type=int, default=8710)
    args = p.parse_args()

    if args.asr == "auto":
        args.asr = ("assemblyai" if os.environ.get("ASSEMBLYAI_API_KEY")
                    else "deepgram" if os.environ.get("DEEPGRAM_API_KEY")
                    else "whisper")
    for backend, envvar in (("deepgram", "DEEPGRAM_API_KEY"),
                            ("assemblyai", "ASSEMBLYAI_API_KEY")):
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
    asr_desc = {"deepgram": "deepgram nova-3 (streaming diarization)",
                "assemblyai": "assemblyai universal-3-5-pro (streaming diarization)",
                }.get(args.asr, f"whisper {args.whisper_model} on {args.device}")
    print(f"[app] asr:      {asr_desc}")
    print(f"[app] log:      {session_log.path}")
    uvicorn.run(make_app(args), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
