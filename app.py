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
You are a silent commentator observing a live intellectual discussion or lecture.
You see a rolling speech-recognition transcript; it contains transcription errors — read through them and never comment on transcription quality.
Lines carry heuristic speaker labels: LECTURER is the speaker with the most airtime, AUDIENCE-n are others. Labels can be wrong, especially early on or for short remarks — treat them as hints, useful for telling audience questions apart from the main thread.

Your output may be projected on a screen in the room, so speak rarely.
Each time you see the transcript, reply with either the single token PASS or one comment.

Reply PASS unless your comment is all of:
- immediately relevant to the most recent minute or two of discussion;
- not a summary or paraphrase of what was said;
- understandable on its own, without extra context;
- at most 25 words;
- clearly different from your previous comments (shown to you).

Useful interventions:
- identifying a hidden assumption;
- connecting two earlier remarks;
- distinguishing two senses of an ambiguous term;
- stating the central unresolved question;
- pointing out an apparent tension;
- proposing a compact example or counterexample;
- supplying a crisp relevant fact, standard term, or canonical reference.

Two refinements:
- Every comment must open with one short verbatim quote of the transcript words you are responding to, on its own line formatted as "> their words" (strip the timestamp and speaker label). The quote does not count toward the word limit.
- If the room addresses you directly (e.g. "Claude", "the screen", "the commentary") or explicitly poses a question for you to answer, answer it — this outranks the PASS criteria, and the answer may run to 80 words.

Each transcript line is prefixed with the wall-clock time it was transcribed, e.g. [14:03:52]. Use it to judge how recent a remark is and how fast the discussion is moving; never include timestamps in comments.

Most turns should be PASS. When you decline, reply "PASS | <ten-word reason>" so the operator can tune the system; the reason is never projected. Otherwise output only the comment text (with its optional quote line) — no preamble or markdown."""

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
            broadcaster.publish({"type": "transcript", "text": line})
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


# ---------------------------------------------------------------- commentary

def build_user_prompt() -> str:
    text = transcript.text()[-12000:]  # rolling window
    prev = "\n".join(f"- {c['text']}" for c in comments[-10:]) or "(none)"
    return (f"Your previous comments:\n{prev}\n\n"
            f"Rolling transcript (most recent speech last):\n{text}\n\n"
            f"Reply with PASS or one comment.")


def build_system_prompt() -> str:
    system = COMMENTATOR_SYSTEM + CHATTINESS_ADDENDA[config["chattiness"]]
    if config["context"]:
        system += ("\n\nBackground provided by the operator (abstract, "
                   "curriculum, notes) — use it to sharpen comments, never "
                   "comment on it directly:\n" + config["context"])
    return system


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
        messages=[{"role": "user", "content": build_user_prompt()}],
        **kwargs,
    ) as stream:
        text = ""
        for delta in stream.text_stream:
            text += delta
            clean = text.lstrip()
            if len(clean) < 4 or clean[:4].upper() == "PASS":
                continue  # withhold until we can rule out a PASS
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
        # Claude sees the freshest words rather than a chunk-boundary-stale view
        flush_done.clear()
        flush_req.set()
        flush_done.wait(timeout=3.0)

        last_fire = time.monotonic()
        seen_words = transcript.word_count()
        comment_id = len(comments)

        broadcaster.publish({"type": "stage", "text": "thinking…"})
        t0 = time.monotonic()
        if args.mock:
            reply, ttfw = stream_mock(comment_id)
        else:
            reply, ttfw = stream_claude(client, args, comment_id)
        dt = time.monotonic() - t0
        broadcaster.publish({"type": "stage", "text": "listening"})

        timing = (f"{ttfw:.1f}s to first words, " if ttfw is not None else "") + f"{dt:.1f}s total"
        print(f"[claude] ({timing}) {reply}")
        if reply.strip().upper().startswith("PASS"):
            reason = reply.split("|", 1)[1].strip() if "|" in reply else ""
            broadcaster.publish({"type": "pass", "text": reason,
                                 "ts": time.strftime("%H:%M:%S"), "dt": round(dt, 1)})
            session_log.log("pass", reason=reason, dt=round(dt, 1))
            continue
        comment = {"id": comment_id, "text": reply,
                   "ts": time.strftime("%H:%M:%S"),
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

        targets = {
            "audio": reader,
            "transcriber": lambda: transcriber_thread(audio_q, args),
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

    @app.get("/meta")
    async def get_meta():
        return meta

    @app.get("/qr.svg")
    async def qr_svg():
        import io
        import qrcode
        import qrcode.image.svg
        img = qrcode.make(meta["url"], image_factory=qrcode.image.svg.SvgPathImage)
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
        session_log.log("grade", id=c["id"], comment=c["text"],
                        grade=body["grade"], note=body.get("note", ""))
        print(f"[grade] #{c['id']} {body['grade']}")
        return {"ok": True}

    @app.get("/events")
    async def events():
        async def gen():
            q = broadcaster.subscribe()
            # replay context for late joiners
            for c in comments[-5:]:
                yield ("data: " + json.dumps({"type": "comment", "id": c["id"],
                                              "text": c["text"], "ts": ""}) + "\n\n")
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
    p.add_argument("--call-gap", type=float, default=10.0,
                   help="min seconds between Claude calls")
    p.add_argument("--min-new-words", type=int, default=30,
                   help="skip commentary unless this many new words arrived")
    p.add_argument("--port", type=int, default=8710)
    args = p.parse_args()

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
    print(f"[app] phones:   {meta['url']}  (QR in the corner of the display)")
    print(f"[app] whisper:  {args.whisper_model} on {args.device}")
    print(f"[app] log:      {session_log.path}")
    uvicorn.run(make_app(args), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
