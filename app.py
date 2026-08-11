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
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2  # s16le mono

COMMENTATOR_SYSTEM = """\
You are a silent commentator observing a live intellectual discussion or lecture.
You see a rolling speech-recognition transcript; it contains transcription errors — read through them and never comment on transcription quality.

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

Most turns should be PASS. When you decline, reply "PASS | <ten-word reason>" so the operator can tune the system; the reason is never projected. Otherwise output only the comment text — no preamble, quotes, or markdown."""

CHATTY_ADDENDUM = """

Demo mode: lower the bar. Comment whenever there is any substantive claim, question, or disagreement you can engage with — roughly every other opportunity. Reserve PASS for pure logistics, small talk, or unintelligible audio."""


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
        self.last_append = time.monotonic()

    def append(self, text: str):
        with self._lock:
            self._segments.append(text)
            self.last_append = time.monotonic()

    def text(self) -> str:
        with self._lock:
            return " ".join(self._segments)

    def word_count(self) -> int:
        return len(self.text().split())


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
comments: list[str] = []


# ---------------------------------------------------------------- transcription

def transcriber_thread(audio_q: queue.Queue, args):
    from faster_whisper import WhisperModel

    print(f"[whisper] loading {args.whisper_model} on {args.device} …")
    model = WhisperModel(args.whisper_model, device=args.device,
                         compute_type="int8" if args.device == "cpu" else "float16")
    print("[whisper] ready")
    broadcaster.publish({"type": "status", "text": "listening"})

    buf = bytearray()
    chunk_bytes = int(BYTES_PER_SEC * args.chunk_sec)
    while True:
        buf.extend(audio_q.get())
        if len(buf) < chunk_bytes:
            continue
        audio = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
        buf.clear()

        if np.sqrt(np.mean(audio**2)) < 0.0015:  # silence gate
            continue

        tail = transcript.text()[-200:]
        t0 = time.monotonic()
        segments, _info = model.transcribe(
            audio, language=args.language, vad_filter=True,
            initial_prompt=tail if tail else None, beam_size=1,
        )
        new = " ".join(s.text.strip() for s in segments).strip()
        dt = time.monotonic() - t0
        if not new:
            continue
        transcript.append(new)
        print(f"[asr] ({dt:.2f}s) {new}")
        broadcaster.publish({"type": "transcript", "text": new})


# ---------------------------------------------------------------- commentary

def build_user_prompt() -> str:
    text = transcript.text()[-12000:]  # rolling window
    prev = "\n".join(f"- {c}" for c in comments[-10:]) or "(none)"
    return (f"Your previous comments:\n{prev}\n\n"
            f"Rolling transcript (most recent speech last):\n{text}\n\n"
            f"Reply with PASS or one comment.")


def ask_claude(client, args) -> str:
    response = client.beta.messages.create(
        model=args.claude_model,
        max_tokens=300,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": args.effort},
        system=COMMENTATOR_SYSTEM + (CHATTY_ADDENDUM if args.chatty else ""),
        messages=[{"role": "user", "content": build_user_prompt()}],
    )
    if response.stop_reason == "refusal":
        return "PASS"
    return next(b.text for b in response.content if b.type == "text").strip()


MOCK_COMMENTS = iter([
    "PASS",
    "Hidden assumption: that group discussion quality is measured against a solo-LLM baseline rather than against the group's own counterfactual.",
    "PASS",
    "Two senses of 'better than an LLM': content coverage vs. shared attention. The disagreement may only concern the first.",
])


def commentator_thread(args):
    if args.mock:
        client = None
    else:
        import anthropic
        client = anthropic.Anthropic()

    seen_words = 0
    last_fire = time.monotonic()
    while True:
        time.sleep(2.0)
        now = time.monotonic()
        # Fire on a lull in speech (comment lands when eyes can go to the
        # screen) or when the interval has elapsed, but only given new speech.
        lull = now - transcript.last_append >= args.lull_sec
        due = now - last_fire >= args.comment_interval
        if not (lull or due):
            continue
        n = transcript.word_count()
        if n - seen_words < args.min_new_words:
            continue
        last_fire = now
        seen_words = n

        t0 = time.monotonic()
        if args.mock:
            reply = next(MOCK_COMMENTS, "PASS")
        else:
            reply = ask_claude(client, args)
        dt = time.monotonic() - t0

        trigger = "lull" if lull else "interval"
        print(f"[claude] ({dt:.1f}s, {trigger}) {reply}")
        if reply.strip().upper().startswith("PASS"):
            continue
        comments.append(reply)
        broadcaster.publish({"type": "comment", "text": reply,
                             "ts": time.strftime("%H:%M:%S")})


# ---------------------------------------------------------------- web

def make_app(args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        broadcaster.loop = asyncio.get_running_loop()

        audio_q: queue.Queue = queue.Queue()
        if args.wav:
            blocks = file_pcm_blocks(args.wav, args.speed, 0.5)
        else:
            blocks = mic_pcm_blocks(0.5)

        def reader():
            for b in blocks:
                audio_q.put(b)
            print("[audio] stream ended")

        def on_thread_crash(exc):
            print(f"\n[FATAL] {exc.thread.name} thread crashed:", flush=True)
            import traceback
            traceback.print_exception(exc.exc_type, exc.exc_value, exc.exc_traceback)
            broadcaster.publish({"type": "status",
                                 "text": f"{exc.thread.name} crashed — see terminal"})

        threading.excepthook = on_thread_crash
        for name, target in (("audio", reader),
                             ("transcriber", lambda: transcriber_thread(audio_q, args)),
                             ("commentator", lambda: commentator_thread(args))):
            threading.Thread(target=target, name=name, daemon=True).start()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "display.html")

    @app.get("/events")
    async def events():
        async def gen():
            q = broadcaster.subscribe()
            # replay context for late joiners
            for c in comments[-5:]:
                yield f"data: {json.dumps({'type': 'comment', 'text': c, 'ts': ''})}\n\n"
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
    p.add_argument("--chunk-sec", type=float, default=7.0, help="transcription chunk length")
    p.add_argument("--claude-model", default="claude-opus-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--chatty", action="store_true",
                   help="lower the commentary bar (good for demos/testing)")
    p.add_argument("--comment-interval", type=float, default=30.0,
                   help="max seconds between commentary opportunities")
    p.add_argument("--lull-sec", type=float, default=4.0,
                   help="a pause in speech this long triggers an early opportunity")
    p.add_argument("--min-new-words", type=int, default=30,
                   help="skip commentary unless this many new words arrived")
    p.add_argument("--port", type=int, default=8710)
    args = p.parse_args()

    ensure_cuda_libs()
    args.device = resolve_device(args.device)

    print(f"[app] project:  {Path(__file__).resolve().parent}")
    print(f"[app] display:  http://localhost:{args.port}  (project this)")
    print(f"[app] whisper:  {args.whisper_model} on {args.device}")
    uvicorn.run(make_app(args), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
