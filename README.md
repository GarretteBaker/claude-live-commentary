# live-commentary

MVP of Alexander's proposal (ILIAD intensive): a room conversation is recorded,
transcribed live, and an LLM occasionally projects a short text comment onto a
screen.

Pipeline: mic (`parecord`) or recording (`ffmpeg`) → 16 kHz PCM →
faster-whisper (~7 s chunks, local GPU: distil-large-v3 by default) → speaker
labeling (ECAPA embeddings, greedy online clustering; the cluster with the most
airtime is `LECTURER`, others are `AUDIENCE-n` — no enrollment needed) → at
each opportunity Claude sees the labeled rolling transcript + its previous
comments and replies `PASS | reason` or one ≤25-word comment → FastAPI page
(SSE) styled for projection.

Commentary opportunities fire on a **lull in speech** (≥4 s pause — the comment
lands when eyes can go to the screen) or at most every `--comment-interval`
seconds, and only if `--min-new-words` new words arrived. Right before each
Claude call the partially-filled audio chunk is flushed through whisper, so the
prompt includes the freshest words. Terminal logs every Claude reply including
PASSes with reasons, so you can see why it stays silent.

Comments may open with a `> quoted words` line showing what they respond to
(rendered as a small quote above the comment), and if the room addresses Claude
directly it may answer at length (~80 words) instead of the usual ≤25.

## Run

```sh
uv run app.py                         # live mic (ANTHROPIC_API_KEY from your shell)
uv run app.py --chatty                # lower the commentary bar (demos, testing)
uv run app.py --wav lecture.mp3       # demo against a recording (any av format)
uv run app.py --wav "https://www.youtube.com/watch?v=..."  # YouTube, as if live
# in YouTube mode the display page embeds the video next to the commentary,
# click-to-start, seeked to the feed position and playing at --speed
# (YouTube caps playback at 2x; above that it resyncs by jumping forward)
uv run app.py --wav talk.mp3 --speed 4 --comment-interval 8   # fast replay
uv run app.py --mock                  # no API key, canned comments
```

## Views

- **http://localhost:8710** — clean projection page.
- **`/?chime`** — adds a soft two-note chime on each comment (click the page
  once to unlock audio).
- **`/?ops`** — operator pane on the right: pipeline stage, trigger state
  (new words / lull / next interval — i.e. *why Claude isn't commenting*),
  recent PASSes with reasons and latencies, and live chattiness controls
  (strict / chatty / eager, effective on the next call).
- **`/?grade`** — 👍/👎 buttons on every comment (also present in `?ops`);
  open it on a phone to grade during a lecture. Each grade appends a line to
  `grades.jsonl` with the comment and the transcript context it responded to.

A crashed pipeline thread shows red in the page corner and a traceback in the
terminal; while a Claude call is in flight the corner shows *thinking…*.

## Knobs

| flag | default | meaning |
|---|---|---|
| `--chattiness` | `strict` | `strict` / `chatty` / `eager`; retunable live from `/?ops` (`--chatty` = `chatty`) |
| `--context` | — | text file (abstract, curriculum, notes) given to the commentator as background |
| `--whisper-model` | `distil-large-v3` | `small.en` is lighter; both fly on the GPU (~0.2–0.5 s / 7 s chunk, ≤2 GB VRAM) |
| `--device` | `auto` | cuda when available (nvidia pip wheels; app re-execs once to set `LD_LIBRARY_PATH`) |
| `--chunk-sec` | 7 | transcription chunk length (latency vs. context) |
| `--no-speakers` | off | disable LECTURER/AUDIENCE labeling |
| `--lull-sec` | 4 | speech pause that triggers an early commentary opportunity |
| `--comment-interval` | 30 | max seconds between opportunities |
| `--min-new-words` | 30 | skip the Claude call if less new speech than this |
| `--claude-model` | `claude-opus-5` | |
| `--effort` | `medium` | Claude reasoning effort — `low` is the main latency lever |

The commentator system prompt is `COMMENTATOR_SYSTEM` at the top of `app.py`
(with `CHATTY_ADDENDUM` for `--chatty`).

Cost: each opportunity is one small Opus call (~1–3k tokens in, ~50 out), about
1–2¢. A 2-hour session at default cadence ≈ $2–5.

## Known limitations (MVP)

- Chunked transcription, no streaming ASR: words at chunk boundaries can be
  mangled; effective latency ≈ chunk + trigger + API call (a few seconds each).
- Speaker labels are heuristic: overlapping speech is unhandled (embeddings of
  voice mixtures are garbage), segments under ~0.8 s inherit the previous
  label, and early segments can be mislabeled before airtime statistics
  accumulate. Disable with `--no-speakers`.
- Whisper hallucinates on silence/music; RMS gate + VAD mitigate, not eliminate.
- On casual banter the default prompt PASSes essentially always — that's by
  design; raise chattiness (flag or `/?ops`) if you want it talkative.
- Latency budget per comment ≈ up to `--chunk-sec` of buffered audio (flushed
  early at trigger time) + `--lull-sec` + the Claude call (2–6 s at `medium`
  effort, less at `low`).
