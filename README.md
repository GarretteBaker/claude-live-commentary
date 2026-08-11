# live-commentary

MVP of Alexander's proposal (ILIAD intensive): a room conversation is recorded,
transcribed live, and an LLM occasionally projects a short text comment onto a
screen.

Pipeline: mic (`parecord`) or recording (`ffmpeg`) → 16 kHz PCM →
faster-whisper (~7 s chunks, local GPU: distil-large-v3 by default) → at each
opportunity Claude sees the rolling transcript + its previous comments and
replies `PASS | reason` or one ≤25-word comment → FastAPI page (SSE) styled for
projection.

Commentary opportunities fire on a **lull in speech** (≥4 s pause — the comment
lands when eyes can go to the screen) or at most every `--comment-interval`
seconds, and only if `--min-new-words` new words arrived. Terminal logs every
Claude reply including PASSes with reasons, so you can see why it stays silent.

## Run

```sh
uv run app.py                         # live mic (ANTHROPIC_API_KEY from your shell)
uv run app.py --chatty                # lower the commentary bar (demos, testing)
uv run app.py --wav lecture.mp3       # demo against a recording (any av format)
uv run app.py --wav talk.mp3 --speed 4 --comment-interval 8   # fast replay
uv run app.py --mock                  # no API key, canned comments
```

Open / project **http://localhost:8710** — or `http://localhost:8710/?chime`
for a soft two-note chime on each comment (click the page once to unlock audio).
A crashed pipeline thread shows red in the page corner and a traceback in the
terminal.

## Knobs

| flag | default | meaning |
|---|---|---|
| `--chatty` | off | demo mode: comment on anything substantive, not just the best openings |
| `--whisper-model` | `distil-large-v3` | `small.en` is lighter; both fly on the GPU (~0.2–0.5 s / 7 s chunk, ≤2 GB VRAM) |
| `--device` | `auto` | cuda when available (nvidia pip wheels; app re-execs once to set `LD_LIBRARY_PATH`) |
| `--chunk-sec` | 7 | transcription chunk length (latency vs. context) |
| `--lull-sec` | 4 | speech pause that triggers an early commentary opportunity |
| `--comment-interval` | 30 | max seconds between opportunities |
| `--min-new-words` | 30 | skip the Claude call if less new speech than this |
| `--claude-model` | `claude-opus-5` | |
| `--effort` | `medium` | Claude reasoning effort (latency lever) |

The commentator system prompt is `COMMENTATOR_SYSTEM` at the top of `app.py`
(with `CHATTY_ADDENDUM` for `--chatty`).

Cost: each opportunity is one small Opus call (~1–3k tokens in, ~50 out), about
1–2¢. A 2-hour session at default cadence ≈ $2–5.

## TODO

- **Speaker detection (diarization)**: attribute transcript lines to speakers
  (e.g. pyannote or whisperX on the same GPU) so the commentator can say "A's
  objection concedes B's earlier premise" — the connect-two-remarks
  interventions are weak without it.
- **Subject context**: give the model the material ahead of time — curriculum,
  lecture notes, session abstract — as a cached system-prompt block (e.g.
  `--context notes.md`), so comments can reference definitions and results the
  room has already seen rather than reconstructing them from the transcript.

## Known limitations (MVP)

- Chunked transcription, no streaming ASR: words at chunk boundaries can be
  mangled; effective latency ≈ chunk + trigger + API call (a few seconds each).
- No diarization — the transcript doesn't attribute remarks to speakers.
- Whisper hallucinates on silence/music; RMS gate + VAD mitigate, not eliminate.
- On casual banter the default prompt PASSes essentially always — that's by
  design; use `--chatty` if you want it talkative.
