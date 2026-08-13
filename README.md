# live-commentary

MVP of Alexander's proposal (ILIAD intensive): a room conversation is recorded,
transcribed live, and an LLM occasionally projects a short text comment onto a
screen.

Pipeline: mic (`parecord`) or recording (`ffmpeg`) → 16 kHz PCM → ASR →
speaker labeling (the speaker with the most airtime is `LECTURER`, others are
`AUDIENCE-n` — no enrollment needed) → at
each opportunity Claude sees the labeled rolling transcript + its previous
comments and replies `PASS | reason` or one ≤25-word comment → FastAPI page
(SSE) styled for projection.

Claude calls fire **continuously**: whenever any new words have arrived
(`--min-new-words`, default 1) and `--call-gap` seconds (default 1) have
passed since the last call — even mid-speech. Calls never overlap, so during
dense speech the effective cadence is bounded by the API latency (~3–4 s). Right before each call the partially-filled audio chunk is flushed
through whisper, so the prompt includes the freshest words, and the reply
**streams to the screen word by word** as it is written (withheld until it is
clear the reply isn't a PASS). Terminal logs every reply including PASSes with
reasons, so you can see why it stays silent.

Every comment opens with a `> quoted words` line showing the transcript words
it responds to (rendered as a small quote above the comment), and if the room
addresses Claude directly it may answer at length (~80 words) instead of the
usual ≤25. Its name in the room is **Marginalia** (it answers to "Claude" too).
Transcript lines carry wall-clock timestamps that Claude sees (for judging
recency) but never quotes. Comments may use LaTeX (`$...$`), rendered with
KaTeX. Vote tallies from `/?grade` are fed back into its prompt ([2↑ 1↓]) so
it can calibrate to the room.

## Run

```sh
uv run app.py                         # live mic (ANTHROPIC_API_KEY from your shell)
uv run app.py --chatty                # lower the commentary bar (demos, testing)
uv run app.py --wav lecture.mp3       # demo against a recording (any av format)
uv run app.py --wav "https://www.youtube.com/watch?v=..."  # YouTube, as if live
# in YouTube mode the display page embeds the video next to the commentary,
# click-to-start, seeked to the feed position and playing at --speed
# (YouTube caps playback at 2x; above that it resyncs by jumping forward)
uv run app.py --wav talk.mp3 --speed 4 --call-gap 5   # fast replay
uv run app.py --mock                  # no API key, canned comments
```

## ASR backends

`--asr auto` (default) picks **Deepgram nova-3 streaming** when
`DEEPGRAM_API_KEY` is set — true streaming with word-level diarization, smart
formatting, and filler words — and otherwise falls back to **local
faster-whisper** (~7 s chunks on the GPU, ECAPA-embedding speaker clustering,
private, no cloud). Force either with `--asr whisper` / `--asr deepgram`.
Deepgram reconnects automatically on network blips. Neither backend tags
laughter/applause; ElevenLabs Scribe has audio events but no streaming
diarization yet, so paralinguistics are limited to nova-3's filler words.

## Views

- **`/margin`** — the margin-notes experiment: the transcript rendered as
  de-emphasized textbook body text (press `t` to fade it to a ghost), and
  Marginalia's comments drawn live in the right margin, each anchored to the
  quoted words (underlined in the text; hover a note to highlight its
  anchor). Notes accumulate down the page like annotations in a used
  textbook, so the room can read them asynchronously instead of racing the
  lecture. A pencil ✎ hovers in the margin while a call is in flight; PASSes
  are silent. Press `f` to cycle the note handwriting font (default: CMU
  Serif italic, the 3b1b font; also `?font=Name`); `?grade` adds pencil-mark
  vote buttons under each note. The commentator prompt leans into the same
  framing: durable margin annotations, not reactions to the last sentence.
- **http://localhost:8710** — projection page: the current comment large on
  the left, and the **full chat history** on the right (`?nofeed` hides it):
  the labeled transcript with Marginalia's comments backfilled in place, plus a
  notch at each point the transcript was sent to Claude, resolving to
  `commented` or `pass · reason` — the room can see exactly what it is
  thinking about. Late joiners get the recent conversation replayed in
  order. A QR code in the corner (hide with `/?noqr`) points phones at the
  LAN address's `/?grade` view, so scanners can vote immediately; the URL is
  also printed at startup.
- **On phones** the same page becomes a scrollable chat feed (current comment
  on top, full conversation below); add `?grade` for grading buttons.
- **`/?chime`** — adds a soft two-note chime on each comment (click the page
  once to unlock audio).
- **`/?ops`** — operator pane on the right: pipeline stage, trigger state
  (new words / time to next call — i.e. *why Claude isn't commenting*),
  recent PASSes with reasons and latencies, live chattiness controls
  (strict / chatty / eager, effective on the next call); spawned web
  searches log here too.
- **`/?grade`** — 👍/👎 buttons on every comment (also present in `?ops`);
  open it on a phone to grade during a lecture. Live vote tallies show next
  to the buttons on every device. After voting, an optional
  "why?" note can be typed — it is **private to Marginalia** (fed into its
  prompt alongside the vote tallies, never shown on any screen) and lands
  in the session log.

A crashed pipeline thread shows red in the page corner and a traceback in the
terminal; while a Claude call is in flight the corner shows *thinking…*.

## Knobs

| flag | default | meaning |
|---|---|---|
| `--chattiness` | `strict` | `strict` / `chatty` / `eager`; retunable live from `/?ops` (`--chatty` = `chatty`) |
| `--context` | — | text file (abstract, curriculum, notes) given to the commentator as background |
| `--asr` | `auto` | `deepgram` (streaming diarization; needs `DEEPGRAM_API_KEY`) or local `whisper` |
| `--whisper-model` | `distil-large-v3` | `small.en` is lighter; both fly on the GPU (~0.2–0.5 s / 7 s chunk, ≤2 GB VRAM) |
| `--device` | `auto` | cuda when available (nvidia pip wheels; app re-execs once to set `LD_LIBRARY_PATH`) |
| `--chunk-sec` | 7 | transcription chunk length (latency vs. context) |
| `--no-speakers` | off | disable LECTURER/AUDIENCE labeling |
| `--call-gap` | 1 | min seconds between Claude calls (calls never overlap) |
| `--min-new-words` | 1 | skip the Claude call if less new speech than this |
| `--claude-model` | `claude-opus-5` | |
| `--effort` | `medium` | Claude reasoning effort — `low` is a latency lever |
| `--fast` | off | Opus fast mode: ~2.5× generation speed at 2× price (~2–4¢/call); cuts time-to-first-words without the quality cost of `--effort low` |

The commentator system prompt is `COMMENTATOR_SYSTEM` at the top of `app.py`
(with `CHATTINESS_ADDENDA` per chattiness level).

## The web-search agent

Marginalia always sees the **full session transcript** (affordable because
the transcript is sent as immutable 40-line blocks with a prompt-cache
breakpoint — each call reads the prefix from cache at ~0.1× price and pays
full rate only for new speech). When the room explicitly asks it to look
something up ("Marginalia, search for…", "chat, look up…"), it replies
`SEARCH | <query>`: a **web-search agent** is spawned in the background (a
separate Claude call with the server-side `web_search` tool) and its report
is injected into the next commentary turn. The feed notch shows
`🔎 search agent · <query>` while it runs; the answer lands as a normal
comment a turn later. It only searches when explicitly asked.

## Session logs

Every run appends to `sessions/<timestamp>.jsonl`: the run's settings, every
transcript line, every Claude reply (comments with time-to-first-words and
total latency; PASSes with reasons), live config changes, grades, and thread
crashes with tracebacks — enough to replay or grade any demo after the fact.

Cost: each call is one small Opus call (~1–3k tokens in, ~50 out), about
1–2¢. At the default continuous cadence (~one call per 3–4 s of dense
speech) a 2-hour session ≈ $30–70; raise `--call-gap` (e.g. 10) and
`--min-new-words` (e.g. 30) for the older, cheaper cadence.

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
- Latency budget per comment ≈ trigger poll (≤1 s) + flush (~1 s) + Claude's
  time-to-first-words (thinking; ~2–4 s at `medium` effort, less at `low`) —
  the rest of the comment streams in as it is written.
