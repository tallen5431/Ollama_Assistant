# 💬 Ollama Chat

A simple, general-purpose chatbot web app you can plug into the
[HTTP Server Manager](https://github.com/tallen5431/HTTP_Server). The AI model
runs locally on your **desktop** (via [Ollama](https://ollama.com)); this app
runs on the **server manager** and talks to it over your LAN or Tailscale, giving
you a clean chat window you can open from any device.

It follows the same integration conventions as the CodeSmith and InventoryOCR
cards: a `Start.sh` / `Start.bat` launcher, `HOST`/`PORT` from the environment,
[waitress](https://docs.pylonsproject.org/projects/waitress/) serving, a
`/healthz` probe, optional Basic Auth, and reverse-proxy awareness.

## Features

- 🗨️ **Chat window** — a phone-friendly conversation UI with message bubbles.
  The whole conversation is kept and re-sent so the model has context.
- ⚡ **Streaming replies** — the answer appears token-by-token as it's generated,
  with a **Stop** button to cancel a long response mid-stream.
- 📊 **Usage stats** — a small line under each reply shows tokens generated and
  throughput (e.g. *312 tokens · 48 tok/s*), read from Ollama's own counts.
- 🧩 **Reasoning panel** — for models that emit their thinking (e.g.
  `deepseek-r1`, `qwen3`), the scratch-work goes into a collapsible **Show
  thinking** panel instead of cluttering the answer.
- 🎤 **Voice input (offline)** — a mic button transcribes speech to text locally
  with [Vosk](https://alphacephei.com/vosk/) — nothing is sent to the cloud.
  *(Needs the app served over HTTPS; browsers only allow the mic on a secure
  origin. See "Voice input" below.)*
- 🔊 **Read aloud** — a speaker button under every reply reads it out, or turn
  **🔊 Speak replies** on and each finished answer reads itself. Uses the voices
  your browser and phone already have, so there's nothing to download and no
  audio leaves the device. Markdown is turned into something worth hearing
  first: a code fence is named rather than recited, table rows are read as
  cells, and the pipes and asterisks go. See "Read aloud" below.
- 🖼️ **Image input** — attach files with 📎, take a photo with 📷, grab a 📸
  screenshot, or paste one in, then ask about it (`llava`, `*-vision`,
  `minicpm-v`, `qwen2.5vl`, `moondream`, …). Images are downscaled in the
  browser before upload. See "Vision models" below.
- 🚗 **Routines** — save a prompt once and tap it instead of typing. A routine
  can set the toggles it needs and refuse to send until its photos are attached,
  so "two odometer photos → how far did I drive and how long did it take" is two
  taps. Four are shipped. See "Routines" below.
- 🗒 **Records** — a routine can keep a row per run. Two odometer photos become
  *distance 68 mi · elapsed 3h 08m* in a table you can correct and pull out as
  CSV or JSON with `curl`. Fields can say what they hold and which of them are
  arithmetic over the others — those are worked out here rather than by a model
  doing sums in prose, so a rate with nothing to divide by comes out empty
  instead of invented. See "Records" below.
- 💾 **Conversation history** — threads are stored server-side, so one started
  on your desktop continues on your phone. ☰ opens the list; rename, delete,
  reopen. **Turn Basic Auth on if you enable this** — see "Conversation history".
- ✍️ **Formatted replies** — markdown is rendered: labelled code blocks with a
  copy button, lists, quotes, headings, and tables (which scroll sideways on a
  phone rather than stretching the page). Formulas written in LaTeX are unwrapped
  into plain text — there's no maths renderer here, and `68 / 3.13 ≈ 21.7` beats
  a raw `\frac` either way.
- 📍 **Photo details (optional)** — a photo remembers when and where it was
  taken; the picture itself doesn't show it and re-encoding on upload throws it
  away. Turn **📍 Photo details** on and the date, time of day, camera and
  coordinates are read in the browser and sent alongside, so you can ask "where
  was this?" or "was this the morning?". On by default; `PHOTO_META=0` turns
  that round. See "Photo details" below.
- 👁️ **Image reading without losing your model** — attach a screenshot while a
  text-only model is selected and an OCR model transcribes it for you, so a
  stack trace doesn't cost you `qwen3-coder:30b` mid-debug. Only if nothing can
  transcribe does the app switch you to a vision model.
- 🌐 **Web access (optional)** — reads a link you paste *and the pages it links
  to*, or has a small model turn your message into search queries, then grounds
  the answer in what it finds and cites it. Knows today's date, falls back to
  search snippets when a page can't be read, and says so when it found nothing
  rather than answering from memory as if it had. Only public addresses are ever
  fetched. Off by default. See "Web access" below.
- 🧠 **Pick your model** — a dropdown lists every model installed on your Ollama
  server; the configured default is pre-selected.
- 🟢 **Connection status** — a dot shows whether the model server is reachable.
- 🔌 **Server-manager ready** — one-click start, health probe, and works behind
  the manager's reverse proxy.
- 🔒 **Optional login** — turn on HTTP Basic Auth before exposing it publicly.

> **On a phone**, the toggles and routine chips fold away while the on-screen
> keyboard is up, so the conversation keeps the screen. They come back when you
> dismiss it. Nothing folds on a desktop, where no keyboard covers anything.

## Quick start

```bash
./Start.sh
```

`Start.sh` creates a virtual environment, installs `requirements.txt`, and
launches the app with waitress. Open the URL it prints (default
<http://localhost:8070>). On Windows, run `Start.bat`.

Manual run:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-voice.txt   # optional: mic support
OLLAMA_HOST=http://<desktop-ip>:11434 .venv/bin/python app.py
```

Voice support lives in a separate `requirements-voice.txt` because `vosk` only
ships wheels for some platforms. The launchers install it too, but treat a
failure as a warning — the app still starts, just without the mic button.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q
```

The suite covers the auth logic, voice-model path handling, request validation,
and the HTTP layer. It does not require `vosk` or a running Ollama server.

## Configuration

All settings are environment variables (the server manager injects them):

| Variable            | Default                     | Purpose |
| ------------------- | --------------------------- | ------- |
| `HOST`              | `0.0.0.0`                   | Bind address |
| `PORT`              | `8070`                      | Port to listen on |
| `SERVER_THREADS`    | `8`                         | How many requests are handled at once. A chat turn holds a worker for as long as the model takes, so waitress's own default of 4 is low — four slow turns and `/healthz` stops answering, which the server manager's card reads as the app being down |
| `OLLAMA_HOST`       | `http://127.0.0.1:11434`    | Where Ollama runs. Point at your **desktop's** LAN/Tailscale address (a trailing `/v1` is accepted). |
| `OLLAMA_MODEL`      | `llama3.1:8b`               | Default model shown/selected in the UI |
| `OLLAMA_TIMEOUT`    | `300`                       | How long to wait for a reply, in seconds — a 30b legitimately takes minutes |
| `OLLAMA_CONNECT_TIMEOUT` | `5`                    | How long to wait to *connect*, separately. A sleeping desktop drops the packet rather than refusing it, so this is what stops a message hanging for the full reply timeout |
| `OLLAMA_KEEP_ALIVE` | *(Ollama's default)*        | How long the answering model stays in VRAM after a turn, e.g. `30m` to skip a 30b's load time between messages. Helper models always unload immediately |
| `CHAT_IMAGE_TURNS`  | `1`                         | How many recent image-bearing turns re-send their attachments. Raise it if you compare images across turns |
| `PHOTO_READ_EACH`   | `0`                         | Also read each photo on its own before answering, so the `[image n]` numbering is reliable. Costs one model call per photo; only applies with more than one — see "Keeping several photos straight" |
| `PHOTO_META`        | `1`                         | Whether a browser that has never touched the toggle starts with **📍 Photo details** on. `0` makes off the default |
| `PHOTO_KEEP_DAYS`   | `30`                        | How long stored photos stay in the history. Every word is kept for good; only the pixels expire. `0` keeps them for good too |
| `CHAT_TITLE`        | `Ollama Chat`               | Title in the tab/header |
| `CHAT_DB`           | `./chat.db`                 | SQLite file holding conversation history |
| `WEB_VISION_MODEL`  | *(unset)*                   | Model used to read an attached image when planning a search. Unset picks the smallest installed vision model |
| `CHAT_MAX_BODY_MB`  | `25`                        | Maximum accepted request body size (guards the audio upload) |
| `WEB_ENABLED`       | `1`                         | Server-side switch for web access; `0` disables it entirely |
| `SEARXNG_URL`       | *(unset)*                   | Self-hosted SearXNG base URL. Only needed when it is **not** at `http://127.0.0.1:8888`, which is checked for automatically — see `SEARXNG_AUTODETECT`. Set it and it becomes required: a missing instance is then an error rather than a quiet fall back to scraping DuckDuckGo. Check it arrived with `/api/health`'s `search_backend`; under the server manager it is set on the card, and a `.env` in this directory is read by nothing |
| `SEARXNG_AUTODETECT` | `1`                        | With `SEARXNG_URL` unset, look for a working SearXNG at `http://127.0.0.1:8888` — the address `searxng/docker-compose.yml` publishes — and use it if one answers. `0` stops it looking |
| `WEB_PLANNER_MODEL` | *(unset)*                   | Small model used to generate search queries. Unset reuses the answering model; avoid reasoning models here |
| `WEB_DISTILLER_MODEL` | *(unset)*                 | Small model that cuts each fetched page down to what bears on your question. Unset means off — see "Distilling pages". Measured 12,016 → 231 characters on a two-page turn |
| `WEB_MAX_DOCS`      | `3`                         | Pages put in front of the model per turn |
| `WEB_FOLLOW_LINKS`  | `2`                         | How many linked pages may also be read, per hop. Same site by default; `0` disables following everywhere |
| `WEB_FOLLOW_ON_SEARCH` | `1`                      | Whether pages found by *searching* have their links followed too, not just a URL you pasted. `0` restores the old pasted-URL-only behaviour |
| `WEB_MAX_HOPS`      | `1`                         | How far retrieval may follow links outward. `1` is one hop; `2` lets a followed page be followed *from* — the spec linked from the release note linked from the search result. Capped at `3` |
| `WEB_FETCH_HOPS`    | `0`                         | How many times the answering model may ask for a numbered link to be read before it answers. Off by default: each request spends a whole generation that produced no reply — see "Asking to read a link" |
| `WEB_LINKS_IN_CONTEXT` | `25`                     | How many links to list per page, before the context budget trims it. The list is ranked against your question, so this is the ceiling rather than the usual number. `0` turns the list off |
| `WEB_LINK_SCOPE`    | `all`                       | Which links the model is *shown*: `all`, or `site` for same-site only (what it used to be). Nothing on this list is fetched |
| `WEB_FOLLOW_SCOPE`  | `site`                      | Which links may actually be **opened**. Deliberately stricter than what is shown; `any` lifts the same-site restriction |
| `WEB_SHARE_LOCATION` | `1`                        | Whether a photo's GPS position may inform a search query. `0` keeps the date and camera and drops the position — see "Photo details" |
| `WEB_TIMEOUT`       | `15`                        | Per-request timeout when fetching a page or searching |
| `WEB_MAX_CHARS`     | `6000`                      | Text kept from each fetched page |
| `WEB_MAX_BYTES`     | `2097152`                   | Hard cap on a downloaded document |
| `OLLAMA_NUM_CTX`    | `8192`                      | Context window requested when web context is attached |
| `CHAT_AUTH_USER` / `CHAT_AUTH_PASSWORD` | *(unset)* | Enable HTTP Basic Auth (or use `CHAT_AUTH=user:password`). Both halves are needed — with only one set, auth stays **off**, and the startup banner says so rather than leaving you to assume otherwise |
| `CHAT_AUTH_REALM`   | `Ollama Chat`               | Name the browser shows in its password prompt. Only read when Basic Auth is on |
| `VOSK_MODEL`        | `en-us`                     | Default Vosk language id (e.g. `es`, `fr`, `de`) |
| `VOSK_MODEL_PATH`   | *(unset)*                   | Path to an unpacked Vosk model. Used as the default when set (no download). |
| `VOSK_MODELS_DIR`   | `./models`                  | Where downloaded Vosk models are stored |

> Both launchers default `OLLAMA_HOST` to `http://127.0.0.1:11434`. Unless Ollama
> runs on the same machine, point it at your desktop's LAN or Tailscale address
> (e.g. `http://192.168.1.50:11434`) — edit the launcher or set it in the
> program's **env** in the server manager.

## Running the model on your desktop

On the desktop that has the GPU:

```bash
# Install Ollama (https://ollama.com), then pull a model:
ollama pull llama3.1:8b        # or any chat model you like

# Let other machines on your network reach it:
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

Then point this app's `OLLAMA_HOST` at that desktop (its LAN IP or Tailscale
address). The chat traffic stays on your own network.

## Voice input

The mic button uses **Vosk** for fully-offline, on-server speech-to-text. Two
requirements:

1. **HTTPS.** Browsers only grant microphone access on a secure origin. Serve the
   app over HTTPS — the simplest is Tailscale:
   ```bash
   sudo tailscale serve 8070      # gives you https://<machine>.<tailnet>.ts.net
   ```
   Open that `https://…` URL (not `http://…:8070`) and the mic button appears.
2. **A model.** On first use the small English model (~40 MB) downloads once into
   `models/`. If the machine can't reach the internet, download
   [`vosk-model-small-en-us-0.15`](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)
   yourself, unzip it, and point `VOSK_MODEL_PATH` at the folder.

The mic button only shows when the `vosk` package is installed; without it the
app runs exactly as before. Audio is captured in the browser, downsampled to
16 kHz mono, and posted to `/api/transcribe` — it never leaves your network.

**Raw mic (on by default).** 🎙 **Raw mic** is ticked out of the box. It turns
off browser echo cancellation, which exists to stop speaker output leaking into
the mic and has nothing to cancel on headphones — while its residual suppressor
ducks the mic whenever playback is loud, so leaving it on makes you go silent
over music. Untick it on laptop speakers, where the leak is real and cancelling
it helps. (It is named for what it does rather than for when to use it: called
"Headphones", everyone not wearing any unticked it, which turned the very thing
that was spoiling dictation back on.)

**Auto-send.** ⚡ **Auto-send** sends the message as soon as speech is
transcribed, instead of waiting for the Send button. Off by default.

**Continuous.** 🔁 **Continuous** keeps the mic open and treats a pause in
speech as the end of a message, so you don't tap the mic once per sentence.
Tick it together with **Auto-send** and one tap runs a whole conversation:
speak, pause, it sends, and it is already listening for your next turn. Tap the
mic again to stop. A mic left open with nothing said closes itself after a
minute. Off by default; all three toggles are remembered per browser.

Silence detection is energy-based, with the threshold riding on a noise floor
that adapts to the room. A pause of 900 ms ends an utterance and anything
shorter than 300 ms is treated as a noise blip rather than speech. If a reply is
still streaming when you finish talking, the text waits in the box rather than
being dropped.

### Dictating for somewhere else

Offline speech-to-text is useful well beyond talking to a model: plenty of apps
have no dictation of their own, and this page can be the one that does. Dictate
here, then take the words away.

Once there is anything in the box, two buttons appear in the composer beside
the mic:

- 📋 **Copy** — on a phone this opens the **share sheet**, which is the whole
  journey in one tap: pick WhatsApp, Signal, Messages, and the text arrives
  there. No clipboard, and it works on a plain-HTTP page where the clipboard API
  does not exist at all. On a desktop it copies to the clipboard instead. Where
  neither is available it selects the text and tells you to press Ctrl+C —
  because a button that quietly does nothing is worse than one that asks for
  help. Whichever happened is said in the hint line underneath; you should never
  have to find out in the other app.
- 🩹 **Clear** — empties the box for the next one.

Copying deliberately does **not** clear the box. Copying is not a decision to
throw the text away — you might copy it and then send it here too — and a
composer that empties itself unbidden costs a whole dictated paragraph the one
time it is wrong.

Both buttons are hidden while the box is empty, so nothing about the page
changes until you have something to act on.

For this use, leave ⚡ **Auto-send** off — it hands each utterance straight to
the model, which is the opposite of what you want here. 🔁 **Continuous** is
worth turning *on*: it keeps the mic open through pauses, so a long message can
be spoken in several goes and accumulates in the box.

### Background noise

Three things happen to the audio before the recogniser sees it, all of them
because a room is noisier than it sounds.

**Most of a room is below the speech.** A fan, a fridge, traffic through a
window, mains hum, a hand resting on the desk — nearly all of a "quiet" room's
energy sits under 100 Hz, where there is no speech at all. It contributes
nothing to recognition, and it was doing real damage on the way in: the level
the silence detector measures is broadband, so the rumble set the threshold that
your voice then had to clear. Measured, a fan and 50 Hz hum put the floor at
0.017 with 89% of that energy below 130 Hz; speech in the same room reached
0.023 against a threshold of 0.050. Nothing was detected and nothing was sent.
A 4th-order high-pass at 100 Hz now runs first — 24 dB down at 50 Hz, 42 dB at
30 Hz, and flat to within 0.2 dB from 150 Hz up, so a low male fundamental keeps
the harmonics the recogniser actually works from.

**The bar was set too high.** Speech had to be 3× the noise floor — 9.5 dB — to
count. It is now 2×, which is 6 dB. That was picked by sweeping it against a
fan, traffic, a television and a quiet room, each with and without a door
slamming, a chair scraping and someone typing: 2× hears the rooms 3× missed
entirely and still ignores everything 3× ignored, and below 2× a door slam
starts getting sent. Together with the filter, the quietest voice a fan will let
through went from 0.24 to 0.08 — three times quieter — and next to traffic from
0.48 to 0.08.

**What gets sent is the sentence.** An utterance used to be everything captured
since the last one, which in a room the detector never triggered in is a minute
of fan noise with three seconds of speech at the end. The recogniser does not
ignore the rest; it looks for words in it and finds some. Now only the speech is
posted, with 250 ms of lead-in so a soft first consonant is not clipped and
300 ms of tail so the last word finishes. A mic left running in an empty room
posts nothing at all rather than uploading the room. Push-to-talk still sends
everything it recorded — no detector runs, and a deliberate tap is a deliberate
request.

Two honest caveats. A television is the case that stays hard: it is broadband
and speech-shaped, so nothing short of telling two voices apart distinguishes it
from you, and you do have to talk over it. And your browser runs its own noise
suppression before the page sees a sample — measured in Chromium, it already
takes about 15 dB off 50 Hz — so how much the filter adds on top depends on how
good your browser's is. It is there because that varies a great deal between
browsers and phones, and 5 multiply-adds per sample is not a price worth
haggling over.

**Choosing a language.** Next to the mic is a small language picker. It lists the
models already on the server first, then the rest of a built-in catalog (English,
Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese,
Korean, Hindi, and a large English model) marked with their download size. Pick
one that isn't downloaded yet and it's fetched once (into `models/`), then used
for transcription. Set `VOSK_MODEL` to change the default language, or
`VOSK_MODEL_PATH` to use your own model directory.

## Read aloud

Under every reply, next to **Copy**, is a 🔊 **Read aloud** button. Press it and
the answer is read out; press it again — the same button, now **◼ Stop
reading** — and it stops. Above the message box, **🔊 Speak replies** makes
every finished answer read itself, which is what you want with the phone on a
dashboard mount.

This one part doesn't run on your hardware: it uses `speechSynthesis`, the
voices your browser and operating system already ship. So there's nothing to
download and nothing to configure, and it works on the phone and the desktop
today. The text stays on the device — it's handed to the OS speech engine, not
to a server — but the voice is the OS's, not yours. Next to the toggle is a
voice picker listing what your browser has, with the page's own language first;
the choice is remembered per browser. A browser without speech synthesis shows
none of these controls.

**What it actually says.** A reply is written to be *read*, and a voice reciting
the layout is unlistenable. Before speaking, the page turns the markdown into
something worth hearing:

- a fenced code block becomes "(code block)" — reciting `async function paint
  open bracket` helps nobody, but silence leaves you wondering what you missed;
- a table is read row by row as "Leg, Miles. Out, 41." rather than a hedge of
  pipes;
- links are read as their words, not their URLs; images are skipped;
- headings, quotes, bullets, rules and emphasis lose their markers and keep
  their words;
- a paragraph break becomes a full stop, so two thoughts don't run together —
  unless there's one there already;
- numbers, units and money survive all of it.

**Why it's spoken in pieces.** Chrome stops speaking after roughly fifteen
seconds of a single utterance, so a long reply spoken as one would simply stop
part-way through. The text is queued a sentence at a time (and long sentences
broken at punctuation, never mid-word), which also means it can be cancelled
between sentences instead of only at the end.

**When it stops.** On a phone, a voice still going after you've moved on is a
nuisance rather than a feature — so it stops when you ask the next question,
open another conversation, start a new chat, press **Stop**, or switch away
from the tab.

Replies are spoken once finished rather than as they stream: chunking a
half-arrived sentence reads it wrong, and re-reading the corrected version
reads it twice.

## Web access

🌐 **Web access** (off by default) lets the app read the web on the model's
behalf. The model itself never gets a network connection — the app does the
fetching and hands over text.

Two paths:

- **A link in your message is read — and so is what it points at.** Paste a URL
  and ask about it. This is the best case for a local model: one clean document
  beats a pile of search snippets, and it's far better than screenshotting a
  page for a vision model, since the model gets real text instead of OCR'd
  pixels.

  A page is rarely self-contained. A wiki article answers half your question
  and links to the page with the other half, so the app collects the links in
  the page's *readable body* — nav, footers, citations and "edit" links are
  already excluded — and does two things with them:

  1. **A link map** goes to the model as context: what else is covered, and
     where. It's marked as not-read, so the model can say "the hinge page covers
     that" rather than inventing what's on it. Each link is numbered `[n.m]` —
     document `n`, link `m` — so the model can point at one exactly.
  2. **A couple are actually opened.** A small model picks which links look like
     they answer the question, and those pages are fetched too.

  Every URL still goes through the address guard, and by default only same-site
  links may be *opened*. A link is chosen by a model out of content written by a
  stranger, so it gets no more trust than a pasted URL does — following an
  arbitrary outbound link would be a much larger surface for very little gain.
  `WEB_FOLLOW_SCOPE=any` lifts that if you want it. `WEB_FOLLOW_LINKS=0` turns
  the following off; the link map stays.

  **The list is ranked against your question**, not taken in page order. This
  matters more than it sounds: a Wikipedia article has hundreds of links and
  room for a couple of dozen, and the first couple of dozen *in document order*
  are the site's navigation furniture every single time. The link that answered
  the question was reliably somewhere in the ones that got dropped. Ranking is
  lexical — question words against anchor text and URL slug — so it costs no
  model call and runs on every page of every web turn.

  **Links that leave the site are shown too**, tagged `(external)`. They used to
  be dropped, which quietly hid the most useful link on a lot of pages: the
  outside source being cited. Showing one is not fetching it — `WEB_FOLLOW_SCOPE`
  still governs what may be opened — but a model that cannot *see* the link
  cannot tell you where to look next, which is the whole job of the list.
  `WEB_LINK_SCOPE=site` restores the old same-site-only list.
- **Otherwise a planner turns your message into search queries.** A short,
  cheap, deterministic call replies with either `NONE` or up to three `Q: `
  lines — search-engine keywords, each attacking the topic from a different
  angle. The app runs all of them, interleaves the results so every query
  contributes, and reads the top few pages. Using a plain call rather than tool
  calling means this works with *every* model, including vision and reasoning
  ones that expose no tool support.

  The planner sees the last few turns, not just your latest message, so a
  follow-up like *"what about the 14b one?"* still produces a usable query.

  If it replies with something unusable — small models do ignore the format —
  the app searches your message as typed rather than quietly skipping the
  search, which from the outside looks identical to a search that found
  nothing.

  **Pages found this way have their links followed too.** For a long time they
  did not, which left out the case that comes up most: a search lands on the
  overview page and the specifics are one click away, exactly as they are on a
  page you paste by hand. The picker is asked once for the whole turn rather
  than once per page — three results from three sites is one question, and
  asking per page means three model calls, none of which can see that the best
  link on the turn was on result two. `WEB_FOLLOW_ON_SEARCH=0` turns it off.

Both the planner and the answering model are told today's date. A model's sense
of "now" is its training cutoff, which is how *"the latest release"* gets
answered with a version from two years ago and how the planner writes queries
anchored to the wrong year.

### Going deeper

Two settings control how far retrieval travels from where it started, and they
answer different questions.

`WEB_MAX_HOPS` is **the app deciding**. At `1` — the default — the pages first
retrieved may have their links followed once, and there it stops. At `2` a page
reached by following can be followed *from* in turn, which is what finds the
specification linked from the release note linked from the search result. Each
hop is another picker call and another round of fetches, on hardware that is
usually running the answering model at the same time, so it is capped at `3`.

`WEB_FETCH_HOPS` is **the model deciding**, and it is off by default. With it
on, the model that has actually read the pages is told it may reply with:

```
FETCH: [2.3]
```

…and nothing else, naming one of the numbered links. The page is fetched, and
the model is asked again with it in front of it. The request is never shown to
you — it is the machinery talking, not the reply — and it is recognised before
it is displayed, so you don't watch a marker arrive and then get overwritten.

This is the better signal of the two: nothing judges whether a page answered
the question as well as the model trying to answer from it. It is also the more
expensive, because a request spends a whole generation that produced no reply.
On a single-GPU desktop that is the difference between a reply in four seconds
and a reply in twenty, which is why it is opt-in.

The offer is only made while a hop remains **and** there are numbered links to
name, and it is withdrawn as soon as it is spent — a model invited to ask for
something that cannot be delivered just burns a generation. Ask for a number
that was never on the list and the app says so in the panel and asks again
rather than leaving a marker where your answer should be.

However these settings multiply out, retrieval never puts more than **eight**
documents in front of the model. At the permitted maximums the arithmetic
reaches fifteen, and fifteen documents do not fit in any window this app runs
at — each would get its 800-character floor and overrun the context budget
several times over.

### Planner model

By default the answering model does the planning. Set `WEB_PLANNER_MODEL` to a
small model to keep a large one from being invoked twice a turn:

```
WEB_PLANNER_MODEL=qwen3.5:4b
```

Worth measuring rather than assuming: this is only faster if both models fit in
VRAM at once. If they don't, Ollama swaps between them every turn and a small
planner ends up *slower* than just reusing the model already loaded. Leave it
unset if VRAM is tight.

**Reasoning models make poor planners.** Planning is a routing decision with a
short deterministic budget, and a reasoning model spends that whole budget on
its scratchpad — so the reply comes back truncated mid-thought with no query in
it. The app asks Ollama for `think: false` on the planner call and strips any
`<think>` block that arrives anyway, so `deepseek-r1` works; it is still slower
and no better than a small instruct model. Prefer one.

### Distilling pages before they reach the model

A fetched page arrives at up to 6,000 characters, of which the part that
answers you is usually a paragraph. The rest is navigation prose, cookie
notices, related-article rails and the parts of the article about something
else. Three pages is most of an 8,192-token window before your conversation is
even added — measured on a real turn: **16,918 characters of page, one useful
paragraph.**

Set `WEB_DISTILLER_MODEL` to a small model and each page is read by it first,
copying out only the sentences bearing on your question:

```
WEB_DISTILLER_MODEL=qwen3.5:4b
```

Measured on a two-page turn: **12,016 → 231 characters**, with the figures
intact. The saving is the smaller half of the point — the larger half is that
you can raise `WEB_MAX_DOCS` and read six or eight pages for less than three
cost before.

It **extracts rather than summarises** — "copy the sentences that answer this,
word for word" — because a model that paraphrases will eventually paraphrase a
figure wrong, and these come back with a `[1]` after them. An invented number
carrying a citation is worse than no number.

Four things it will not do:

- **Lose a page.** If the distiller can't be reached, times out, returns
  nothing, or returns something that didn't come from the page, that page keeps
  its full text and the panel says how many were kept in full. A summariser
  that can lose information is a worse bug than a long context. The
  came-from-the-page check matters more than it sounds: a 1–4B model handed a
  medical or legal page answers "I'm sorry, I can't assist with that", and that
  is non-empty, so it would otherwise replace six thousand characters of page
  with itself — while the panel reported the loss as a saving.
- **Mix pages up.** Each page keeps its own distillation even when some
  succeed and others fail — a citation pointing at the wrong source is worse
  than no citation.
- **Be trusted.** The distilled text came off an untrusted page and through a
  model that read an untrusted page. It is fenced, numbered and labelled
  exactly like raw page text; a model having touched it makes it no cleaner.
- **Quietly drop a page about something else.** That comes back as *"read, but
  nothing in it bears on the question"*, because a page vanishing silently
  reads as a retrieval failure and sends you looking for a bug that isn't
  there.

Off unless you set it, and the same VRAM caveat as the planner applies: this is
one model call per page, and a model that has to be swapped in can cost more in
latency than the context it saves is worth. **Show what it did** gets a
**Distilled** line with the before and after, so you can judge both the saving
and the quality yourself.

### Looking at a photo before searching for it

An OCR model is the right reader for a screenshot and the wrong one for a
photograph. It used to be preferred either way, so a photo of an animal was
transcribed, found to contain no text, and the search was planned from your
words alone — and "what creature is this", searched verbatim, returns pages of
advertising for animal-identifier apps and nothing about the animal.

Now, when the transcription comes back empty (or as a sentence explaining that
it is empty), something that can see is asked what the photo *shows*, and the
search is planned from that instead. The panel says so: **Looked at the image
instead**.

The describer is your answering model when it can see — it is the best model
you have, it is about to read the image anyway, and Ollama already has it
loaded, so this costs one more pass rather than one more model in VRAM. A
screenshot that transcribes to real text is never read twice.

### When a page can't be read

Paywalls, JS-only pages and dead links are the normal case, not the exception.
A result whose page can't be fetched still contributes its **search snippet**,
labelled in the context as a summary rather than the page, so the model treats
it as a lead instead of established fact. Results are also capped at two per
host, so three angles on one topic don't come back as three pages of the same
site wearing three hats.

If retrieval produces nothing at all, the model is told so explicitly and asked
to say it couldn't check. A failed search that reads like a successful one is
the worst outcome for a feature whose whole point is not guessing.

Progress appears live above the reply ("Searching for…", "Reading example.com…")
and the pages used are listed underneath, numbered to match the `[n]` citations
the model is asked to use.

### Checking it works

The test suite covers this against stand-in servers. Two things a stand-in
cannot check are whether DuckDuckGo's HTML still parses and whether link
extraction finds anything on a real page, so:

```bash
.venv/bin/python tools/check_web.py
.venv/bin/python tools/check_web.py https://en.wikipedia.org/wiki/Ada_Lovelace
```

It exercises search, fetch, decoding, link extraction and context assembly, and
prints what it found. Nothing in it talks to Ollama, so a sleeping desktop
doesn't matter — run it when a web answer looks wrong and it will tell you
which half is at fault.

Being offline is also its blind spot. Whether a small model actually *replies*
in the two shapes link-following depends on — bare numbers from the picker, a
bare `FETCH: [n.m]` from the answering model — can only be answered by asking a
real model, so that has a checker of its own. This one does wake the desktop:

```bash
.venv/bin/python tools/check_links.py
.venv/bin/python tools/check_links.py URL "a question the page does not answer"
.venv/bin/python tools/check_links.py URL "question" --model qwen3:8b
```

It fetches the page, shows the link order the model will see and which links
ranking promoted, asks your real picker model and reports whether its reply
parsed, then asks your real answering model and classifies what came back —
a well-formed request, a request for a link that doesn't exist, a malformed
attempt, or a direct answer. It also checks that the numbering shown and the
numbering resolved agree, which is the one failure that would silently fetch
the wrong page.

**Choose the question deliberately.** The interesting case is one the page
*mentions* but doesn't answer, where a link plainly would — that is what the
feature is for. If the page answers it outright, a direct answer is the correct
result, and the tool says so rather than marking it wrong.

The two `❌`s worth acting on: a picker whose reply parses to nothing means
following will silently never happen on that model (use a small non-reasoning
model for `WEB_PLANNER_MODEL`), and a malformed `FETCH` means the marker would
reach the user instead of an answer (leave `WEB_FETCH_HOPS` at `0` on that
model).

### Search backend

With no configuration, search uses DuckDuckGo's HTML endpoint. It needs no key
and it works until it doesn't: there is no API here, only a page meant for a
person, and DuckDuckGo is entitled to decide a given address is not one.

**When it stops, this is what it looks like.** The panel under the reply says
something like:

> `html.duckduckgo.com served a bot challenge (DuckDuckGo — Unfortunately, bots
> use DuckDuckGo too. Please complete the following challenge to confirm this
> search was made by a human…)`

That is not a bug to be fixed here and not something waiting reliably clears.
Your address has been flagged, and a captcha is not a thing a program can
answer. (If instead it says *"returned a page with no results in it"* with no
challenge, that **is** a bug here — DuckDuckGo moved its markup, and it needs
fixing in `web.py`.)

Both endpoints report separately, and both being challenged means the address
is flagged rather than one endpoint being unlucky. Run your own search — below.
Note that SearXNG runs on the *same address*, so DuckDuckGo will refuse it
there too; what fixes the search is the other engines answering, which is the
whole reason to put a metasearch engine in front. When a SearXNG search comes
back with nothing, the app now names the engines that refused it, so "nobody
has written about this" and "everything I asked is refusing me" stop looking
the same.

### Running your own search — the durable fix

[SearXNG](https://docs.searxng.org/) is a metasearch engine: it queries the
engines for you and answers with JSON, so this app never scrapes anyone's HTML.
It runs on the same box as everything else and needs no key. A compose file and
a working config are in [`searxng/`](searxng/):

```bash
cd searxng
mkdir -p config && cp settings.yml.example config/settings.yml
sed -i "s|ultrasecretkey|$(python3 -c 'import secrets;print(secrets.token_hex(32))')|" config/settings.yml
grep secret_key config/settings.yml   # must be 64 hex characters, not "ultrasecretkey"
docker compose up -d
```

`python3` rather than `openssl` on purpose: `openssl` is not installed
everywhere, and `$(openssl …)` on a box without it expands to nothing and the
`sed` silently writes an empty key. Check the `grep` output rather than
trusting the command — SearXNG refuses to start on the default key, and the
container then restart-loops with the reason only in `docker logs searxng`.

**`config/` is the directory the container is given, and it must not be one git
manages.** SearXNG chowns whatever it is handed to its own user — that is what
`cap_add: CHOWN` is for — so an earlier version of these instructions, which
mounted the checkout itself, ended with the container owning tracked files.
The symptoms are a long way from the cause:

```
sed: couldn't open temporary file ./sedv5wFbs: Permission denied
error: unable to unlink old 'searxng/settings.yml': Permission denied
```

If you are coming from that version, **remove the container first**, then take
the directory back. That order matters and is not obvious: a container with
`--restart unless-stopped` that is failing to start is restarting every few
seconds, and each start chowns the directory again — so a `chown` on its own
appears to do nothing at all. Nothing is lost, since `config/settings.yml` is
generated from the example:

```bash
docker rm -f searxng
sudo chown -R "$USER:$USER" ~/HTTP_Server/projects/ollama_assistant/searxng
rm -rf searxng/config searxng/instance     # leftovers of the old mount
```

If a `git pull` already failed against the unwritable directory, it will have
left the working tree half-updated — git writes the files it can and stops at
the first it cannot, without moving HEAD. The next pull then refuses with
*"Your local changes to the following files would be overwritten by merge"*,
naming files you never touched: they are the new content, written by the pull
that could not finish. Once the directory is yours again, take the branch tip
whole:

```bash
git reset --hard origin/<your-branch>
```

Safe here because none of those changes are yours, and untracked files — your
`.env` among them — are not touched by it.

The steps above then work as written — and their order is load-bearing too.
`config/settings.yml` has to exist *before* the container starts: given an
empty directory it tries to create one from its own template, which it cannot
do in a directory owned by you, and it exits saying
`"/etc/searxng/settings.yml" is not a valid file`. Given the file, it takes
ownership of the directory and starts.

If it still cannot write there, hand the directory over explicitly — `config/`
is gitignored, so it is free to own it:

```bash
docker run --rm --entrypoint id docker.io/searxng/searxng:latest   # its uid/gid
sudo chown -R <uid>:<gid> searxng/config
docker restart searxng
```

One consequence worth knowing: once it is running, `config/settings.yml`
belongs to the container's user, so editing it later needs `sudo` — and if you
`chown` it back, stop the container first or it will simply take it again.

`config/` is gitignored, so updates never touch your key and the container
never touches your checkout.

`docker compose` is the v2 plugin, which Ubuntu's `docker.io` package does not
include. If that line fails — `unknown shorthand flag: 'd' in -d`, or
`docker: unknown command: docker compose` — you have Docker without it.

**Do not reach for the hyphenated `docker-compose`.** That is v1, it has been
end-of-life since 2023, and against any current Docker Engine it fails on the
recreate path with:

```
KeyError: 'ContainerConfig'
```

Docker no longer emits `ContainerConfig` in image inspect and v1 reads it
unconditionally, so this is not something in the compose file and not something
you can configure around. It bites the second time you run it, not the first,
which makes it look like the file broke rather than the tool.

So: install the plugin (`sudo apt install docker-compose-v2`, or
`docker-compose-plugin` from Docker's own repository), or skip compose
altogether. This is the same container, spelled out, and depends on nothing but
`docker` itself:

```bash
docker run -d --name searxng --restart unless-stopped \
  -p 127.0.0.1:8888:8080 \
  -v "$PWD/config:/etc/searxng:rw" \
  -e SEARXNG_BASE_URL=http://localhost:8888/ \
  --cap-drop ALL --cap-add CHOWN --cap-add SETGID --cap-add SETUID \
  --log-driver json-file --log-opt max-size=1m --log-opt max-file=1 \
  docker.io/searxng/searxng:latest
```

Run that from inside `searxng/`, so `$PWD/config` is the directory holding the
`settings.yml` you generated above. If a previous attempt left a container behind — `docker run`
will say `the container name "/searxng" is already in use` — clear it first
with `docker rm -f searxng`; nothing in it is worth keeping, since all of the
state is the `settings.yml` you are mounting in.

Check it came up with `docker logs searxng` — a container that exits
immediately is almost always the secret key still being the default.

Then come back up to the project root and check it end to end, without opening
the app (from there, because that is where the virtualenv is):

```bash
cd ..
export SEARXNG_URL=http://127.0.0.1:8888
.venv/bin/python tools/check_web.py
```

That names the backend it used and prints the results it got, so a pass means
the whole path works.

**You should not have to set `SEARXNG_URL` at all** when you run the instance
above. With it unset, the app checks `http://127.0.0.1:8888` — the one address
`searxng/docker-compose.yml` publishes — and uses it if a search there actually
works. Not a scan: one loopback address, checked once every few minutes, and a
refused connection costs microseconds. It looks for a *working* instance rather
than an open port, so the three ways a fresh SearXNG does not work (HTML only,
the limiter on, something else on that port) are all declined rather than
adopted. `SEARXNG_AUTODETECT=0` stops it looking.

The startup banner says when it found one:

```
  Web search   : http://127.0.0.1:8888 (found running; SEARXNG_URL is unset)
```

Set `SEARXNG_URL` explicitly when your instance is somewhere else, or when you
want it to be an error if it is missing — an address you configured fails
loudly, while one that was merely found falls back to DuckDuckGo without
comment.

**And note that `export` only reaches this shell**, which is the step people
miss when they do set it: the check passes while the app carries on scraping
DuckDuckGo, and it looks like the app ignoring a setting rather than never
receiving one.

Under the HTTP Server Manager, the environment is the card's own. The manager
runs `bash Start.sh` with `{...process.env, ...program.env}`, and `program.env`
is the per-program `env` object in its `config.json`. So:

1. **Edit** this program in the manager UI
2. Under **Environment Variables**, add `SEARXNG_URL` = `http://127.0.0.1:8888`
3. Save, then **Restart** the card

or add it to that program's `env` in `config.json` directly, alongside
`HOST` and `PORT`:

```json
"env": {
  "HOST": "0.0.0.0",
  "PORT": "8070",
  "SEARXNG_URL": "http://127.0.0.1:8888"
}
```

> **A `.env` file in this directory does nothing.** Nothing reads one — not the
> manager, not `Start.sh`, not the app. It is an easy and invisible place to
> put a setting that will never arrive, and it looks like every other project
> where that would have worked.

Running the app directly instead of through the manager, an `export` in the
same shell is all it takes:

```bash
SEARXNG_URL=http://127.0.0.1:8888 .venv/bin/python app.py
```

The app says which backend it has on startup, so you can tell the two apart at
a glance:

```
  Web search   : http://127.0.0.1:8888
```

against

```
  Web search   : DuckDuckGo (no SEARXNG_URL — see the README)
```

**Two settings do all the work**, and a stock SearXNG has both wrong for this:

| Symptom | Cause | Fix |
| --- | --- | --- |
| `SearXNG returned HTTP 403` | It serves HTML only out of the box | add `json` under `search: formats:` |
| `SearXNG returned HTTP 429` | The bot limiter is on | set `server: limiter: false` |

Both are already set in `searxng/settings.yml`, and the app names the fix in
the error if you are using your own config. The limiter exists to stop a
*public* instance being scraped by deciding whether a caller looks like a
browser; this app isn't one, and an instance on loopback has nobody to keep
out.

The compose file binds to `127.0.0.1` deliberately. An open SearXNG is
something other people find and use, and nothing needs to reach it but the app
on the same machine — you already get to the box over Tailscale.

`SEARXNG_URL` is allowed to point at a private address, unlike anything a model
or a page asks for; see "What it will and won't reach" below.

### What it will and won't reach

Every URL is resolved and checked before it is fetched, and **only public
addresses are allowed**. Loopback, LAN, link-local, cloud metadata and the
Tailscale `100.64/10` range are all refused, so a link in a message — or a
redirect chosen by a remote server, which is re-checked at every hop — can't
turn this into a probe for your own network. The single exception is
`SEARXNG_URL`, which is operator configuration rather than something a model or
a page chose.

The address is checked twice: once on the name before the request, and once on
the socket after it connects. A nameserver can answer differently the second
time it is asked — public when the guard checks, `127.0.0.1` when the
connection is made — and the second check is where that stops, before any of
the request has been sent. (A configured HTTP proxy is the exception: the
socket then goes to the proxy, so its address says nothing, and the name check
is doing the work on its own.)

Retrieved text is fenced and labelled as reference material, with an explicit
instruction to ignore any directions inside it. That reduces prompt injection
but does not eliminate it — so nothing here can *act*. There are no tools with
side effects; the only thing a page can do is be read. Treat answers grounded in
a web page with the same suspicion you'd treat the page itself.

`WEB_ENABLED=0` disables the whole feature server-side, whatever the UI says.

### Context window

An attached page is large, so a wider window is requested (`OLLAMA_NUM_CTX`,
default 8192) whenever web context is present. Ollama otherwise defaults to a
modest window regardless of what the model supports, and the conversation would
be silently pushed out of it.

## Conversation history

Threads are stored in SQLite (`CHAT_DB`, default `./chat.db`) rather than in the
browser, so one started on a desktop can be picked up on a phone. The ☰ button
opens the list; conversations are named from their first message and can be
renamed or deleted. **New chat** starts a fresh thread.

A conversation is only created once you send something, so idly opening the app
doesn't litter the list — and a turn that produces nothing takes its thread away
again.

**The reply is written down by the server, not the browser.** That matters on a
phone: lock the screen past the wake lock, switch apps, or hand off from wifi to
cellular in the middle of a long answer, and the tab that asked the question is
no longer there when the answer arrives. The model also keeps generating once
nobody is listening — a WSGI response is *pulled*, so tying the work to the
connection meant a dropped phone kept only the first sentence. Reopen the app
and the whole answer is waiting in the thread. **Stop** still stops it: the
button tells the server as well as closing the connection, and keeps whatever
had been said by then.

**And you don't have to reopen it.** Switch tabs or apps while the model is
thinking and the connection often goes with you — the radio sleeps, the OS
suspends the tab, Tailscale comes back on a different path. The page used to
call that a network error: it took your question off the screen, put it back in
the composer, and deleted the thread it had just made, which is the one action
that actually loses the answer, since that is the thread the reply was about to
be written into.

Now the question stays where it is, the bubble says *"Lost the connection — the
reply is still being written"*, and the page goes and collects it. It asks the
server whether that turn is still running (`GET /api/chat/status`), waits, and
re-reads the thread when the answer lands — with its reasoning, its steps and
its sources, by the same path that shows a phone's turn on a desktop. Coming
back to the tab makes it look again straight away rather than sitting out a
backoff. If the turn really did die, it says so and leaves your question on
screen to send again.

**Search** looks inside the messages and the records, not just the titles — the
box is at the top of the ☰ list. Typing "A23" finds the thread where that only
ever appeared in a reply; typing "Brighton" also turns up the trip a routine
wrote down. Escape or the ✕ clears it.

> ⚠️ **Anything stored here is readable by anyone who can reach the app.** Until
> now there was nothing to steal; with history on, your past conversations are
> on disk and served to whoever asks. If you keep history, turn Basic Auth on:
> ```
> CHAT_AUTH=you:a-long-random-passphrase
> ```
> `WEB_ENABLED=0` and an unset `CHAT_DB` are unrelated — history has no separate
> kill switch, so auth is the control.

Attached images are stored with their message so a reopened thread still makes
sense — along with their photo details, if you had that on, so reopening a
thread and asking "where was that?" gets the same answer as asking it the first
time. Which means an image-heavy history grows the database. The drawer shows
what it currently costs, and deleting conversations genuinely gives the space
back — the file is compacted when enough of it has become dead space, rather
than only marking pages free.

**Photos expire; words do not.** The browser caps a re-encode at about 900KB,
which is roughly 1.2MB of base64 in the row, so a two-photo routine run every
day is most of a gigabyte a year — and what is worth having a year later is the
reading that came off the photo, not the photo. After `PHOTO_KEEP_DAYS` (default
30) the pixels are dropped and everything else stays: the question, the reply,
the record, and the photo's own note of when and where it was taken. A thread
whose photos have gone says so rather than showing a gap. The sweep runs itself
at most once an hour; **Free up space now** in the drawer applies it on demand.
`PHOTO_KEEP_DAYS=0` keeps every photo for good.

Only the most recent image-bearing turn re-sends its attachments to the model.
Re-uploading every screenshot in a thread on every turn was slow over a phone
connection and rarely what was meant; earlier turns keep their text, so the
conversation still reads. Set `CHAT_IMAGE_TURNS` higher if you compare images
across turns.

### Keeping several photos straight

A routine with two photos asks a model to do something the input does not
support. The pictures arrive as pixels with **no labels attached to them**,
while the details beside them say "Image 1", "Image 2" — so using a capture
time means aligning two lists across two messages by position, and then joining
each time to the odometer read out of the matching picture. Nothing in the
input anchors that join. It is a *binding* problem rather than a hard one,
which is why it fails on large models as readily as small ones.

Two things address it, and they stack:

- **Take the times off the file** (above). A field declared `= earliest photo
  taken` never goes near a model, so the commonest version of this join simply
  stops existing. This is on wherever a routine declares it, and the shipped
  🚗 Trip routine does.
- **`PHOTO_READ_EACH=1`** reads each photo separately *first*, the way this app
  has always read photos for models without vision, and hands the answering
  model the labelled readings alongside the pictures. The join becomes text to
  text, matched on a number — a thing models are reliably good at. The pictures
  stay, and the preamble tells the model to trust its own eyes where a reading
  disagrees. Off by default because it costs a model call per photo, and skipped
  for a single photo, which has nothing to be confused with.

The photo-details block counts the same photos. It labels its lines "Image 1",
"Image 2", and the pictures carry no labels of their own — so the numbering is
only true if it counts exactly what the model is about to see. It used to
describe the newest turn alone: at `CHAT_IMAGE_TURNS=3` with two photos a turn,
"Image 1" pointed at the *third* photo on screen and every time in the answer
belonged to a different picture.

## Vision models

There are three ways to attach an image, up to four per message:

- **📎 Attach** — pick image files.
- **📸 Screenshot** — capture a window, tab, or whole screen. The browser's own
  picker decides what is shared; a single frame is taken and the capture stream
  is dropped immediately, so nothing is recorded. Desktop only — the button is
  hidden where `getDisplayMedia` isn't supported, which includes mobile browsers.
- **📷 Camera** — take a photo there and then. Shown only on touch devices,
  where it opens the rear camera directly rather than the gallery; on a desktop
  it would just duplicate 📎.
- **Paste** — paste an image anywhere on the page, usually the quickest route
  for a screenshot taken with the OS shortcut.

Two different jobs, deliberately given to different models:

**Answering about an image.** If the selected model can see, it just answers.

If it can't, and an OCR model is installed, the image is transcribed and the text
is handed to the model you're already on — so a screenshot of a stack trace
doesn't cost you `qwen3-coder:30b` in favour of a 3B generalist at exactly the
moment you're debugging. The hint line says which model is doing the reading.

Only when there's no transcriber installed does the app switch models, picking
the *smallest* general vision model — silently loading a 19 GB model because you
pasted a screenshot is a poor surprise. You can always change the dropdown.

If the OCR pass finds no text at all — a photo of a plant, not a screenshot —
a general vision model describes the image instead, so you get an answer rather
than advice to change a dropdown. Only if that also comes back empty does the
reply say so and suggest picking a vision model, rather than inventing one. A
reader model that could not be *reached* is reported as exactly that, never as
a claim about what was in your picture.

**Reading text out of an image**, when **Web access** is on. Before the search is
planned, the image is transcribed so the exact error text, product names and
version numbers reach the query. Without it, a screenshot plus "what's this?"
plans nothing worth running.

An **OCR model is preferred for that second job** if one is installed
(`glm-ocr`, `got-ocr`, and similar are recognised by name). Transcription is
precisely what makes a good search query, where a general vision model
paraphrases the error away. OCR models are *excluded* from the first job for the
same reason — they transcribe rather than reason, so they make poor
conversationalists. `WEB_VISION_MODEL` pins the reader if you'd rather choose.

Capability comes from what Ollama reports in `/api/tags` — the `clip` and
`mllama` families, an explicit `vision` capability — with a name-based fallback
for builds whose details block is sparse.

All three paths handle the image identically. Anything up to 1920 px on the long
edge is kept at its original size and encoded as **PNG** — losslessly, so the
small text in a screenshot stays legible, which is usually the entire reason for
sending one. A 1920×1080 screenshot costs about 60 KB that way.

Only when PNG exceeds ~1.5 MB is the image treated as a photograph: downscaled to
1920 px and encoded as JPEG (dropping a quality step if it is still very large),
with transparency flattened onto white since JPEG has no alpha. EXIF orientation
is honoured throughout, so photos aren't sent sideways.

Re-encoding also removes every other trace of EXIF — the date, the camera, the
GPS position — which is why "Photo details" above exists and why it has to run
before this step rather than after.

They travel as base64 in the message, exactly as Ollama's native API expects:

```json
{ "role": "user", "content": "what is this?", "images": ["<base64>"] }
```

Note the whole conversation is re-sent on every turn, so an image stays in the
context for the rest of the chat — **New chat** clears it.

## Routines

A routine is a prompt you save once and tap instead of typing. It sits as a chip
above the message box; tapping it drops its text into the composer, sets the
toggles it asks for, and holds **Send** until the photos it expects are attached.

The case it was built for: photograph the odometer at the start of a trip and
again at the end, tap **🚗 Trip**, attach both, send.

> **you:** *(taps 🚗 Trip, attaches two photos)*
> **model:** Image 1 reads 041233 km, taken 09:04. Image 2 reads 041589 km, taken 17:32. That's 356 km over 8 h 28 min — an average of 42 km/h.

The distance comes from the pictures; the times come from the photos' own EXIF,
which is why **🚗 Trip** turns **📍 Photo details** on for you. Both photos have
to go on **one** message, because only the most recent image-bearing turn keeps
its attachments — the photo-count guard is what makes that automatic instead of
something you have to remember.

**Setting one up.** ☰ → **Routines** → **＋ New routine**, or the ⚙ at the end of
the chip strip. A routine has:

| | |
| --- | --- |
| **Name** | what the chip says — keep it short, an emoji helps you find it |
| **Prompt** | the text that goes into the box |
| **Photos to attach** | 0–4. Send is refused below this, with a count |
| **📍 Photo details** | force on, force off, or leave your toggle alone |
| **🌐 Web access** | the same three choices |

The two forcings are genuinely three-state. "Leave as it is" is the right answer
for most routines — a routine that has no opinion about the web shouldn't be
made to state one. A forcing lasts for **one turn**: the toggles go back where
they were once the message is sent, or as soon as you tap the lit chip again.

**Four to start with**, added by **☰ → Routines → ＋ Add the starter routines**
(nothing is installed until you ask):

- **🚗 Trip** — two odometer photos → distance, elapsed time, average speed
- **📊 Before / after** — two photos of the same thing → what changed
- **📄 Read this** — transcribe a label, receipt or serial number verbatim
- **✍️ Plain words** — explain something jargon-free in under 200 words

They're ordinary routines: edit them, rename them, delete them. Deleting one and
pressing **＋ Add the starter routines** again brings it back.

**What a routine actually is**, mechanically: the text becomes *your* message.
It goes into the composer where you can read and edit it before sending ("this
was the rental, it reads km"), and what the model receives is exactly what the
bubble shows and what the history stores. That also means the routine's text is
re-sent with every later turn of that thread, like anything else you type —
**＋ New chat** clears it.

> ⚠️ A saved routine is an instruction that gets delivered as though you had
> typed it. Anyone who can reach this app can add or edit one. That's the same
> exposure conversation history has, and the same answer applies: if this is
> reachable by anything you don't control, turn Basic Auth on.

One combination is worth knowing about: a routine that attaches photos **and**
forces web access on sends the photo's coordinates to the search planner, whose
queries go out to a search engine. The editor says so at the moment you pick it,
`WEB_SHARE_LOCATION=0` prevents it, and no shipped routine does it.

## Records

A routine can keep a record of every run. Give it field names and each run
writes a row you can look at, correct, and export.

Values are stored as plain text. Models reach for LaTeX the moment an answer
contains arithmetic, and `3 hours and 8 minutes (or $\approx 3.13$ hours)` is
not a spreadsheet cell — so the extraction is asked for plain text and strips
any that arrives anyway. Money is left alone: `$54.20` and `$5 to $10` are what
a receipt routine records, and reading those as arithmetic would be the worse
failure.

The trip case: **🚗 Trip** with `distance, elapsed, average speed`. Photograph
the odometer twice, tap the chip, send — and under the reply a line appears:

```
🗒 Kept: distance 68 miles · elapsed 3 h 08 min · average speed 21.7 mph
```

☰ → **Records** opens the log over the whole window — not in the drawer. A
drawer is where you pick something; records is a table you read, filter,
correct and export, and it is the only part of the app that wants the width. In
a 21rem drawer it was eight columns in 320px. The composer goes away while it
is open, since there is nothing to type at, and the back arrow (or Escape)
returns you to the conversation.

The table, newest first:

| When | Routine | distance | elapsed | average speed |
| --- | --- | --- | --- | --- |
| 07/08/26, 16:45 | 🚗 Trip | 68 miles | 3 h 08 min | 21.7 mph |

**How the fields are found.** The model that just answered is asked to restate
its own answer as those fields, as JSON — temperature 0, a hard token cap, and
nothing invented: a field the answer didn't state comes back empty rather than
guessed. Extraction rather than pattern-matching, because the answers are prose
and no regex survives the next model.

It is good, not beyond question, and the app doesn't pretend otherwise: **every
cell is editable**. Tap it, type, and it saves. A log you can't correct is one
you stop trusting.

It never costs you an answer. The extraction runs after the reply has finished,
not inside the stream, so a model that's asleep or that returns something
unparseable costs you a row and nothing else. Nothing extractable means no row,
rather than a blank one — a blank record is worse than none.

### Saying what a field holds

A field can be a bare name, as it always was — read it off the answer and work
out what it is from the column. It can also say what it holds, and it can say
that it is arithmetic over the others:

```
Start odometer: distance
End odometer: distance
Distance traveled = End odometer - Start odometer
Start time: timestamp
End time: timestamp
Elapsed time = End time - Start time
Total earnings: money
Earnings per mile = Total earnings / Distance traveled
Earnings per hour = Total earnings / Elapsed time
Average speed = Distance traveled / Elapsed time
```

Ten columns, and **the model is asked for five**. The rest are worked out here,
in Python, from those five.

That is not a tidiness argument, it is a correctness one. A real log had one
trip recorded twice, five minutes apart: `$26.23` an hour and then `$23.19` an
hour, for the same 93 miles and the same $115.94. The second capture had no
start or end time in it *at all* — so its hourly rate had been worked out from
nothing. The model was never asked to divide; it was asked what the answer
said, and it obliged. **Now that field comes out empty**, and the table says
why when you hover it: *"Nothing to work it out from — needs Total earnings /
Elapsed time"*.

`name: kind` takes any of **money, distance, speed, duration, timestamp,
number, text**. `name = a op b` takes `-`, `+`, `*`, `/` over two other fields,
and a computed field can build on one declared above it — the hourly rate
divides by an elapsed time that was itself computed from two timestamps.

### Times come from the file, not from the model

```
Start time = earliest photo taken
End time   = latest photo taken
```

A field declared that way is filled straight from the photo's own EXIF, and
**the model is never asked for it**.

This one is worth explaining, because it looks like a model failing at an easy
job and it isn't. The capture times reach a model as a block of text saying
*"Image 1: taken Friday 07 August 2026 at 13:37"* — while the photos themselves
arrive as pixels in a different message, carrying no labels at all. To use a
time, the model has to align two lists across two messages by position and then
join each time to the odometer it read out of the matching picture. That join
has no anchor in the input, so it is a *binding* problem rather than a hard
one — which is exactly why large models get it wrong too, and confidently.

The app never had that problem: your browser reads the EXIF before the image is
re-encoded, so the exact time is already in hand. It was being rendered to
prose, read back by a model, rewritten as prose, and parsed again — four hops
for a figure that started out exact, and two of them can invent.

`photo 1 taken` picks by position; `earliest`/`latest photo taken` pick by the
recorded instant, which is what a trip actually wants — a gallery hands photos
over in whatever order it likes, and the later one is the end of the trip
whichever slot it landed in.

Where the file records no time — a screenshot, an edited copy, or **📍 Photo
details** switched off — the field is empty and anything built on it is empty
too, with a note saying so. And where the times carry no zone (EXIF very often
records none), the elapsed time is still worked out, with a note that it is out
by whole hours if the clock moved in between. That caveat used to live in the
routine's prompt; it now lives where the arithmetic does.

The shipped **🚗 Trip** routine uses this. It declares seven fields and asks the
model for **two** — the two odometer readings, which is the only part of the job
that needs eyes.

A declared kind also beats the column vote. Inference is a good guess across a
column and a good guess is still a guess; it also cannot work on the *first*
row of a new routine, where there is no column to look at yet.

Two things it will not do, for the same reason the standardiser will not:

- **Invent a figure.** Missing input, or a divide by zero, gives an empty cell
  and a note saying which input was missing — never a number.
- **Force a value into its declared kind.** A `Total earnings` that reads
  `unknown` is reported and *left as it is*. Overwriting it with a blank would
  throw away the one thing it told you, and leave a log that looks complete.

Everything written before this still works: a list of bare names is a list of
untyped read fields, which is exactly what it always meant. The shipped
**🚗 Trip** routine now ships with the declaration above, as a worked example.

### One shape per column

The fields are written by whichever model answered, in whatever words it
reached for that day. A real log had one trip recorded twice, five minutes
apart, agreeing about every fact and about none of the formatting:

```
"102,072"    "102,072 mi"      "100,409 miles"
93           93 mi             66 miles
$1.2465      $1.24 per mile    $0.55 per mile ($36.00 / 66 mi)
21.2 mph     ≈ 23.21 mph       21.70 MPH
```

Every one of those is correct and none of them sorts against the row above it.
So a value is put into a standard shape on its way in, and records kept before
that existed are rewritten once at startup.

**The rule: the presentation is standardised, the number never is.** `$1.2465`
does not become `$1.25` and `$36.00` does not become `$36` — rounding is a
change to the data. A minus sign and a leading decimal point are part of the
figure, not decoration: `-$12.50` stays negative and `.5 mi` is half a mile.
What comes off is only decoration: a thousands separator, a repeated unit, an
"approximately", a bracket showing the working.

For any value that changed, **the model's own wording is kept alongside it**. A
standardised cell is underlined with a faint dotted line in the Records table,
and hovering it (or long-pressing on a phone) says what it used to read — so
the tidy-up can always be checked against what it replaced rather than taken on
faith.

Two things it deliberately will not do:

- **Rewrite prose.** A value that merely *contains* a number is a sentence.
  "54 miles to Brighton" stays exactly that; turning it into "54 mi" would
  delete where the trip went and take the word Brighton out of search with it.
- **Guess a unit from one value.** Whether a bare `93` is ninety-three miles is
  a fact only its column knows, so the column votes: a single value that names
  its unit settles it for the rest, and a column that is mostly notes stays
  notes.

Timestamps come out as `2026-08-25 20:06 UTC-04:00` — sortable, and keeping the
offset, which is the one part of a timestamp you cannot recover by looking at it
again. A stated date with no time stays a date rather than being padded to
midnight, which would read as a measurement rather than a gap.

### Checking a log

Standardising a value is safe, so it happens on its own. Changing a *number* is
not, so nothing does it for you:

```bash
.venv/bin/python tools/check_records.py
.venv/bin/python tools/check_records.py --csv trips.csv
.venv/bin/python tools/check_records.py --routine "🚗 Uber Trip"
```

It reports three things and edits nothing. **Rows that disagree with their own
arithmetic** — the derived fields are a model doing sums in prose, and a model
given no duration will still produce an hourly rate. **The same run recorded
twice.** And **a preview of what standardising would change**, so you can see it
before it happens.

Run against the log those samples came from, it found the same trip logged
twice at $26.23/hour and $23.19/hour — the second row had no start or end time
in it at all, so its rate had been worked out from nothing.

**Getting it out.** Two links in the Records pane, and both are ordinary
endpoints, so another machine can pull them:

```bash
curl -s http://nucbox:8070/api/records.csv > trips.csv
curl -s http://nucbox:8070/api/records | jq '.records[].fields'
curl -s 'http://nucbox:8070/api/records.csv?routine=🚗%20Trip'
```

The CSV timestamps are ISO 8601, so a spreadsheet and a database both parse
them, and the columns are the union of every field across the log — a routine
whose fields changed doesn't lose the older runs' data.

**The unit goes in the header and the cell holds the number alone** —
`Total earnings (USD)` with `115.94` under it, `Distance traveled (mi)` with
`93`. A cell reading `$115.94` or `93 mi` is *text* to a spreadsheet: it will
not sum, it will not chart, and it sorts `100 mi` before `93 mi`. Timestamps
and free text are written out as they stand, because a number would say less
than they do.

Records outlive the routine that made them. Deleting **🚗 Trip** doesn't touch
a year of trips; the routine's name is copied into each row rather than looked
up, so the log stands on its own.

> ⚠️ Records are stored in the same `chat.db` and are readable by anyone who can
> reach the app — the same exposure conversation history has, and the same
> answer: turn Basic Auth on if this is reachable by anything you don't control.

## Photo details

A photo carries a small record of its own making — when the shutter fired, which
camera, and on a phone usually where you were standing. None of that is in the
picture, so no vision model can tell you, however good it is at seeing.

It also doesn't survive the upload. Every image is re-encoded through a canvas
before it's sent, which produces clean pixels and nothing else: by the time an
attachment reaches the server the metadata is already gone. So it has to be read
in the browser, from the original file, before that step — which is what the
**📍 Photo details** toggle turns on.

With it on, ask the ordinary question:

> **you:** where was this taken, and roughly what time?
> **model:** Tuesday 14 July 2026, about half six in the evening, at 51.510000, -0.127500 — Westminster, London.

What the model is actually given is one line per photo:

```
- Photo: taken Tuesday 14 July 2026 at 18:42 (evening) (UTC+01:00), at
  51.510000, -0.127500 (give or take 8 m), 42 m above sea level, on a Google
  Pixel 8 (Pixel 8 back camera), 4080×3072 (13 MP) as taken, 1/120 s, f/1.8,
  6.7 mm, 24 mm equivalent, ISO 64, no flash
```

What gets read, when the photo has it:

| | |
| --- | --- |
| **When** | date and time taken, the time-zone offset, and the GPS clock in UTC |
| **Where** | latitude, longitude, altitude, and how far out the fix might be — both the recorded position error and the DOP, which is the satellite geometry at the time |
| **Motion** | speed and the direction the camera was pointing |
| **Camera** | make, model and lens; the pixel dimensions as taken |
| **The shot** | shutter speed, aperture, focal length (and 35 mm equivalent), ISO, flash |
| **The file** | orientation, the software that wrote it, artist and copyright |

Anything missing is simply absent. It's rendered into a sentence rather than
passed as raw fields, with the day of the week and the time of day named,
because "was that the morning?" is the question people actually ask.

The time zone is worth calling out. EXIF capture times carry none of their own,
so two photos either side of a zone change are hours apart in a way nothing
downstream can detect. Where the camera recorded an offset it's passed through;
where it didn't but there's a GPS fix, the GPS clock is UTC and gives the same
answer the long way round.

### Seeing what was read

**Tap a thumbnail** before you send and a sheet lists everything pulled out of
that file, with the position linking to a map. That's the quickest way to answer
"does this photo actually have a location in it?", and it needs no terminal.

The thumbnail itself carries a summary: **🕘** a date was read, **📍** a location
was, **📍̸** the camera asked for a fix and didn't get one, and a faint **·** for
a file with neither. When a position is missing the app says why in the hint
line, because the three reasons need different fixes.

To check a file from the command line — including one that never reached the
app — point the diagnostic at it. It runs the app's own parser, so what it
prints is exactly what the app would read:

```
.venv/bin/python tools/check_photo.py ~/Downloads/odometer.jpg
.venv/bin/python tools/check_photo.py --raw photo.jpg     # every tag, by number
```

`--raw` dumps each IFD entry before any interpretation. That's the one worth
having when a photo demonstrably carries a position and the app reports none:
it shows whether the GPS block is there and what's in it, which settles whether
the file is unusual or the parser is wrong.

The usual reasons a photo has no location:

- **The camera never recorded one.** On Android that's Camera → Settings →
  Location, and the camera app also needs the location permission; on iPhone,
  Settings → Privacy → Location Services → Camera. This is by far the most
  common answer, and the tool will say so — if the date and camera read fine,
  the file is intact and simply has no position in it.
- **It was stripped in transit.** Messaging apps re-encode, and Google Photos'
  *share* link is not the original file — use *download* instead.
- **It's a screenshot.** Those have no EXIF at all.
- **The camera asked and got nothing.** A GPS block with no usable fix in it —
  0, 0, which is the Gulf of Guinea — is common indoors and in a garage. It's
  reported as *no position*, not as Null Island, but it's a different answer
  from "the setting is off": wait for a fix rather than change a setting.

A position that *is* there can still be poor. Where the file records a position
error or a DOP, both are passed on in words — "give or take 8 m; a good fix,
DOP 1.2" — so the model qualifies an answer rather than reading six decimal
places as a doorstep.

**On by default**, because on a box only you can reach — over Tailscale, say —
that's the useful setting. Set `PHOTO_META=0` on the server and a fresh browser
starts with it off instead. Either way the toggle is yours to flip, and your
choice is remembered per browser; the server default only applies to a browser
that has never touched it. The page starts it *off* and lets `/api/health` turn
it on rather than the reverse, so `PHOTO_META=0` has no window in which a
position is read anyway.

**Web search is the one place it can leave the house.** The planner turns your
message into queries and those go to DuckDuckGo or your SearxNG. The planner
itself runs on your own hardware, so it's told the photo's date, camera and
position — "what's this building?" is a much better search when the file says
you were at 51.51, -0.13. But what it writes is sent onward, so a query *could*
carry your location. If that's not what you want:

```
WEB_SHARE_LOCATION=0     # the planner keeps the date and camera, loses the position
```

The answering model gets the full version regardless — turning this off shapes
the search, not the answer. And a photo only informs the search on the turn it
was attached to; last week's photo stops steering today's queries.

Where it goes:

- to the model you're chatting with, on your own machine, as one system turn;
- to the search planner, also on your machine, as a short note — see above;
- into the conversation history, so reopening the thread and asking again works;
- nowhere else.

Camera names are treated as untrusted text like anything else the app didn't
write — folded to one line, capped, and fenced — since a file can claim any
make it likes.

### Choosing a microphone

The **🎙 Voice** row has a microphone picker beside the language. The browser's
default is chosen by the operating system and is regularly not the one you are
speaking into — a laptop offers the webcam's, a desktop offers whatever the
monitor came with. A chosen device is demanded rather than preferred, so it is
honoured or the attempt fails loudly; one that has been unplugged falls back to
the default and says so.

Beside it is a level meter, live while recording, with a mark that sticks at
the loudest moment. "It transcribed it wrongly", "it heard nothing" and "it was
clipping" are three different problems that look identical without one, and a
silent recording now says which rather than only that no speech was found.

Audio reaches the recogniser as 16 kHz mono. Where the browser will give a
16 kHz capture context — most of them will — that is what it records and no
resampling happens at all. Where it will not, the downsample is a windowed-sinc
low-pass measured at **-65 dB** across everything above the 8 kHz Nyquist, with
the speech band flat to 0.01 dB. The box average it replaced measured -13 dB,
which left most of a 12 kHz sound folded down onto a 4 kHz vowel.

### When the search stops working

Unset, `SEARXNG_URL` means scraping `html.duckduckgo.com` and, if that yields
nothing, `lite.duckduckgo.com`. Both are asked as an ordinary browser: a
User-Agent that announces itself as a tool gets served an empty result page,
which is indistinguishable from a query that matched nothing — and that is
exactly how a broken search became an answer saying it could not check
anything against a source.

Results are found by their `/l/?uddg=` redirector when the class names no
longer match — that redirector is the product, the class names are decoration
and get reshuffled. An endpoint returning a page with no results in it is
reported as a fault rather than as an empty list, and the panel shows the first
line of whatever did come back. A query that genuinely matches nothing still
comes back as nothing, because those are different answers.

The panel distinguishes the two failures by name, because they have opposite
answers:

- **"served a bot challenge"** — the address has been flagged and is being
  asked to prove it is a person. Nothing in this repo can answer a captcha.
  This is the end of the road for scraping; run SearXNG.
- **"returned a page with no results in it"** — the markup moved. That is a bug
  here, in `_duck_results` and `_DUCK_ENDPOINTS`, and the gist in the panel is
  the evidence for fixing it.

Scraping is fragile by nature and this is what it looks like when it gives out.
If web access matters to you, run SearXNG and point `SEARXNG_URL` at it — it is
a JSON API that does not change under you. See "Running your own search" above;
the compose file is in `searxng/`.

## Seeing what a turn did

Under every reply, next to **Show thinking**, is **Show what it did** —
collapsed, because on a turn that went fine it is noise. Open it and the turn
is on the record, in order:

```
Photo details    1 photo(s) carried their own record; 937 characters given
                 to the model                                        [show]
Images           1 in the thread; qwen3-vl:30b reads them itself
Read the image   minicpm-v (description)                             [show]
Planned searches hosyond 3.5 320x480 arduino screen
Search results   0 from 1 query group(s)                             [show]
Sent to the model  3 turns, 7412 characters of text, num_ctx 8192    [show]
```

The `[show]` toggles reveal the exact text — the metadata block, the
transcription, the URLs, the system turns the model was actually given.

Both this panel and **Show thinking** are stored with the reply, so a chat
started on a phone still has them when it is opened on a desktop.

This exists because two very different failures look identical from the
outside. A reply saying "I cannot read image metadata" while **📍 Photo
details** is ticked is either the photo carrying no EXIF (a screenshot, or a
copy that stripped it) or the model ignoring what it was handed — and the panel
says which. The same goes for a web answer that hedges: the search may never
have been planned, may have been planned from too little, or may have run and
found nothing, and only the last of those means the retrieval worked.

## API endpoints

| Method & path        | Purpose |
| -------------------- | ------- |
| `GET /`              | Chat UI |
| `GET /healthz`       | Plain `ok` health probe (stays open even when auth is on) |
| `GET /api/health`    | JSON status (Ollama host and whether it answers, default model, auth + voice on/off, and `search_backend` — the SearXNG URL this process actually has, or `"duckduckgo"`) |
| `GET /api/models`    | Installed models (proxy to Ollama `/api/tags`) |
| `POST /api/chat`     | Chat completion. Streams `{"debug": {"step", "detail", …}}` lines alongside the reply — what the turn did, for the panel under it. Streams NDJSON by default; pass `{"stream": false}` for a single JSON reply. Body: `{ "model"?, "messages": [...], "conversation_id"? }` or `{ "prompt": "..." }`. Given a `conversation_id` the server writes the finished turn into that thread itself, and keeps generating even if the client disappears — that stream also sends a blank line every 20 s while the model is quiet, so skip empty lines rather than parsing them (both readers in the page already do). Messages may carry `"images": ["<base64>"]` for vision models, and `"image_meta": [{...}]` alongside it — one entry per image, `{"taken","lat","lon","altitude","camera"}`, all optional. |
| `POST /api/chat/cancel` | Stop the turn running for a conversation — body `{ "conversation_id" }`. Needed because generation outlives the connection |
| `GET /api/chat/status` | Is a turn still being generated for this conversation? `?conversation_id=` → `{"running": true\|false}`. Lets a page whose connection dropped tell "still coming" from "the server died", which otherwise look identical |
| `GET /api/search`    | Find a phrase across conversations and records — `?q=`. Returns `{ "conversations": [...], "records": [...] }`; under two characters returns both empty |
| `POST /api/photos/forget` | Apply the photo retention now — body `{ "days"? }`, defaulting to `PHOTO_KEEP_DAYS`. Returns what it dropped |
| `GET /manifest.webmanifest` | Web app manifest, so a home-screen shortcut opens without browser chrome |
| `GET /api/routines`  | Every saved routine, in strip order |
| `POST /api/routines` | Save one — body `{ "name", "body", "photos"?, "web"?, "photo_meta"? }`. `web`/`photo_meta` are `true`/`false`/`null`, where null means "leave the toggle alone" |
| `POST /api/routines/starters` | Install the shipped routines, skipping names already taken. Idempotent |
| `PATCH /api/routines/<id>` | Change any subset of those fields. An absent key leaves it alone; an explicit `null` clears a forcing |
| `DELETE /api/routines/<id>` | Remove one |
| `GET /api/records`   | Kept records, newest first, plus the union of their columns. `?routine=` narrows it |
| `POST /api/records`  | Restate an answer as fields and keep it — body `{ "answer", "fields": [...], "routine_name"?, "routine_id"?, "conversation_id"?, "model"? }`. Returns `{"record": null}` when nothing could be pulled out, which is an outcome rather than an error |
| `PATCH /api/records/<id>` | Correct fields — body `{ "fields": {...} }`. Merges, so an edit never drops a column it didn't mention |
| `DELETE /api/records/<id>` | Remove one |
| `GET /api/records.csv` | The whole log as CSV, ISO timestamps, `?routine=` narrows it |
| `GET /api/voice/models` | Available + downloadable Vosk speech models |
| `POST /api/voice/download` | Download a catalog model — body `{ "id": "fr" }`. Streams NDJSON progress (`{"downloaded","total","percent"}`) then the final object; an unknown id is a 400 before any of it |
| `POST /api/transcribe` | Speech-to-text: POST WAV audio (`?model=<id>` optional), returns `{ "text": ... }` |

## Expose it to the internet (optional)

The app ships with **no authentication**. Before putting it on a public tunnel,
turn the login on:

```bash
export CHAT_AUTH_USER="tj"
export CHAT_AUTH_PASSWORD="a-long-random-passphrase"
# or the shorthand:  export CHAT_AUTH="tj:a-long-random-passphrase"
```

Then expose it over HTTPS with a Tailscale Funnel or Cloudflare Tunnel (both
terminate TLS for you). `/healthz` stays open for uptime probes.

## Project layout

| File | Role |
| ---- | ---- |
| `app.py`          | Flask routes (streaming chat, transcribe) + waitress entrypoint |
| `config.py`       | Environment + logging helpers |
| `ollama_client.py`| Thin HTTP client for the Ollama API (incl. streaming) |
| `chat_ui.py`      | Single-page chat UI (inline HTML/CSS/JS) |
| `authz.py`        | Optional HTTP Basic Auth |
| `voice.py`        | Offline speech-to-text with Vosk |
| `web.py`          | Optional web fetching/search used to ground answers |
| `store.py`        | SQLite conversation history, routines and records |
| `records.py`      | Restating a routine's answer as the fields it declared |
| `tests/`          | pytest suite (no Ollama or `vosk` needed) |
| `Start.sh` / `Start.bat` | Launchers for Linux / Windows |

## Updating a checkout

```bash
git pull
./Start.sh          # reinstalls requirements if they changed
```

If `git pull` reports **"You have divergent branches"** on a checkout you have
not committed to, or fetches a branch you were not expecting, the clone is
probably shallow and single-branch — which is what `git clone --depth=…` gives
you, and what several tools clone with by default. Two symptoms, one cause:

* `git log` shows `(grafted)` next to the newest commit. Without the older
  history git cannot find a common ancestor, so an ordinary fast-forward is
  reported as a divergence.
* `git branch -r` lists one remote branch, and `origin/<anything-else>` "does
  not exist" however many times you fetch. A `--depth` clone sets a refspec
  that only ever fetches the branch it was cloned with.

Both are fixed once, and neither touches your working tree:

```bash
git fetch --unshallow origin                                    # get the history
git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
git fetch origin                                                # get the branches
git branch --set-upstream-to=origin/$(git rev-parse --abbrev-ref HEAD)
git merge --ff-only @{u}
```

`--unshallow` on an already-complete repository is an error rather than a
no-op; if you get that, skip it and run the rest. After this, `git pull` does
the obvious thing.
