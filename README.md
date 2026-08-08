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
- 🖼️ **Image input** — attach files with 📎, take a photo with 📷, grab a 📸
  screenshot, or paste one in, then ask about it (`llava`, `*-vision`,
  `minicpm-v`, `qwen2.5vl`, `moondream`, …). Images are downscaled in the
  browser before upload. See "Vision models" below.
- 🚗 **Routines** — save a prompt once and tap it instead of typing. A routine
  can set the toggles it needs and refuse to send until its photos are attached,
  so "two odometer photos → how far did I drive and how long did it take" is two
  taps. Four are shipped. See "Routines" below.
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
| `OLLAMA_HOST`       | `http://127.0.0.1:11434`    | Where Ollama runs. Point at your **desktop's** LAN/Tailscale address (a trailing `/v1` is accepted). |
| `OLLAMA_MODEL`      | `llama3.1:8b`               | Default model shown/selected in the UI |
| `OLLAMA_TIMEOUT`    | `300`                       | How long to wait for a reply, in seconds — a 30b legitimately takes minutes |
| `OLLAMA_CONNECT_TIMEOUT` | `5`                    | How long to wait to *connect*, separately. A sleeping desktop drops the packet rather than refusing it, so this is what stops a message hanging for the full reply timeout |
| `OLLAMA_KEEP_ALIVE` | *(Ollama's default)*        | How long the answering model stays in VRAM after a turn, e.g. `30m` to skip a 30b's load time between messages. Helper models always unload immediately |
| `CHAT_IMAGE_TURNS`  | `1`                         | How many recent image-bearing turns re-send their attachments. Raise it if you compare images across turns |
| `PHOTO_META`        | `1`                         | Whether a browser that has never touched the toggle starts with **📍 Photo details** on. `0` makes off the default |
| `CHAT_TITLE`        | `Ollama Chat`               | Title in the tab/header |
| `CHAT_DB`           | `./chat.db`                 | SQLite file holding conversation history |
| `WEB_VISION_MODEL`  | *(unset)*                   | Model used to read an attached image when planning a search. Unset picks the smallest installed vision model |
| `CHAT_MAX_BODY_MB`  | `25`                        | Maximum accepted request body size (guards the audio upload) |
| `WEB_ENABLED`       | `1`                         | Server-side switch for web access; `0` disables it entirely |
| `SEARXNG_URL`       | *(unset)*                   | Self-hosted SearXNG base URL. Unset falls back to DuckDuckGo HTML |
| `WEB_PLANNER_MODEL` | *(unset)*                   | Small model used to generate search queries. Unset reuses the answering model; avoid reasoning models here |
| `WEB_MAX_DOCS`      | `3`                         | Pages put in front of the model per turn |
| `WEB_FOLLOW_LINKS`  | `2`                         | How many pages linked from a URL you pasted may also be read. Same site, one hop; `0` disables it |
| `WEB_SHARE_LOCATION` | `1`                        | Whether a photo's GPS position may inform a search query. `0` keeps the date and camera and drops the position — see "Photo details" |
| `WEB_TIMEOUT`       | `15`                        | Per-request timeout when fetching a page or searching |
| `WEB_MAX_CHARS`     | `6000`                      | Text kept from each fetched page |
| `WEB_MAX_BYTES`     | `2097152`                   | Hard cap on a downloaded document |
| `OLLAMA_NUM_CTX`    | `8192`                      | Context window requested when web context is attached |
| `CHAT_AUTH_USER` / `CHAT_AUTH_PASSWORD` | *(unset)* | Enable HTTP Basic Auth (or use `CHAT_AUTH=user:password`) |
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

**Headphones (on by default).** Next to the language picker is a 🎧 **Headphones**
tickbox, ticked out of the box. It turns off browser echo cancellation, which
exists to stop speaker output leaking into the mic and has nothing to cancel on
headphones — while its residual suppressor ducks the mic whenever playback is
loud, so leaving it on makes you go silent over music. Untick it on laptop
speakers, where the leak is real and cancelling it helps.

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

**Choosing a language.** Next to the mic is a small language picker. It lists the
models already on the server first, then the rest of a built-in catalog (English,
Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese,
Korean, Hindi, and a large English model) marked with their download size. Pick
one that isn't downloaded yet and it's fetched once (into `models/`), then used
for transcription. Set `VOSK_MODEL` to change the default language, or
`VOSK_MODEL_PATH` to use your own model directory.

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

  1. **A link map** goes to the model as context: what else the site covers, and
     where. It's marked as not-read, so the model can say "the hinge page covers
     that" rather than inventing what's on it.
  2. **A couple are actually opened.** A small model picks which links look like
     they answer the question, and those pages are fetched too.

  One hop, same site only, and every URL still goes through the address guard.
  A link is chosen by a model out of content written by a stranger, so it gets
  no more trust than a pasted URL does — following an arbitrary outbound link
  would be a much larger surface for very little gain. `WEB_FOLLOW_LINKS=0`
  turns the following off; the link map stays.
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

Both the planner and the answering model are told today's date. A model's sense
of "now" is its training cutoff, which is how *"the latest release"* gets
answered with a version from two years ago and how the planner writes queries
anchored to the wrong year.

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

### Search backend

With no configuration, search uses DuckDuckGo's HTML endpoint — no key, but
best-effort and rate-limited. For something sturdier, self-host
[SearXNG](https://docs.searxng.org/) and set `SEARXNG_URL=http://127.0.0.1:8888`.
That endpoint is allowed to be on localhost precisely because *you* configured
it; see below.

### What it will and won't reach

Every URL is resolved and checked before it is fetched, and **only public
addresses are allowed**. Loopback, LAN, link-local, cloud metadata and the
Tailscale `100.64/10` range are all refused, so a link in a message — or a
redirect chosen by a remote server, which is re-checked at every hop — can't
turn this into a probe for your own network. The single exception is
`SEARXNG_URL`, which is operator configuration rather than something a model or
a page chose.

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
doesn't litter the list.

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

Only the most recent image-bearing turn re-sends its attachments to the model.
Re-uploading every screenshot in a thread on every turn was slow over a phone
connection and rarely what was meant; earlier turns keep their text, so the
conversation still reads. Set `CHAT_IMAGE_TURNS` higher if you compare images
across turns.

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
| **Where** | latitude, longitude, altitude, and how far out the fix might be |
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

### Nothing came through?

`📍 Photo details` being on and a photo *having* a position are different
things. The thumbnail says which before you send: **🕘** if a date was read,
**📍** if a location was, and a faint **·** if the file had neither.

To check a file directly, point the diagnostic at it. It runs the app's own
parser, so what it prints is exactly what the app would read:

```
.venv/bin/python tools/check_photo.py ~/Downloads/odometer.jpg
```

The usual reasons a photo has no location:

- **The camera never recorded one.** On Android that's Camera → Settings →
  Location, and the camera app also needs the location permission; on iPhone,
  Settings → Privacy → Location Services → Camera. This is by far the most
  common answer, and the tool will say so — if the date and camera read fine,
  the file is intact and simply has no position in it.
- **It was stripped in transit.** Messaging apps re-encode, and Google Photos'
  *share* link is not the original file — use *download* instead.
- **It's a screenshot.** Those have no EXIF at all.
- **The position is exactly 0, 0.** That's what a camera writes for a fix it
  never got. It's treated as no position, because reporting the Gulf of Guinea
  is worse than reporting nothing.

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

## API endpoints

| Method & path        | Purpose |
| -------------------- | ------- |
| `GET /`              | Chat UI |
| `GET /healthz`       | Plain `ok` health probe (stays open even when auth is on) |
| `GET /api/health`    | JSON status (Ollama host, default model, auth + voice on/off) |
| `GET /api/models`    | Installed models (proxy to Ollama `/api/tags`) |
| `POST /api/chat`     | Chat completion. Streams NDJSON by default; pass `{"stream": false}` for a single JSON reply. Body: `{ "model"?, "messages": [...] }` or `{ "prompt": "..." }`. Messages may carry `"images": ["<base64>"]` for vision models, and `"image_meta": [{...}]` alongside it — one entry per image, `{"taken","lat","lon","altitude","camera"}`, all optional. |
| `GET /api/routines`  | Every saved routine, in strip order |
| `POST /api/routines` | Save one — body `{ "name", "body", "photos"?, "web"?, "photo_meta"? }`. `web`/`photo_meta` are `true`/`false`/`null`, where null means "leave the toggle alone" |
| `POST /api/routines/starters` | Install the shipped routines, skipping names already taken. Idempotent |
| `PATCH /api/routines/<id>` | Change any subset of those fields. An absent key leaves it alone; an explicit `null` clears a forcing |
| `DELETE /api/routines/<id>` | Remove one |
| `GET /api/voice/models` | Available + downloadable Vosk speech models |
| `POST /api/voice/download` | Download a catalog model — body `{ "id": "fr" }` |
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
| `store.py`        | SQLite conversation history |
| `tests/`          | pytest suite (no Ollama or `vosk` needed) |
| `Start.sh` / `Start.bat` | Launchers for Linux / Windows |
