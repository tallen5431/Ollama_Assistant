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
- 🖼️ **Image input** — attach files with 📎, grab a 📸 screenshot, or paste one
  in, then ask a vision model about it (`llava`, `*-vision`, `minicpm-v`,
  `qwen2.5vl`, `moondream`, …). Images are downscaled in the browser before
  upload. See "Vision models" below.
- 💾 **Conversation history** — threads are stored server-side, so one started
  on your desktop continues on your phone. ☰ opens the list; rename, delete,
  reopen. **Turn Basic Auth on if you enable this** — see "Conversation history".
- ✍️ **Formatted replies** — markdown is rendered, with labelled code blocks and
  a copy button. Matters most with the coder models.
- 👁️ **Vision routing** — attach an image while a text-only model is selected
  and the app switches to one that can actually see it.
- 🌐 **Web access (optional)** — reads a link you paste, or has a small model
  turn your message into search queries, then grounds the answer in the pages it
  finds and cites them. Only public addresses are ever fetched. Off by default.
  See "Web access" below.
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
| `OLLAMA_TIMEOUT`    | `300`                       | Per-request timeout, in seconds |
| `CHAT_TITLE`        | `Ollama Chat`               | Title in the tab/header |
| `CHAT_DB`           | `./chat.db`                 | SQLite file holding conversation history |
| `WEB_VISION_MODEL`  | *(unset)*                   | Model used to read an attached image when planning a search. Unset picks the smallest installed vision model |
| `CHAT_MAX_BODY_MB`  | `25`                        | Maximum accepted request body size (guards the audio upload) |
| `WEB_ENABLED`       | `1`                         | Server-side switch for web access; `0` disables it entirely |
| `SEARXNG_URL`       | *(unset)*                   | Self-hosted SearXNG base URL. Unset falls back to DuckDuckGo HTML |
| `WEB_PLANNER_MODEL` | *(unset)*                   | Small model used to generate search queries. Unset reuses the answering model |
| `WEB_MAX_DOCS`      | `3`                         | Pages put in front of the model per turn |
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

- **A link in your message is read.** Paste a URL and ask about it. This is the
  best case for a local model: one clean document beats a pile of search
  snippets, and it's far better than screenshotting a page for a vision model,
  since the model gets real text instead of OCR'd pixels.
- **Otherwise a planner turns your message into search queries.** A short,
  cheap, deterministic call replies with either `NONE` or up to three `Q: `
  lines — search-engine keywords, each attacking the topic from a different
  angle. The app runs all of them, interleaves the results so every query
  contributes, and reads the top few pages. Using a plain call rather than tool
  calling means this works with *every* model, including vision and reasoning
  ones that expose no tool support.

  The planner sees the last few turns, not just your latest message, so a
  follow-up like *"what about the 14b one?"* still produces a usable query.

### Planner model

By default the answering model does the planning. Set `WEB_PLANNER_MODEL` to a
small model to keep a large one from being invoked twice a turn:

```
WEB_PLANNER_MODEL=qwen2.5-coder:0.5b
```

Worth measuring rather than assuming: this is only faster if both models fit in
VRAM at once. If they don't, Ollama swaps between them every turn and a small
planner ends up *slower* than just reusing the model already loaded. Leave it
unset if VRAM is tight.

Progress appears live above the reply ("Searching for…", "Reading example.com…")
and the pages used are listed underneath, numbered to match the `[n]` citations
the model is asked to use.

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
sense, which means an image-heavy history grows the database. Delete old
conversations to reclaim it; the file is plain SQLite if you'd rather prune it
yourself.

## Vision models

There are three ways to attach an image, up to four per message:

- **📎 Attach** — pick image files.
- **📸 Screenshot** — capture a window, tab, or whole screen. The browser's own
  picker decides what is shared; a single frame is taken and the capture stream
  is dropped immediately, so nothing is recorded. Desktop only — the button is
  hidden where `getDisplayMedia` isn't supported, which includes mobile browsers.
- **Paste** — paste an image anywhere on the page, usually the quickest route
  for a screenshot taken with the OS shortcut.

The app picks the model for you: attach an image while a text-only model is
selected and it switches to an installed one that can see, saying so in the hint
line. Capability comes from what Ollama reports in `/api/tags` (the `clip` and
`mllama` families, an explicit `vision` capability) with a name-based fallback
for builds whose details block is sparse. Change the dropdown back if you'd
rather it didn't.

With **Web access** on, an attached image is also read *before* the search is
planned — the exact error text, product names and versions are pulled out of it
and folded into the queries. Without that, a screenshot with "what's this?"
plans nothing worth running. `WEB_VISION_MODEL` pins which model does that pass;
unset picks the smallest installed one, since it only needs to describe.

All three paths handle the image identically. Anything up to 1920 px on the long
edge is kept at its original size and encoded as **PNG** — losslessly, so the
small text in a screenshot stays legible, which is usually the entire reason for
sending one. A 1920×1080 screenshot costs about 60 KB that way.

Only when PNG exceeds ~1.5 MB is the image treated as a photograph: downscaled to
1920 px and encoded as JPEG (dropping a quality step if it is still very large),
with transparency flattened onto white since JPEG has no alpha. EXIF orientation
is honoured throughout, so photos aren't sent sideways.

They travel as base64 in the message, exactly as Ollama's native API expects:

```json
{ "role": "user", "content": "what is this?", "images": ["<base64>"] }
```

Note the whole conversation is re-sent on every turn, so an image stays in the
context for the rest of the chat — **New chat** clears it.

## API endpoints

| Method & path        | Purpose |
| -------------------- | ------- |
| `GET /`              | Chat UI |
| `GET /healthz`       | Plain `ok` health probe (stays open even when auth is on) |
| `GET /api/health`    | JSON status (Ollama host, default model, auth + voice on/off) |
| `GET /api/models`    | Installed models (proxy to Ollama `/api/tags`) |
| `POST /api/chat`     | Chat completion. Streams NDJSON by default; pass `{"stream": false}` for a single JSON reply. Body: `{ "model"?, "messages": [...] }` or `{ "prompt": "..." }`. Messages may carry `"images": ["<base64>"]` for vision models. |
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
