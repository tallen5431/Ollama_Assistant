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
OLLAMA_HOST=http://<desktop-ip>:11434 .venv/bin/python app.py
```

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
| `CHAT_AUTH_USER` / `CHAT_AUTH_PASSWORD` | *(unset)* | Enable HTTP Basic Auth (or use `CHAT_AUTH=user:password`) |
| `VOSK_MODEL_PATH`   | *(unset)*                   | Path to an unpacked Vosk model. If unset, the small English model is downloaded once into `models/` on first use. |
| `VOSK_MODELS_DIR`   | `./models`                  | Where downloaded Vosk models are stored |

> `Start.sh` defaults `OLLAMA_HOST` to `http://100.98.112.1:11434` (the Tailscale
> address the other cards use) — change it to wherever your desktop's Ollama is
> reachable, or set it in the program's **env** in the server manager.

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

## API endpoints

| Method & path        | Purpose |
| -------------------- | ------- |
| `GET /`              | Chat UI |
| `GET /healthz`       | Plain `ok` health probe (stays open even when auth is on) |
| `GET /api/health`    | JSON status (Ollama host, default model, auth + voice on/off) |
| `GET /api/models`    | Installed models (proxy to Ollama `/api/tags`) |
| `POST /api/chat`     | Chat completion. Streams NDJSON by default; pass `{"stream": false}` for a single JSON reply. Body: `{ "model"?, "messages": [...] }` or `{ "prompt": "..." }` |
| `POST /api/transcribe` | Speech-to-text: POST WAV audio, returns `{ "text": ... }` |

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
| `Start.sh` / `Start.bat` | Launchers for Linux / Windows |
