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

## API endpoints

| Method & path      | Purpose |
| ------------------ | ------- |
| `GET /`            | Chat UI |
| `GET /healthz`     | Plain `ok` health probe (stays open even when auth is on) |
| `GET /api/health`  | JSON status (Ollama host, default model, auth on/off) |
| `GET /api/models`  | Installed models (proxy to Ollama `/api/tags`) |
| `POST /api/chat`   | Chat completion — body: `{ "model"?, "messages": [...] }` or `{ "prompt": "..." }` |

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
| `app.py`          | Flask routes + waitress entrypoint |
| `config.py`       | Environment + logging helpers |
| `ollama_client.py`| Thin HTTP client for the Ollama API |
| `chat_ui.py`      | Single-page chat UI (inline HTML/CSS/JS) |
| `authz.py`        | Optional HTTP Basic Auth |
| `Start.sh` / `Start.bat` | Launchers for Linux / Windows |
