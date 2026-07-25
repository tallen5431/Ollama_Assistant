#!/usr/bin/env python3
"""Single-page chat UI for the Ollama Chat app.

Kept in its own module so ``app.py`` stays focused on routing. The page is a
self-contained HTML document (inline CSS + JS, no build step, no CDN) so it
works offline behind the server manager and on a phone. It talks to this app's
own ``/api/models`` and ``/api/chat`` endpoints.
"""

from __future__ import annotations

from string import Template

_PAGE = Template(
    """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>$title</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <style>
      :root {
        --bg:#0b1120; --panel:#0f172a; --panel2:#111c33; --border:#1e293b;
        --text:#e5e7eb; --muted:#94a3b8; --accent:#2563eb; --accent2:#1d4ed8;
        --user:#2563eb; --assistant:#1e293b; --danger:#ef4444; --ok:#22c55e;
      }
      * { box-sizing:border-box; }
      html, body { height:100%; margin:0; }
      body {
        background:var(--bg); color:var(--text);
        font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        display:flex; flex-direction:column; height:100dvh;
      }
      header {
        display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap;
        padding:0.6rem 1rem; background:var(--panel);
        border-bottom:1px solid var(--border);
      }
      header h1 { font-size:1.05rem; margin:0; font-weight:600; white-space:nowrap; }
      .status { display:flex; align-items:center; gap:0.35rem; font-size:0.8rem; color:var(--muted); }
      .dot { width:0.6rem; height:0.6rem; border-radius:50%; background:var(--muted); }
      .dot.ok { background:var(--ok); } .dot.bad { background:var(--danger); }
      .spacer { flex:1 1 auto; }
      select, button {
        font-family:inherit; font-size:0.9rem;
        background:var(--panel2); color:var(--text);
        border:1px solid var(--border); border-radius:0.5rem;
        padding:0.4rem 0.6rem;
      }
      select { max-width:16rem; }
      button { cursor:pointer; }
      button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
      button.primary:hover { background:var(--accent2); }
      button:disabled { opacity:0.55; cursor:default; }
      #chat {
        flex:1 1 auto; overflow-y:auto; padding:1.25rem 1rem;
        display:flex; flex-direction:column; gap:0.85rem;
      }
      .wrap { width:100%; max-width:820px; margin:0 auto; }
      .msg { display:flex; }
      .msg.user { justify-content:flex-end; }
      .bubble {
        max-width:85%; padding:0.65rem 0.85rem; border-radius:0.9rem;
        white-space:pre-wrap; word-wrap:break-word; line-height:1.45;
      }
      .msg.user .bubble { background:var(--user); color:#fff; border-bottom-right-radius:0.2rem; }
      .msg.assistant .bubble { background:var(--assistant); border-bottom-left-radius:0.2rem; }
      .msg.error .bubble { background:#3f1d1d; border:1px solid var(--danger); color:#fecaca; }
      .role { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.04em;
              color:var(--muted); margin:0 0.3rem 0.2rem; }
      .empty { color:var(--muted); text-align:center; margin-top:15vh; font-size:0.95rem; }
      .typing span {
        display:inline-block; width:0.4rem; height:0.4rem; margin:0 1px;
        background:var(--muted); border-radius:50%; animation:blink 1.2s infinite both;
      }
      .typing span:nth-child(2) { animation-delay:0.2s; }
      .typing span:nth-child(3) { animation-delay:0.4s; }
      @keyframes blink { 0%,80%,100%{opacity:0.2;} 40%{opacity:1;} }
      footer { border-top:1px solid var(--border); background:var(--panel); padding:0.6rem 1rem; }
      .composer { display:flex; gap:0.6rem; align-items:flex-end; }
      textarea {
        flex:1 1 auto; resize:none; font-family:inherit; font-size:0.95rem;
        background:var(--panel2); color:var(--text);
        border:1px solid var(--border); border-radius:0.7rem;
        padding:0.6rem 0.75rem; max-height:40vh; min-height:2.6rem; line-height:1.4;
      }
      .hint { color:var(--muted); font-size:0.72rem; margin:0.35rem 0.2rem 0; }
    </style>
  </head>
  <body>
    <header class="wrap-full">
      <h1>$title</h1>
      <div class="status"><span class="dot" id="dot"></span><span id="statusText">connecting…</span></div>
      <div class="spacer"></div>
      <label class="status" for="model">Model</label>
      <select id="model" title="Choose which local model answers"></select>
      <button id="newChat" title="Clear the conversation">New chat</button>
    </header>

    <main id="chat">
      <div class="wrap"><div class="empty" id="empty">
        Send a message to start chatting with your local model.
      </div></div>
    </main>

    <footer>
      <div class="wrap">
        <div class="composer">
          <textarea id="input" rows="1" placeholder="Type a message…  (Enter to send, Shift+Enter for a new line)"></textarea>
          <button class="primary" id="send">Send</button>
        </div>
        <p class="hint" id="hint"></p>
      </div>
    </footer>

    <script>
      const chatEl   = document.getElementById("chat");
      const emptyEl  = document.getElementById("empty");
      const inputEl  = document.getElementById("input");
      const sendBtn  = document.getElementById("send");
      const newBtn   = document.getElementById("newChat");
      const modelEl  = document.getElementById("model");
      const dotEl    = document.getElementById("dot");
      const statusEl = document.getElementById("statusText");
      const hintEl   = document.getElementById("hint");

      // Conversation history sent to /api/chat so the model keeps context.
      let messages = [];
      let busy = false;

      function setStatus(state, text) {
        dotEl.className = "dot" + (state ? " " + state : "");
        statusEl.textContent = text;
      }

      function autosize() {
        inputEl.style.height = "auto";
        inputEl.style.height = Math.min(inputEl.scrollHeight, window.innerHeight * 0.4) + "px";
      }

      function addBubble(role, text) {
        if (emptyEl) emptyEl.remove();
        const wrap = document.createElement("div");
        wrap.className = "wrap";
        const msg = document.createElement("div");
        msg.className = "msg " + role;
        const inner = document.createElement("div");
        const label = document.createElement("div");
        label.className = "role";
        label.textContent = role === "user" ? "You" : (role === "error" ? "Error" : (modelEl.value || "Assistant"));
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        inner.appendChild(label);
        inner.appendChild(bubble);
        msg.appendChild(inner);
        wrap.appendChild(msg);
        chatEl.appendChild(wrap);
        bubble.textContent = text;
        chatEl.scrollTop = chatEl.scrollHeight;
        return bubble;
      }

      function typingBubble() {
        const wrap = document.createElement("div");
        wrap.className = "wrap";
        wrap.innerHTML = '<div class="msg assistant"><div><div class="role">' +
          (modelEl.value || "Assistant") +
          '</div><div class="bubble typing"><span></span><span></span><span></span></div></div></div>';
        chatEl.appendChild(wrap);
        chatEl.scrollTop = chatEl.scrollHeight;
        return wrap;
      }

      async function loadModels() {
        try {
          const resp = await fetch("api/models");
          const data = await resp.json();
          const list = (data.models || []).map(m => (typeof m === "string" ? m : (m.name || m.model))).filter(Boolean);
          modelEl.innerHTML = "";
          if (!list.length) {
            const o = document.createElement("option");
            o.textContent = data.default || "(no models found)";
            o.value = data.default || "";
            modelEl.appendChild(o);
          } else {
            for (const name of list) {
              const o = document.createElement("option");
              o.value = name; o.textContent = name;
              if (name === data.default) o.selected = true;
              modelEl.appendChild(o);
            }
          }
          setStatus("ok", "connected");
          hintEl.textContent = list.length ? (list.length + " model(s) available") : "";
        } catch (err) {
          setStatus("bad", "no connection");
          hintEl.textContent = "Could not reach the model server. Check that Ollama is running and OLLAMA_HOST is set.";
        }
      }

      async function send() {
        const text = inputEl.value.trim();
        if (!text || busy) return;
        busy = true; sendBtn.disabled = true;
        addBubble("user", text);
        messages.push({ role: "user", content: text });
        inputEl.value = ""; autosize();
        const typing = typingBubble();
        try {
          const resp = await fetch("api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: modelEl.value || undefined, messages })
          });
          const data = await resp.json();
          typing.remove();
          if (!resp.ok || data.error) {
            addBubble("error", data.error || ("Request failed (HTTP " + resp.status + ")"));
          } else {
            const reply = data.reply || "(empty response)";
            addBubble("assistant", reply);
            messages.push({ role: "assistant", content: reply });
            setStatus("ok", "connected");
          }
        } catch (err) {
          typing.remove();
          addBubble("error", "Request failed: " + err);
          setStatus("bad", "no connection");
        } finally {
          busy = false; sendBtn.disabled = false; inputEl.focus();
        }
      }

      function newChat() {
        messages = [];
        chatEl.innerHTML = '<div class="wrap"><div class="empty" id="empty">Send a message to start chatting with your local model.</div></div>';
        inputEl.focus();
      }

      sendBtn.addEventListener("click", send);
      newBtn.addEventListener("click", newChat);
      inputEl.addEventListener("input", autosize);
      inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
      });

      loadModels();
      inputEl.focus();
    </script>
  </body>
</html>
"""
)


def render_page(title: str) -> str:
    """Return the chat page HTML with ``title`` substituted in."""
    return _PAGE.substitute(title=title)
