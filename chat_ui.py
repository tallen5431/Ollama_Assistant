#!/usr/bin/env python3
"""Single-page chat UI for the Ollama Chat app.

Kept in its own module so ``app.py`` stays focused on routing. The page is a
self-contained HTML document (inline CSS + JS, no build step, no CDN) so it
works offline behind the server manager and on a phone. It talks to this app's
own ``/api/models``, ``/api/chat`` (streaming), ``/api/health`` and
``/api/transcribe`` endpoints.
"""

from __future__ import annotations

_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>__TITLE__</title>
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
      button.danger { background:var(--danger); border-color:var(--danger); color:#fff; font-weight:600; }
      button:disabled { opacity:0.55; cursor:default; }
      #chat {
        flex:1 1 auto; overflow-y:auto; padding:1.25rem 1rem;
        display:flex; flex-direction:column; gap:0.85rem;
      }
      .wrap { width:100%; max-width:820px; margin:0 auto; }
      .msg { display:flex; }
      .msg.user { justify-content:flex-end; }
      .col { display:flex; flex-direction:column; max-width:85%; }
      .msg.user .col { align-items:flex-end; }
      .bubble { padding:0.65rem 0.85rem; border-radius:0.9rem;
        white-space:pre-wrap; word-wrap:break-word; line-height:1.45; }
      .msg.user .bubble { background:var(--user); color:#fff; border-bottom-right-radius:0.2rem; }
      .msg.assistant .bubble { background:var(--assistant); border-bottom-left-radius:0.2rem; }
      .msg.error .bubble { background:#3f1d1d; border:1px solid var(--danger); color:#fecaca; }
      .role { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.04em;
              color:var(--muted); margin:0 0.3rem 0.2rem; }
      .meta { font-size:0.7rem; color:var(--muted); margin:0.25rem 0.3rem 0; }
      details.think {
        margin:0 0 0.4rem; background:var(--panel2); border:1px solid var(--border);
        border-radius:0.6rem; padding:0.2rem 0.55rem;
      }
      details.think summary { cursor:pointer; font-size:0.75rem; color:var(--muted); padding:0.25rem 0; }
      .think-body { white-space:pre-wrap; font-size:0.85rem; color:var(--muted);
        border-top:1px solid var(--border); padding:0.4rem 0; margin-top:0.2rem; }
      .empty { color:var(--muted); text-align:center; margin-top:15vh; font-size:0.95rem; }
      footer { border-top:1px solid var(--border); background:var(--panel); padding:0.6rem 1rem; }
      .composer { display:flex; gap:0.5rem; align-items:flex-end; }
      textarea {
        flex:1 1 auto; resize:none; font-family:inherit; font-size:0.95rem;
        background:var(--panel2); color:var(--text);
        border:1px solid var(--border); border-radius:0.7rem;
        padding:0.6rem 0.75rem; max-height:40vh; min-height:2.6rem; line-height:1.4;
      }
      #mic { font-size:1.1rem; line-height:1; padding:0.5rem 0.6rem; }
      #mic.rec { background:var(--danger); border-color:var(--danger); animation:pulse 1.2s infinite; }
      #voiceModel { max-width:11rem; font-size:0.82rem; padding:0.5rem 0.4rem; }
      @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.55;} }
      .hint { color:var(--muted); font-size:0.72rem; margin:0.35rem 0.2rem 0; min-height:1rem; }
    </style>
  </head>
  <body>
    <header>
      <h1>__TITLE__</h1>
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
          <select id="voiceModel" title="Speech recognition language" hidden></select>
          <button id="mic" title="Speak (offline transcription)" hidden>🎤</button>
          <textarea id="input" rows="1" placeholder="Type a message…  (Enter to send, Shift+Enter for a new line)"></textarea>
          <button class="primary" id="send">Send</button>
          <button class="danger" id="stop" hidden>Stop</button>
        </div>
        <p class="hint" id="hint"></p>
      </div>
    </footer>

    <script>
      const chatEl   = document.getElementById("chat");
      const emptyEl  = document.getElementById("empty");
      const inputEl  = document.getElementById("input");
      const sendBtn  = document.getElementById("send");
      const stopBtn  = document.getElementById("stop");
      const newBtn   = document.getElementById("newChat");
      const micBtn   = document.getElementById("mic");
      const voiceSel = document.getElementById("voiceModel");
      const modelEl  = document.getElementById("model");
      const dotEl    = document.getElementById("dot");
      const statusEl = document.getElementById("statusText");
      const hintEl   = document.getElementById("hint");

      let messages = [];       // conversation sent to /api/chat for context
      let busy = false;
      let controller = null;   // AbortController for the in-flight stream

      function setStatus(state, text) {
        dotEl.className = "dot" + (state ? " " + state : "");
        statusEl.textContent = text;
      }
      function autosize() {
        inputEl.style.height = "auto";
        inputEl.style.height = Math.min(inputEl.scrollHeight, window.innerHeight * 0.4) + "px";
      }
      function scrollDown() { chatEl.scrollTop = chatEl.scrollHeight; }

      // Split assistant text into visible content + inline <think> reasoning.
      function splitThink(raw) {
        let content = "", thinking = "", last = 0, m;
        const re = /<think>([\\s\\S]*?)(<\\/think>|$)/g;
        while ((m = re.exec(raw))) {
          content += raw.slice(last, m.index);
          thinking += m[1];
          last = m.index + m[0].length;
        }
        content += raw.slice(last);
        return { content, thinking };
      }

      function fmtUsage(u) {
        const parts = [];
        if (u.eval_count) parts.push(u.eval_count + " tokens");
        if (u.eval_count && u.eval_duration)
          parts.push((u.eval_count / (u.eval_duration / 1e9)).toFixed(0) + " tok/s");
        if (u.prompt_eval_count) parts.push(u.prompt_eval_count + " prompt");
        return parts.join("  ·  ");
      }

      function addUser(text) {
        if (emptyEl) emptyEl.remove();
        const wrap = document.createElement("div");
        wrap.className = "wrap";
        wrap.innerHTML = '<div class="msg user"><div class="col"><div class="role">You</div>' +
                         '<div class="bubble"></div></div></div>';
        wrap.querySelector(".bubble").textContent = text;
        chatEl.appendChild(wrap); scrollDown();
      }

      // Build an assistant message with a (hidden until used) thinking panel,
      // the bubble, and a usage meta line. Returns handles to update live.
      function addAssistant() {
        if (emptyEl) emptyEl.remove();
        const wrap = document.createElement("div");
        wrap.className = "wrap";
        const label = modelEl.value || "Assistant";
        wrap.innerHTML =
          '<div class="msg assistant"><div class="col">' +
            '<div class="role">' + label + '</div>' +
            '<details class="think" hidden><summary>Show thinking</summary>' +
              '<div class="think-body"></div></details>' +
            '<div class="bubble">…</div>' +
            '<div class="meta"></div>' +
          '</div></div>';
        chatEl.appendChild(wrap); scrollDown();
        return {
          root: wrap,
          bubble: wrap.querySelector(".bubble"),
          think: wrap.querySelector("details.think"),
          thinkBody: wrap.querySelector(".think-body"),
          meta: wrap.querySelector(".meta"),
        };
      }

      function markError(msg) {
        const wrap = document.createElement("div");
        wrap.className = "wrap";
        wrap.innerHTML = '<div class="msg error"><div class="col"><div class="role">Error</div>' +
                         '<div class="bubble"></div></div></div>';
        wrap.querySelector(".bubble").textContent = msg;
        chatEl.appendChild(wrap); scrollDown();
      }

      async function loadModels() {
        try {
          const resp = await fetch("api/models");
          const data = await resp.json();
          const list = (data.models || [])
            .map(m => (typeof m === "string" ? m : (m.name || m.model))).filter(Boolean);
          modelEl.innerHTML = "";
          if (!list.length) {
            const o = document.createElement("option");
            o.textContent = data.default || "(no models found)"; o.value = data.default || "";
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
          hintEl.textContent = "Could not reach the model server. Check Ollama is running and OLLAMA_HOST is set.";
        }
      }

      async function checkVoice() {
        try {
          const data = await (await fetch("api/health")).json();
          if (data.voice) { micBtn.hidden = false; await loadVoiceModels(); }
        } catch (e) { /* leave mic hidden */ }
      }

      // Populate the speech-model picker: downloaded languages first, then the
      // rest of the catalog (marked with size + ⬇, downloaded on first pick).
      async function loadVoiceModels() {
        try {
          const data = await (await fetch("api/voice/models")).json();
          if (data.error) return;
          voiceSel.innerHTML = "";
          const seen = new Set();
          for (const m of (data.available || [])) {
            seen.add(m.id);
            voiceSel.appendChild(new Option(m.label, m.id, false, m.id === data.default));
          }
          for (const m of (data.catalog || [])) {
            if (m.downloaded || seen.has(m.id)) continue;
            const o = new Option(m.label + " — " + m.size + " ⬇", m.id);
            o.dataset.download = "1";
            voiceSel.appendChild(o);
          }
          voiceSel.hidden = voiceSel.options.length <= 1;
        } catch (e) { /* leave picker hidden */ }
      }

      // Downloading a not-yet-present language happens on selection so the mic
      // is ready before you speak.
      voiceSel.addEventListener("change", async () => {
        const opt = voiceSel.selectedOptions[0];
        if (!opt || opt.dataset.download !== "1") return;
        const id = opt.value;
        voiceSel.disabled = true; micBtn.disabled = true;
        hintEl.textContent = "Downloading " + opt.textContent.replace(" ⬇", "") + "… (one-time)";
        try {
          const resp = await fetch("api/voice/download", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id }),
          });
          const j = await resp.json();
          if (j.error) { hintEl.textContent = j.error; }
          else { hintEl.textContent = "Ready — " + (j.label || id) + " downloaded."; await loadVoiceModels(); voiceSel.value = id; }
        } catch (e) { hintEl.textContent = "Download failed: " + e; }
        finally { voiceSel.disabled = false; micBtn.disabled = false; }
      });

      async function send() {
        const text = inputEl.value.trim();
        if (!text || busy) return;
        busy = true; sendBtn.disabled = true; stopBtn.hidden = false;
        addUser(text);
        messages.push({ role: "user", content: text });
        inputEl.value = ""; autosize();

        const view = addAssistant();
        controller = new AbortController();
        let rawContent = "", thinkingField = "", started = false, usage = null;

        try {
          const resp = await fetch("api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: modelEl.value || undefined, messages }),
            signal: controller.signal,
          });
          if (!resp.ok) {
            const j = await resp.json().catch(() => ({}));
            throw new Error(j.error || ("HTTP " + resp.status));
          }
          const reader = resp.body.getReader();
          const dec = new TextDecoder();
          let buf = "";
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let nl;
            while ((nl = buf.indexOf("\\n")) >= 0) {
              const line = buf.slice(0, nl).trim();
              buf = buf.slice(nl + 1);
              if (!line) continue;
              const obj = JSON.parse(line);
              if (obj.error) throw new Error(obj.error);
              if (obj.thinking) thinkingField += obj.thinking;
              const piece = (obj.message && obj.message.content) || obj.content || "";
              if (piece) rawContent += piece;
              if (obj.done) usage = obj;

              const { content, thinking } = splitThink(rawContent);
              const allThink = thinkingField + thinking;
              if (piece || obj.thinking) { started = true; view.bubble.textContent = content || "…"; }
              if (allThink) { view.think.hidden = false; view.thinkBody.textContent = allThink; }
              scrollDown();
            }
          }
          const finalContent = splitThink(rawContent).content;
          view.bubble.textContent = finalContent || "(empty response)";
          if (finalContent) messages.push({ role: "assistant", content: finalContent });
          if (usage) view.meta.textContent = fmtUsage(usage);
          setStatus("ok", "connected");
        } catch (err) {
          if (err.name === "AbortError") {
            const partial = splitThink(rawContent).content;
            view.bubble.textContent = (partial || "") + "  ⏹ stopped";
            if (partial) messages.push({ role: "assistant", content: partial });
          } else {
            view.root.remove();
            markError(err.message || String(err));
            setStatus("bad", "error");
          }
        } finally {
          busy = false; sendBtn.disabled = false; stopBtn.hidden = true;
          controller = null; inputEl.focus();
        }
      }

      function stop() { if (controller) controller.abort(); }

      function newChat() {
        if (controller) controller.abort();
        messages = [];
        chatEl.innerHTML = '<div class="wrap"><div class="empty" id="empty">' +
          'Send a message to start chatting with your local model.</div></div>';
        inputEl.focus();
      }

      // ---- Voice input (offline, via /api/transcribe) ----
      let recording = false, mediaStream = null, audioCtx = null, srcNode = null, procNode = null, buffers = [];

      async function toggleMic() {
        if (recording) { await stopMic(); return; }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          hintEl.textContent = "Microphone needs HTTPS. Serve the app over https (e.g. tailscale serve / Funnel).";
          return;
        }
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (e) {
          hintEl.textContent = "Mic blocked. Allow microphone access, and note it only works over HTTPS or localhost.";
          return;
        }
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        srcNode = audioCtx.createMediaStreamSource(mediaStream);
        procNode = audioCtx.createScriptProcessor(4096, 1, 1);
        buffers = [];
        procNode.onaudioprocess = (e) => buffers.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        srcNode.connect(procNode); procNode.connect(audioCtx.destination);
        recording = true; micBtn.classList.add("rec"); micBtn.textContent = "⏹";
        hintEl.textContent = "Listening… tap the mic to stop.";
      }

      async function stopMic() {
        recording = false; micBtn.classList.remove("rec"); micBtn.textContent = "🎤";
        try { procNode.disconnect(); srcNode.disconnect(); } catch (e) {}
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        const inRate = audioCtx.sampleRate;
        try { await audioCtx.close(); } catch (e) {}
        const wav = encodeWav(mergeBuffers(buffers), inRate, 16000);
        buffers = [];
        if (!wav) { hintEl.textContent = "No audio captured."; return; }
        hintEl.textContent = "Transcribing…";
        try {
          const mq = voiceSel.value ? ("?model=" + encodeURIComponent(voiceSel.value)) : "";
          const resp = await fetch("api/transcribe" + mq, { method: "POST",
            headers: { "Content-Type": "audio/wav" }, body: wav });
          const j = await resp.json();
          if (j.error) { hintEl.textContent = j.error; return; }
          const t = (j.text || "").trim();
          if (t) { inputEl.value = (inputEl.value ? inputEl.value.trim() + " " : "") + t; autosize(); hintEl.textContent = ""; }
          else { hintEl.textContent = "No speech detected."; }
          inputEl.focus();
        } catch (e) { hintEl.textContent = "Transcription failed: " + e; }
      }

      function mergeBuffers(list) {
        let len = 0; for (const b of list) len += b.length;
        const out = new Float32Array(len); let off = 0;
        for (const b of list) { out.set(b, off); off += b.length; }
        return out;
      }

      // Downsample Float32 samples to 16 kHz 16-bit mono and wrap in a WAV blob.
      function encodeWav(samples, inRate, outRate) {
        if (!samples.length) return null;
        const ratio = inRate / outRate;
        const outLen = Math.floor(samples.length / ratio);
        const pcm = new Int16Array(outLen);
        for (let i = 0; i < outLen; i++) {
          const s = Math.max(-1, Math.min(1, samples[Math.floor(i * ratio)] || 0));
          pcm[i] = s * 0x7fff;
        }
        const buf = new ArrayBuffer(44 + pcm.length * 2);
        const dv = new DataView(buf);
        const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
        ws(0, "RIFF"); dv.setUint32(4, 36 + pcm.length * 2, true); ws(8, "WAVE");
        ws(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
        dv.setUint32(24, outRate, true); dv.setUint32(28, outRate * 2, true);
        dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
        ws(36, "data"); dv.setUint32(40, pcm.length * 2, true);
        for (let i = 0; i < pcm.length; i++) dv.setInt16(44 + i * 2, pcm[i], true);
        return new Blob([buf], { type: "audio/wav" });
      }

      sendBtn.addEventListener("click", send);
      stopBtn.addEventListener("click", stop);
      newBtn.addEventListener("click", newChat);
      micBtn.addEventListener("click", toggleMic);
      inputEl.addEventListener("input", autosize);
      inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
      });

      loadModels();
      checkVoice();
      inputEl.focus();
    </script>
  </body>
</html>
"""


def render_page(title: str) -> str:
    """Return the chat page HTML with ``title`` substituted in."""
    return _PAGE.replace("__TITLE__", title)
