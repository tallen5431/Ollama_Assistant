#!/usr/bin/env python3
"""Single-page chat UI for the Ollama Chat app.

Kept in its own module so ``app.py`` stays focused on routing. The page is a
self-contained HTML document (inline CSS + JS, no build step, no CDN) so it
works offline behind the server manager and on a phone. It talks to this app's
own ``/api/models``, ``/api/chat`` (streaming), ``/api/health`` and
``/api/transcribe`` endpoints.
"""

from __future__ import annotations

import html

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
        flex:1 1 auto; min-width:0; resize:none; font-family:inherit; font-size:0.95rem;
        background:var(--panel2); color:var(--text);
        border:1px solid var(--border); border-radius:0.7rem;
        padding:0.6rem 0.75rem; max-height:40vh; min-height:2.6rem; line-height:1.4;
      }
      .composer button { flex:0 0 auto; white-space:nowrap; }
      #mic { font-size:1.1rem; line-height:1; padding:0.5rem 0.6rem; }
      #mic.rec { background:var(--danger); border-color:var(--danger); animation:pulse 1.2s infinite; }
      .voicebar { display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem; flex-wrap:wrap; }
      .voicebar-label { font-size:0.75rem; color:var(--muted); white-space:nowrap; }
      .voicebar-check {
        display:flex; align-items:center; gap:0.3rem; cursor:pointer;
        font-size:0.75rem; color:var(--muted); white-space:nowrap;
      }
      .voicebar-check input { accent-color:var(--accent); margin:0; }
      #attach, #shot { font-size:1.05rem; line-height:1; padding:0.5rem 0.6rem; }
      .thumbs { display:flex; gap:0.4rem; flex-wrap:wrap; margin-bottom:0.5rem; }
      .thumb { position:relative; width:3.5rem; height:3.5rem; border-radius:0.5rem;
        overflow:hidden; border:1px solid var(--border); }
      .thumb img { width:100%; height:100%; object-fit:cover; display:block; }
      .thumb button {
        position:absolute; top:0; right:0; padding:0 0.28rem; font-size:0.7rem;
        line-height:1.3; border:0; border-radius:0 0 0 0.4rem;
        background:rgba(0,0,0,0.65); color:#fff;
      }
      .bubble img { max-width:min(320px,100%); border-radius:0.5rem;
        margin-bottom:0.35rem; display:block; }
      #voiceModel { flex:0 1 auto; min-width:0; max-width:16rem; font-size:0.82rem; padding:0.45rem 0.4rem; }
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
        <div class="voicebar" id="voicebar" hidden>
          <span class="voicebar-label">🎙 Voice</span>
          <select id="voiceModel" title="Speech recognition language"></select>
          <label class="voicebar-check" title="On by default. Turns off echo cancellation, which has nothing to cancel on headphones and otherwise mutes your voice while audio is playing. Untick it on laptop speakers.">
            <input type="checkbox" id="headset" checked> 🎧 Headphones
          </label>
          <label class="voicebar-check" title="Send as soon as speech is transcribed, instead of waiting for you to press Send.">
            <input type="checkbox" id="autosend"> ⚡ Auto-send
          </label>
          <label class="voicebar-check" title="Keep the mic open and treat a pause in speech as the end of a message. With Auto-send on, this runs a whole conversation from one tap.">
            <input type="checkbox" id="continuous"> 🔁 Continuous
          </label>
        </div>
        <div class="thumbs" id="thumbs" hidden></div>
        <div class="composer">
          <textarea id="input" rows="1" placeholder="Type a message…  (Enter to send, Shift+Enter for a new line)"></textarea>
          <button id="attach" title="Attach an image (needs a vision model)">📎</button>
          <button id="shot" title="Capture a screenshot to analyse" hidden>📸</button>
          <button id="mic" title="Speak (offline transcription)" hidden>🎤</button>
          <button class="primary" id="send">Send</button>
          <button class="danger" id="stop" hidden>Stop</button>
        </div>
        <input type="file" id="file" accept="image/*" multiple hidden>
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
      const voiceBar = document.getElementById("voicebar");
      const voiceSel = document.getElementById("voiceModel");
      const headsetEl = document.getElementById("headset");
      const autoSendEl = document.getElementById("autosend");
      const continuousEl = document.getElementById("continuous");
      const attachBtn = document.getElementById("attach");
      const shotBtn  = document.getElementById("shot");
      const fileEl   = document.getElementById("file");
      const thumbsEl = document.getElementById("thumbs");
      const modelEl  = document.getElementById("model");
      const dotEl    = document.getElementById("dot");
      const statusEl = document.getElementById("statusText");
      const hintEl   = document.getElementById("hint");

      let messages = [];       // conversation sent to /api/chat for context
      let busy = false;
      let controller = null;   // AbortController for the in-flight stream
      let pendingImages = [];  // [{ b64, url }] attached but not yet sent

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

      function addUser(text, images) {
        if (emptyEl) emptyEl.remove();
        const wrap = document.createElement("div");
        wrap.className = "wrap";
        wrap.innerHTML = '<div class="msg user"><div class="col"><div class="role">You</div>' +
                         '<div class="bubble"></div></div></div>';
        const bubble = wrap.querySelector(".bubble");
        for (const img of images || []) {
          const el = document.createElement("img");
          el.src = img.url; el.alt = "attached image";
          bubble.appendChild(el);
        }
        if (text) bubble.appendChild(document.createTextNode(text));
        chatEl.appendChild(wrap); scrollDown();
      }

      // ---- Image attachments (for vision models: llava, *-vision, moondream…) ----

      function renderThumbs() {
        thumbsEl.innerHTML = "";
        thumbsEl.hidden = pendingImages.length === 0;
        pendingImages.forEach((img, i) => {
          const cell = document.createElement("div");
          cell.className = "thumb";
          const el = document.createElement("img");
          el.src = img.url; el.alt = "attachment";
          const rm = document.createElement("button");
          rm.textContent = "✕"; rm.title = "Remove";
          rm.addEventListener("click", () => { pendingImages.splice(i, 1); renderThumbs(); });
          cell.appendChild(el); cell.appendChild(rm);
          thumbsEl.appendChild(cell);
        });
      }

      function loadBitmap(file) {
        // imageOrientation honours EXIF so phone photos aren't sent sideways.
        if (window.createImageBitmap) {
          return createImageBitmap(file, { imageOrientation: "from-image" })
            .catch(() => createImageBitmap(file));
        }
        return new Promise((resolve, reject) => {
          const el = new Image(), url = URL.createObjectURL(file);
          el.onload = () => { URL.revokeObjectURL(url); resolve(el); };
          el.onerror = () => { URL.revokeObjectURL(url); reject(new Error("unreadable image")); };
          el.src = url;
        });
      }

      // Downscale before sending: vision models work from a few hundred pixels,
      // and a full-size photo or screen grab would blow past the body limit.
      function drawScaled(source, w, h, maxDim) {
        const scale = Math.min(1, maxDim / Math.max(w, h));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(w * scale));
        canvas.height = Math.max(1, Math.round(h * scale));
        canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
        const url = canvas.toDataURL("image/jpeg", 0.9);
        return { url, b64: url.slice(url.indexOf(",") + 1) };
      }

      async function toAttachment(file, maxDim = 1024) {
        const bmp = await loadBitmap(file);
        const out = drawScaled(bmp, bmp.width || bmp.naturalWidth,
                                    bmp.height || bmp.naturalHeight, maxDim);
        if (bmp.close) bmp.close();
        return out;
      }

      function addAttachment(att) {
        if (pendingImages.length >= 4) { hintEl.textContent = "Up to 4 images per message."; return false; }
        pendingImages.push(att); renderThumbs();
        return true;
      }

      attachBtn.addEventListener("click", () => fileEl.click());

      fileEl.addEventListener("change", async () => {
        for (const file of Array.from(fileEl.files)) {
          try { if (!addAttachment(await toAttachment(file))) break; }
          catch (e) { hintEl.textContent = "Could not read " + file.name; }
        }
        fileEl.value = "";   // so re-picking the same file fires change again
      });

      // Screenshots: grab a single frame from a display-capture stream, then
      // drop the stream immediately — nothing is recorded, and the browser's own
      // picker decides what is shared. Kept at a higher resolution than photos
      // because screenshots are mostly text, which does not survive downscaling.
      async function grabScreenshot() {
        let stream;
        try {
          stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
        } catch (e) {
          hintEl.textContent = "Screen capture cancelled.";
          return;
        }
        try {
          const video = document.createElement("video");
          video.srcObject = stream; video.muted = true; video.playsInline = true;
          await video.play();
          if (video.readyState < 2) {
            await new Promise((res) => { video.onloadeddata = res; });
          }
          // One extra frame: the first can still be blank on some compositors.
          await new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(res)));
          if (!video.videoWidth) { hintEl.textContent = "Screen capture produced no frame."; return; }
          addAttachment(drawScaled(video, video.videoWidth, video.videoHeight, 1600));
          hintEl.textContent = "Screenshot attached — describe what you want to know about it.";
          inputEl.focus();
        } catch (e) {
          hintEl.textContent = "Screen capture failed: " + e;
        } finally {
          stream.getTracks().forEach((t) => t.stop());
        }
      }

      // getDisplayMedia is desktop-only; hide the button rather than offer a
      // control that always fails (mobile browsers do not implement it).
      if (navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia) {
        shotBtn.hidden = false;
        shotBtn.addEventListener("click", grabScreenshot);
      }

      // Paste an image straight in — usually the quickest route for a screenshot
      // taken with the OS shortcut.
      inputEl.addEventListener("paste", async (e) => {
        const files = Array.from((e.clipboardData && e.clipboardData.items) || [])
          .filter((it) => it.type && it.type.startsWith("image/"))
          .map((it) => it.getAsFile())
          .filter(Boolean);
        if (!files.length) return;
        e.preventDefault();
        for (const file of files) {
          try { if (!addAttachment(await toAttachment(file))) break; }
          catch (err) { hintEl.textContent = "Could not read the pasted image."; }
        }
      });

      // Build an assistant message with a (hidden until used) thinking panel,
      // the bubble, and a usage meta line. Returns handles to update live.
      function addAssistant() {
        if (emptyEl) emptyEl.remove();
        const wrap = document.createElement("div");
        wrap.className = "wrap";
        wrap.innerHTML =
          '<div class="msg assistant"><div class="col">' +
            '<div class="role"></div>' +
            '<details class="think" hidden><summary>Show thinking</summary>' +
              '<div class="think-body"></div></details>' +
            '<div class="bubble">…</div>' +
            '<div class="meta"></div>' +
          '</div></div>';
        // Model names come from the Ollama server — set as text, never markup.
        wrap.querySelector(".role").textContent = modelEl.value || "Assistant";
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
          voiceBar.hidden = voiceSel.options.length === 0;
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
        const images = pendingImages.slice();
        if ((!text && !images.length) || busy) return;
        busy = true; sendBtn.disabled = true; stopBtn.hidden = false;
        pendingImages = []; renderThumbs();
        addUser(text, images);
        // Ollama takes images as bare base64 alongside the text, not as a data URL.
        const userMsg = { role: "user", content: text };
        if (images.length) userMsg.images = images.map(img => img.b64);
        messages.push(userMsg);
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
        pendingImages = []; renderThumbs();
        chatEl.innerHTML = '<div class="wrap"><div class="empty" id="empty">' +
          'Send a message to start chatting with your local model.</div></div>';
        inputEl.focus();
      }

      // ---- Voice input (offline, via /api/transcribe) ----
      let recording = false, mediaStream = null, audioCtx = null, srcNode = null, procNode = null, buffers = [];
      let micRate = 16000;

      // Silence detection, so a pause ends an utterance and the mic can stay on
      // for a whole conversation instead of being tapped once per sentence.
      const VAD_SILENCE_MS = 900;      // pause length that closes an utterance
      const VAD_MIN_SPEECH_MS = 300;   // shorter than this is a noise blip
      const VAD_IDLE_STOP_MS = 60000;  // close a mic that was left on by accident
      let vadFloor = 0, vadSpeechMs = 0, vadSilenceMs = 0, vadIdleMs = 0, vadHasSpeech = false;

      function vadReset() {
        vadFloor = 0; vadSpeechMs = 0; vadSilenceMs = 0; vadIdleMs = 0; vadHasSpeech = false;
      }

      // Decide speech vs silence for one buffer, and close the utterance on a
      // long enough pause. The threshold rides on a noise floor that falls fast
      // and rises slowly, so it adapts to the room rather than to a constant.
      function vadStep(chunk) {
        let sum = 0;
        for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
        const rms = Math.sqrt(sum / chunk.length);
        const ms = (chunk.length / micRate) * 1000;

        if (vadFloor === 0) vadFloor = rms;
        else vadFloor = rms < vadFloor ? vadFloor * 0.9 + rms * 0.1
                                       : vadFloor * 0.995 + rms * 0.005;

        if (rms > Math.max(vadFloor * 3, 0.006)) {
          vadSpeechMs += ms; vadSilenceMs = 0; vadIdleMs = 0;
          if (vadSpeechMs >= VAD_MIN_SPEECH_MS) vadHasSpeech = true;
        } else {
          vadSilenceMs += ms; vadIdleMs += ms;
          if (vadHasSpeech && vadSilenceMs >= VAD_SILENCE_MS) flushUtterance();
          else if (!vadHasSpeech && vadIdleMs >= VAD_IDLE_STOP_MS) setTimeout(stopMic, 0);
        }
      }

      // Cut what has been captured so far into its own utterance and send it off
      // to be transcribed, without tearing down the mic.
      function flushUtterance() {
        const captured = buffers;
        buffers = [];
        vadSpeechMs = 0; vadSilenceMs = 0; vadHasSpeech = false;
        const wav = encodeWav(mergeBuffers(captured), micRate, 16000);
        if (wav) transcribeBlob(wav);
      }

      async function toggleMic() {
        if (recording) { await stopMic(); return; }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          hintEl.textContent = "Microphone needs HTTPS. Serve the app over https (e.g. tailscale serve / Funnel).";
          return;
        }
        try {
          // Ask for the cleanup filters explicitly: they are on by default in
          // some browsers only, and they are what keeps speaker audio (music,
          // a video call) from bleeding into the recording.
          // Echo cancellation exists to stop speaker output leaking back into
          // the mic. On headphones there is no such leak, and its residual
          // suppressor ducks the mic whenever playback is loud — so it silences
          // you over music. Noise suppression and AGC act on the mic only, so
          // they stay on either way.
          mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              echoCancellation: !headsetEl.checked,
              noiseSuppression: true,
              autoGainControl: true,
            },
          });
        } catch (e) {
          hintEl.textContent = "Mic blocked. Allow microphone access, and note it only works over HTTPS or localhost.";
          return;
        }
        // Ask for a 16 kHz context so the browser resamples the mic itself,
        // with a real anti-alias filter. Falling back to the device rate means
        // resampling by hand below, which is cruder.
        const Ctx = window.AudioContext || window.webkitAudioContext;
        try {
          audioCtx = new Ctx({ sampleRate: 16000 });
        } catch (e) {
          audioCtx = new Ctx();
        }
        micRate = audioCtx.sampleRate;
        srcNode = audioCtx.createMediaStreamSource(mediaStream);
        procNode = audioCtx.createScriptProcessor(4096, 1, 1);
        buffers = []; vadReset();
        procNode.onaudioprocess = (e) => {
          const chunk = new Float32Array(e.inputBuffer.getChannelData(0));
          buffers.push(chunk);
          if (continuousEl.checked) vadStep(chunk);
        };
        srcNode.connect(procNode); procNode.connect(audioCtx.destination);
        recording = true; micBtn.classList.add("rec"); micBtn.textContent = "⏹";
        hintEl.textContent = continuousEl.checked
          ? "Listening… pause to send an utterance. Tap the mic to stop."
          : "Listening… tap the mic to stop.";
      }

      async function stopMic() {
        if (!recording) return;
        recording = false; micBtn.classList.remove("rec"); micBtn.textContent = "🎤";
        try { procNode.disconnect(); srcNode.disconnect(); } catch (e) {}
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        const captured = buffers;
        buffers = []; vadReset();
        try { await audioCtx.close(); } catch (e) {}
        const wav = encodeWav(mergeBuffers(captured), micRate, 16000);
        if (!wav) { hintEl.textContent = ""; return; }
        await transcribeBlob(wav);
      }

      // Post one utterance for transcription. Runs both for a whole push-to-talk
      // recording and for each pause-delimited chunk while continuous is on, so
      // it must never assume the mic has stopped.
      async function transcribeBlob(wav) {
        hintEl.textContent = "Transcribing…";
        try {
          const mq = voiceSel.value ? ("?model=" + encodeURIComponent(voiceSel.value)) : "";
          const resp = await fetch("api/transcribe" + mq, { method: "POST",
            headers: { "Content-Type": "audio/wav" }, body: wav });
          const j = await resp.json();
          if (j.error) { hintEl.textContent = j.error; return; }
          const t = (j.text || "").trim();
          if (!t) { hintEl.textContent = recording ? "Listening…" : "No speech detected."; return; }
          inputEl.value = (inputEl.value ? inputEl.value.trim() + " " : "") + t;
          autosize();
          hintEl.textContent = recording ? "Listening…" : "";
          // Auto-send skips the Send button so speaking alone drives the chat.
          // While a reply is still streaming, hold the text in the box instead of
          // dropping it — send() would refuse it and the words would be lost.
          if (autoSendEl.checked && !busy) { send(); return; }
          if (autoSendEl.checked && busy) hintEl.textContent = "Waiting for the reply to finish…";
          if (!recording) inputEl.focus();
        } catch (e) { hintEl.textContent = "Transcription failed: " + e; }
      }

      function mergeBuffers(list) {
        let len = 0; for (const b of list) len += b.length;
        const out = new Float32Array(len); let off = 0;
        for (const b of list) { out.set(b, off); off += b.length; }
        return out;
      }

      // Reduce the sample rate, averaging each source window rather than
      // picking one sample from it. Plain decimation folds everything above the
      // new Nyquist (8 kHz) back down into the speech band — a 12 kHz cymbal
      // lands at 4 kHz — which is why background music wrecked recognition.
      // Averaging is a crude low-pass that takes most of that out. Usually a
      // no-op: the capture context above is already 16 kHz where supported.
      function resample(samples, inRate, outRate) {
        if (inRate === outRate) return samples;
        const ratio = inRate / outRate;
        const outLen = Math.floor(samples.length / ratio);
        const out = new Float32Array(outLen);
        for (let i = 0; i < outLen; i++) {
          const start = Math.floor(i * ratio);
          const end = Math.min(Math.floor((i + 1) * ratio), samples.length);
          let sum = 0, n = 0;
          for (let j = start; j < end; j++) { sum += samples[j]; n++; }
          out[i] = n ? sum / n : 0;
        }
        return out;
      }

      // Convert Float32 samples to 16 kHz 16-bit mono and wrap in a WAV blob.
      function encodeWav(samples, inRate, outRate) {
        if (!samples.length) return null;
        const src = resample(samples, inRate, outRate);
        const pcm = new Int16Array(src.length);
        for (let i = 0; i < src.length; i++) {
          const s = Math.max(-1, Math.min(1, src[i] || 0));
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

      // Remember the voice toggles — they describe your hardware and habits, not
      // this visit. Headphones defaults ON (see the checkbox in the markup);
      // only an explicit stored choice overrides it. Storage can throw in a
      // locked-down browser, so every access is guarded.
      function rememberToggle(el, key, dflt) {
        try {
          const saved = localStorage.getItem(key);
          el.checked = saved === null ? dflt : saved === "1";
        } catch (e) { el.checked = dflt; }
        el.addEventListener("change", () => {
          try { localStorage.setItem(key, el.checked ? "1" : "0"); } catch (e) {}
        });
      }
      rememberToggle(headsetEl, "chatHeadset", true);
      rememberToggle(autoSendEl, "chatAutoSend", false);
      rememberToggle(continuousEl, "chatContinuous", false);

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
    """Return the chat page HTML with ``title`` substituted in.

    ``title`` comes from ``CHAT_TITLE``; escape it so a stray ``<`` or ``&`` in
    the configured name can't break out of the tag it lands in.
    """
    return _PAGE.replace("__TITLE__", html.escape(title))
