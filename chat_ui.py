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
      /* Author display rules outrank the UA [hidden] rule, so every
         flex element here would ignore .hidden without this. */
      [hidden] { display:none !important; }
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
      .sources { font-size:0.72rem; color:var(--muted); margin:0.3rem 0.3rem 0; }
      .sources a { color:var(--muted); text-decoration:underline; }
      .sources a:hover { color:var(--text); }
      .webstatus { font-size:0.75rem; color:var(--muted); font-style:italic;
        margin:0 0.3rem 0.3rem; }
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
      #attach, #shot, #camera { font-size:1.05rem; line-height:1; padding:0.5rem 0.6rem; }
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
      /* Rendered markdown. white-space:normal because the renderer supplies the
         structure; pre-wrap would double every blank line. */
      .bubble.md { white-space:normal; }
      .bubble.md > *:first-child { margin-top:0; }
      .bubble.md > *:last-child { margin-bottom:0; }
      .bubble.md p { margin:0.5rem 0; }
      .bubble.md h3, .bubble.md h4, .bubble.md h5, .bubble.md h6 {
        margin:0.7rem 0 0.35rem; font-size:1rem; }
      .bubble.md ul, .bubble.md ol { margin:0.5rem 0; padding-left:1.2rem; }
      .bubble.md li { margin:0.15rem 0; }
      .bubble.md blockquote { margin:0.5rem 0; padding-left:0.7rem;
        border-left:2px solid var(--border); color:var(--muted); }
      .bubble.md a { color:#93c5fd; }
      .bubble.md code { background:rgba(0,0,0,0.35); padding:0.1rem 0.3rem;
        border-radius:0.3rem; font-size:0.88em;
        font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
      .code { position:relative; margin:0.6rem 0; }
      .code pre { margin:0; padding:0.7rem 0.8rem; overflow-x:auto;
        background:#0b1222; border:1px solid var(--border); border-radius:0.6rem; }
      .code pre code { background:none; padding:0; font-size:0.85em; line-height:1.45; }
      .code pre[data-lang]::before { content:attr(data-lang); position:absolute;
        top:0.35rem; left:0.7rem; font-size:0.65rem; color:var(--muted);
        text-transform:uppercase; letter-spacing:0.06em; }
      .code pre[data-lang] { padding-top:1.4rem; }
      .code .copy { position:absolute; top:0.3rem; right:0.35rem; font-size:0.68rem;
        padding:0.15rem 0.4rem; background:var(--panel2); color:var(--muted);
        border:1px solid var(--border); border-radius:0.35rem; opacity:0.75; }
      .code .copy:hover { opacity:1; color:var(--text); }
      #voiceModel { flex:0 1 auto; min-width:0; max-width:16rem; font-size:0.82rem; padding:0.45rem 0.4rem; }
      @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.55;} }
      .hint { color:var(--muted); font-size:0.72rem; margin:0.35rem 0.2rem 0; min-height:1rem; }
      #menu { font-size:1rem; line-height:1; padding:0.35rem 0.55rem; }
      .backdrop { position:fixed; inset:0; background:rgba(0,0,0,0.5); z-index:20; }
      .drawer {
        position:fixed; top:0; left:0; bottom:0; z-index:21; width:min(20rem,86vw);
        background:var(--panel); border-right:1px solid var(--border);
        display:flex; flex-direction:column; padding:0.75rem;
      }
      .drawer-head { display:flex; align-items:center; justify-content:space-between;
        margin-bottom:0.6rem; font-size:0.95rem; }
      .drawer-head button { padding:0.2rem 0.45rem; font-size:0.8rem; }
      .drawer-new { width:100%; margin-bottom:0.6rem; font-weight:600; }
      #convoList { overflow-y:auto; display:flex; flex-direction:column; gap:0.3rem; }
      .convo { display:flex; align-items:stretch; gap:0.2rem; }
      .convo-open {
        flex:1 1 auto; min-width:0; text-align:left; display:flex; flex-direction:column;
        gap:0.1rem; padding:0.4rem 0.5rem; background:var(--panel2);
      }
      .convo.active .convo-open { border-color:var(--accent); }
      .convo-title { font-size:0.82rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .convo-meta { font-size:0.68rem; color:var(--muted); }
      .convo-act { flex:0 0 auto; padding:0 0.4rem; font-size:0.75rem; color:var(--muted); }
      .convo-act:hover { color:var(--text); }
      .convo-empty { color:var(--muted); font-size:0.8rem; margin:0.4rem 0.2rem; }

      /* Phones: reclaim vertical space and stop the header wrapping. */
      @media (max-width: 640px) {
        header { gap:0.5rem; padding:0.5rem 0.7rem; flex-wrap:nowrap; }
        header h1 { font-size:0.95rem; overflow:hidden; text-overflow:ellipsis; }
        .model-label { display:none; }          /* the dropdown speaks for itself */
        #statusText { display:none; }           /* the coloured dot already says it */
        #model { max-width:9rem; font-size:0.8rem; padding:0.35rem 0.4rem; }
        #newChat { font-size:0.8rem; padding:0.35rem 0.5rem; white-space:nowrap; }
        #chat { padding:0.9rem 0.7rem; }
        footer { padding:0.5rem 0.7rem; }
        .col { max-width:92%; }
        textarea { font-size:1rem; }            /* < 1rem makes iOS zoom on focus */
        /* Four controls plus the textarea do not fit a phone: measured at
           360px the input collapsed to 63px, and to a 0px content box at 320px.
           Wrapping is scoped here — applied globally it moves the textarea onto
           its own row on the desktop, which is a redesign, not a fix. */
        .composer { gap:0.35rem; flex-wrap:wrap; }
        #input { flex:1 1 100%; }
        .composer button { padding:0.5rem 0.45rem; }
        button.primary { padding:0.5rem 0.7rem; }
        .voicebar { gap:0.4rem 0.6rem; }
        #webnote { display:none; }   /* the tooltip covers it */
      }
      .insecure { margin:0 0.2rem 0.5rem; }
      .insecure a { color:var(--accent); word-break:break-all; }
    </style>
  </head>
  <body>
    <div class="backdrop" id="backdrop" hidden></div>
    <aside class="drawer" id="drawer" hidden>
      <div class="drawer-head">
        <strong>Conversations</strong>
        <button id="drawerClose" title="Close">✕</button>
      </div>
      <button class="drawer-new" id="drawerNew" type="button">＋ New chat</button>
      <div id="convoList"></div>
    </aside>

    <header>
      <button id="menu" title="Saved conversations" hidden>☰</button>
      <h1>__TITLE__</h1>
      <div class="status"><span class="dot" id="dot"></span><span id="statusText">connecting…</span></div>
      <div class="spacer"></div>
      <label class="status model-label" for="model">Model</label>
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
        <p class="hint insecure" id="insecureNote" hidden></p>
        <div class="voicebar" id="webbar" hidden>
          <label class="voicebar-check" title="Let this app read the web for you: a link in your message is fetched, and otherwise the model is asked whether a search would help. Sources are listed under the reply.">
            <input type="checkbox" id="web"> 🌐 Web access
          </label>
          <span class="voicebar-label" id="webnote">links you paste are read; the model decides when to search</span>
        </div>
        <div class="thumbs" id="thumbs" hidden></div>
        <div class="composer">
          <textarea id="input" rows="1" placeholder="Type a message…"></textarea>
          <button id="attach" title="Attach an image (needs a vision model)">📎</button>
          <button id="camera" title="Take a photo" hidden>📷</button>
          <button id="shot" title="Capture a screenshot to analyse" hidden>📸</button>
          <button id="mic" title="Speak (offline transcription)" hidden>🎤</button>
          <button class="primary" id="send">Send</button>
          <button class="danger" id="stop" hidden>Stop</button>
        </div>
        <input type="file" id="file" accept="image/*" multiple hidden>
        <input type="file" id="cameraFile" accept="image/*" capture="environment" hidden>
        <p class="hint" id="hint"></p>
      </div>
    </footer>

    <script>
      const chatEl   = document.getElementById("chat");
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
      const webEl    = document.getElementById("web");
      const webBar   = document.getElementById("webbar");
      const insecureNote = document.getElementById("insecureNote");
      const drawerEl = document.getElementById("drawer");
      const backdropEl = document.getElementById("backdrop");
      const convoListEl = document.getElementById("convoList");
      const menuBtn  = document.getElementById("menu");
      const attachBtn = document.getElementById("attach");
      const shotBtn  = document.getElementById("shot");
      const fileEl   = document.getElementById("file");
      const cameraBtn = document.getElementById("camera");
      const cameraFileEl = document.getElementById("cameraFile");
      const thumbsEl = document.getElementById("thumbs");
      const modelEl  = document.getElementById("model");
      const dotEl    = document.getElementById("dot");
      const statusEl = document.getElementById("statusText");
      const hintEl   = document.getElementById("hint");

      let messages = [];       // conversation sent to /api/chat for context
      let busy = false;
      let controller = null;   // AbortController for the in-flight stream
      let pendingImages = [];  // [{ b64, url }] attached but not yet sent
      let pendingAutoSend = false;  // an utterance arrived while busy

      function setStatus(state, text) {
        dotEl.className = "dot" + (state ? " " + state : "");
        statusEl.textContent = text;
      }
      function autosize() {
        inputEl.style.height = "auto";
        inputEl.style.height = Math.min(inputEl.scrollHeight, window.innerHeight * 0.4) + "px";
      }
      // Coalesce to one layout per frame. Writing the bubble then reading
      // scrollHeight on every token forces a synchronous re-wrap of the whole
      // reply, which is quadratic in its length.
      let scrollPending = false;
      function scrollDown() {
        if (scrollPending) return;
        scrollPending = true;
        requestAnimationFrame(() => { scrollPending = false; chatEl.scrollTop = chatEl.scrollHeight; });
      }


      // ---- Markdown ----
      // A small renderer rather than a library: the page ships inline with no
      // CDN. Everything is escaped BEFORE any markup is produced, so model
      // output can never introduce a tag — the transforms below only add
      // structure to already-inert text.
      const MD_SENTINEL_RE = new RegExp(String.fromCharCode(1), "g");

      function esc(t) {
        return String(t).replace(MD_SENTINEL_RE, "")
                        .replace(/&/g, "&amp;").replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
      }

      // A placeholder character that cannot appear in model output: esc()
      // strips it, and it is built here rather than written literally.
      const MD_SENTINEL = String.fromCharCode(1);
      const MD_SLOT = new RegExp(MD_SENTINEL + "(\\\\d+)" + MD_SENTINEL, "g");

      function inlineMd(t) {
        // Lift code spans out before anything else runs. Applying the emphasis
        // and link rules to inlineMd's own output means markdown *inside* a code
        // span gets eaten: `*args` rendered as an italic "args", and two spans
        // each containing `*` mis-nested tags across both of them.
        const spans = [];
        let out = t.replace(/`([^`\\n]+)`/g, function (_, code) {
          spans.push(code);
          return MD_SENTINEL + (spans.length - 1) + MD_SENTINEL;
        });
        // Emphasis needs a non-space character inside both delimiters, as
        // CommonMark requires. Without that, prose like "SELECT * FROM a …
        // SELECT * FROM b" turns everything between two unrelated asterisks
        // into <em> and deletes both asterisks from the page.
        out = out
          .replace(/\\*\\*([^\\s*][^*\\n]*[^\\s*]|[^\\s*])\\*\\*/g, "<strong>$1</strong>")
          .replace(/(^|[^*\\w])\\*([^\\s*][^*\\n]*[^\\s*]|[^\\s*])\\*/g, "$1<em>$2</em>")
          .replace(/\\[([^\\]\\n]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)/g,
                   '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        return out.replace(MD_SLOT, function (_, i) {
          return "<code>" + spans[Number(i)] + "</code>";
        });
      }

      // Walk the text line by line rather than deciding a whole blank-line
      // chunk's type at once. The chunk approach needed a blank line before
      // every construct, so the very common "## Summary\\nText" and
      // "Here are steps:\\n- one" rendered their markers literally.
      function renderMarkdown(raw) {
        // Split on any line ending, not just \\n. A stray \\r survives a plain
        // split and then defeats every block pattern below, because "." does not
        // match a carriage return and "$" without the m flag only matches
        // end-of-input — so CRLF output rendered its markers literally.
        const lines = esc(raw).split(/\\r\\n|\\r|\\n/);
        const out = [];
        let para = [];        // paragraph lines awaiting a <br>-joined <p>
        let list = null;      // { tag: "ul"|"ol", items: [], start: n }
        let quote = [];       // blockquote lines
        let fence = null;     // { lang, body: [] } while inside ```

        function flushPara() {
          if (!para.length) return;
          out.push("<p>" + inlineMd(para.join("<br>")) + "</p>");
          para = [];
        }
        function flushList() {
          if (!list) return;
          const attr = list.tag === "ol" && list.start !== 1 ? ' start="' + list.start + '"' : "";
          out.push("<" + list.tag + attr + ">" +
                   list.items.map(function (i) { return "<li>" + inlineMd(i) + "</li>"; }).join("") +
                   "</" + list.tag + ">");
          list = null;
        }
        function flushQuote() {
          if (!quote.length) return;
          out.push("<blockquote>" + inlineMd(quote.join("<br>")) + "</blockquote>");
          quote = [];
        }
        function flushAll() { flushPara(); flushList(); flushQuote(); }
        function emitCode(lang, body) {
          out.push('<div class="code"><button class="copy" type="button">Copy</button>' +
                   "<pre" + (lang ? ' data-lang="' + lang + '"' : "") + "><code>" +
                   body + "</code></pre></div>");
        }

        for (const line of lines) {
          // A fence is only a fence at the start of a line — an inline triple
          // backtick mid-sentence used to swallow the rest of the reply.
          const fenceMatch = line.match(/^\\s*```(.*)$/);
          if (fence) {
            if (fenceMatch) {
              emitCode(fence.lang, fence.body.join("\\n"));
              fence = null;
              // Prose after the closing fence used to be dropped on the floor:
              // the capture was matched and then never read.
              const rest = fenceMatch[1].trim();
              if (rest) para.push(rest);
            } else {
              fence.body.push(line);
            }
            continue;
          }
          if (fenceMatch) {
            flushAll();
            // A fence that opens AND closes on one line — ```pip install foo``` —
            // is a whole code block, not the start of one. Treating it as an
            // opener swallowed the rest of the reply into the block and painted
            // the command itself as the language badge.
            const solo = line.match(/^\\s*```(.*?)```\\s*$/);
            if (solo) {
              emitCode("", solo[1]);
              continue;
            }
            fence = { lang: fenceMatch[1].trim(), body: [] };
            continue;
          }

          if (!line.trim()) { flushAll(); continue; }

          const heading = line.match(/^ {0,3}(#{1,6})\\s+(.*)$/);
          if (heading) {
            flushAll();
            const level = Math.min(6, heading[1].length + 2);
            out.push("<h" + level + ">" + inlineMd(heading[2].trim()) + "</h" + level + ">");
            continue;
          }

          const bullet = line.match(/^\\s*[-*+]\\s+(.*)$/);
          if (bullet) {
            flushPara(); flushQuote();
            if (!list || list.tag !== "ul") { flushList(); list = { tag: "ul", items: [], start: 1 }; }
            list.items.push(bullet[1]);
            continue;
          }

          const numbered = line.match(/^\\s*(\\d+)[.)]\\s+(.*)$/);
          // As CommonMark has it, an ordered list may only interrupt a running
          // paragraph when it starts at 1 — otherwise prose that wraps onto
          // "1908. It changed everything" turns into a list.
          if (numbered && !(para.length && parseInt(numbered[1], 10) !== 1)) {
            flushPara(); flushQuote();
            if (!list || list.tag !== "ol") {
              flushList();
              // Keep the author's first number, so "5. fifth" doesn't renumber
              // to 1 and a year like "1908. Ford…" isn't silently rewritten.
              list = { tag: "ol", items: [], start: parseInt(numbered[1], 10) || 1 };
            }
            list.items.push(numbered[2]);
            continue;
          }

          const quoted = line.match(/^\\s*&gt;\\s?(.*)$/);
          if (quoted) {
            flushPara(); flushList();
            quote.push(quoted[1]);
            continue;
          }

          // A plain line while a list is open continues its last item rather
          // than ending the list — otherwise the interrupt guard above rejects
          // every following marker and half the list renders as raw text.
          if (list && list.items.length) {
            list.items[list.items.length - 1] += " " + line.trim();
            continue;
          }
          flushList(); flushQuote();
          para.push(line.trim());
        }

        // An unterminated fence still renders as code — the reply was cut off,
        // not malformed, and showing it raw would be worse.
        if (fence) emitCode(fence.lang, fence.body.join("\\n"));
        flushAll();
        return out.join("");
      }

      // Painted once the reply completes, not per token: a half-arrived fence
      // renders as garbage, and re-parsing every chunk is wasted work.
      function paintMarkdown(el, raw) {
        el.innerHTML = renderMarkdown(raw);
        el.classList.add("md");
        el.querySelectorAll("button.copy").forEach(function (btn) {
          btn.addEventListener("click", function () {
            const code = btn.parentElement.querySelector("code");
            navigator.clipboard.writeText(code.textContent).then(
              function () {
                btn.textContent = "Copied";
                setTimeout(function () { btn.textContent = "Copy"; }, 1200);
              },
              function () { btn.textContent = "Press Ctrl+C"; }
            );
          });
        });
      }

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

      // Look the placeholder up each time: newChat() replaces the node, so a
      // reference cached at load goes stale after the first conversation.
      function clearPlaceholder() {
        const el = document.getElementById("empty");
        if (el) el.remove();
      }

      function addUser(text, images) {
        clearPlaceholder();
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

      // Cap on the long edge. Only genuinely large sources are reduced; a
      // 1920x1080 screenshot passes through untouched, because halving it is
      // what makes small text unreadable to a vision model.
      const IMG_MAX_DIM = 1920;
      // Above this, the image is a photograph rather than a screen grab, and
      // lossless encoding buys nothing worth the bytes.
      const PNG_BUDGET = 1500000;

      function toCanvas(source, w, h, maxDim) {
        const scale = Math.min(1, maxDim / Math.max(w, h));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(w * scale));
        canvas.height = Math.max(1, Math.round(h * scale));
        canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
        return canvas;
      }

      function approxBytes(dataUrl) {
        return Math.floor((dataUrl.length - dataUrl.indexOf(",") - 1) * 0.75);
      }

      function asAttachment(dataUrl) {
        return { url: dataUrl, b64: dataUrl.slice(dataUrl.indexOf(",") + 1) };
      }

      // PNG first. A screenshot is flat colour and sharp text: it compresses
      // better as PNG than JPEG *and* stays legible, where JPEG rings around
      // every glyph — which is exactly what a vision model then has to read.
      // Photographs blow the budget and fall back to JPEG, where lossy is fine.
      function encodeAttachment(canvas) {
        const png = canvas.toDataURL("image/png");
        if (approxBytes(png) <= PNG_BUDGET) return asAttachment(png);
        // JPEG has no alpha and renders transparency black, so flatten onto
        // white before encoding.
        const flat = document.createElement("canvas");
        flat.width = canvas.width; flat.height = canvas.height;
        const ctx = flat.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, flat.width, flat.height);
        ctx.drawImage(canvas, 0, 0);
        let jpeg = flat.toDataURL("image/jpeg", 0.9);
        // A very large or very noisy photograph can still encode huge. Step the
        // quality down once rather than putting megabytes into every turn of
        // the conversation, which is re-sent in full each time.
        if (approxBytes(jpeg) > 900000) jpeg = flat.toDataURL("image/jpeg", 0.75);
        return asAttachment(jpeg);
      }

      async function toAttachment(file, maxDim = IMG_MAX_DIM) {
        const bmp = await loadBitmap(file);
        const canvas = toCanvas(bmp, bmp.width || bmp.naturalWidth,
                                     bmp.height || bmp.naturalHeight, maxDim);
        if (bmp.close) bmp.close();
        return encodeAttachment(canvas);
      }

      function addAttachment(att) {
        if (pendingImages.length >= 4) { hintEl.textContent = "Up to 4 images per message."; return false; }
        pendingImages.push(att); renderThumbs();
        switchToVisionModel();
        return true;
      }

      attachBtn.addEventListener("click", () => fileEl.click());

      async function takeFiles(input) {
        for (const file of Array.from(input.files)) {
          try { if (!addAttachment(await toAttachment(file))) break; }
          catch (e) { hintEl.textContent = "Could not read " + (file.name || "that image"); }
        }
        input.value = "";   // so re-picking the same file fires change again
      }

      fileEl.addEventListener("change", () => takeFiles(fileEl));
      cameraFileEl.addEventListener("change", () => takeFiles(cameraFileEl));

      // Only offer the camera where there is likely to be one pointed at the
      // world. On a desktop, capture= just opens the same picker 📎 already
      // gives you, so the button would be a duplicate.
      if (window.matchMedia && window.matchMedia("(pointer: coarse)").matches) {
        cameraBtn.hidden = false;
        cameraBtn.addEventListener("click", () => {
          // Check before opening: on a phone this is a full-screen camera
          // intent, and finding out afterwards that there was no room means
          // framing and shooting a photo for nothing.
          if (pendingImages.length >= 4) {
            hintEl.textContent = "Up to 4 images per message.";
            return;
          }
          cameraFileEl.click();
        });
      }

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
          addAttachment(encodeAttachment(
            toCanvas(video, video.videoWidth, video.videoHeight, IMG_MAX_DIM)));
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
      document.addEventListener("paste", async (e) => {
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
        clearPlaceholder();
        const wrap = document.createElement("div");
        wrap.className = "wrap";
        wrap.innerHTML =
          '<div class="msg assistant"><div class="col">' +
            '<div class="role"></div>' +
            '<div class="webstatus" hidden></div>' +
            '<details class="think" hidden><summary>Show thinking</summary>' +
              '<div class="think-body"></div></details>' +
            '<div class="bubble">…</div>' +
            '<div class="meta"></div>' +
            '<div class="sources" hidden></div>' +
          '</div></div>';
        // Model names come from the Ollama server — set as text, never markup.
        wrap.querySelector(".role").textContent = modelEl.value || "Assistant";
        chatEl.appendChild(wrap); scrollDown();
        return {
          root: wrap,
          bubble: wrap.querySelector(".bubble"),
          status: wrap.querySelector(".webstatus"),
          think: wrap.querySelector("details.think"),
          thinkBody: wrap.querySelector(".think-body"),
          meta: wrap.querySelector(".meta"),
          sources: wrap.querySelector(".sources"),
        };
      }

      // Render the pages a reply was grounded in, numbered to match the [n]
      // citations the model is asked to use.
      function showSources(view, list) {
        view.sources.innerHTML = "";
        if (!list || !list.length) { view.sources.hidden = true; return; }
        view.sources.appendChild(document.createTextNode("Sources: "));
        list.forEach((s, i) => {
          if (i) view.sources.appendChild(document.createTextNode(" · "));
          const a = document.createElement("a");
          a.href = s.url; a.target = "_blank"; a.rel = "noopener noreferrer";
          a.textContent = "[" + (i + 1) + "] " + (s.title || s.url);
          a.title = s.url;
          view.sources.appendChild(a);
        });
        view.sources.hidden = false;
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
          modelVision = {};
          for (const m of (data.models || [])) {
            if (m && typeof m === "object") {
              // OCR models can see, but they transcribe rather than reason, so
              // they must never be auto-selected to answer about an image.
              modelVision[m.name || m.model] = !!m.vision && !m.ocr;
            }
          }
          visionDefault = data.vision_default || null;
          ocrAvailable = data.ocr_default || null;
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
          if (data.web) webBar.hidden = false;
          if (data.history) { historyOn = true; menuBtn.hidden = false; refreshConversations(); }
          if (!data.voice) return;
          if (window.isSecureContext) {
            micBtn.hidden = false;
            await loadVoiceModels();
            return;
          }
          // The mic API simply does not exist off a secure origin, so show the
          // exact URL that works rather than a button that can only fail.
          const secureUrl = "https://" + location.hostname + location.pathname;
          insecureNote.innerHTML = "";
          insecureNote.appendChild(document.createTextNode("🎤 Voice needs a secure page. Open "));
          const link = document.createElement("a");
          link.href = secureUrl; link.textContent = secureUrl;
          insecureNote.appendChild(link);
          insecureNote.appendChild(document.createTextNode(" (no port) — this page is plain HTTP."));
          insecureNote.hidden = false;
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
        // Deliberately not saved yet: a failed turn is rolled back on screen and
        // in messages[], and writing first would leave the store holding a turn
        // the UI discarded — which a retry then duplicates. Committed by
        // commitTurn() once the turn has actually produced something.

        const view = addAssistant();
        // Abort handling runs after newChat() may have swapped `messages`;
        // hold the identity so a cancelled reply can't land in a fresh chat.
        const thread = messages;

        // Write the pair together, whatever the outcome: a completed reply, a
        // Stop press, or an error that still produced text. Never one without
        // the other, so the stored thread always alternates the way the live one
        // does. No-ops if this thread was abandoned meanwhile.
        let turnSaved = false;
        async function commitTurn(reply, sources) {
          if (turnSaved || messages !== thread) return;
          turnSaved = true;
          await ensureConversation(text);
          await saveMessage("user", text, userMsg.images || null, null);
          if (reply) await saveMessage("assistant", reply, null, sources || null);
          refreshConversations();
        }
        controller = new AbortController();
        let rawContent = "", thinkingField = "", started = false, usage = null, lastSources = null;

        try {
          const resp = await fetch("api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model: modelEl.value || undefined,
              messages,
              web: webEl.checked || undefined,
            }),
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
              // Web-grounding progress, emitted before the model starts.
              if (obj.status !== undefined) {
                view.status.textContent = obj.status;
                view.status.hidden = !obj.status;
                scrollDown();
                continue;
              }
              if (obj.sources) { lastSources = obj.sources; showSources(view, obj.sources); scrollDown(); continue; }
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
          view.status.hidden = true;
          if (finalContent) paintMarkdown(view.bubble, finalContent);
          else view.bubble.textContent = "(empty response)";
          if (finalContent && messages === thread) messages.push({ role: "assistant", content: finalContent });
          if (finalContent) commitTurn(finalContent, lastSources);
          if (usage) view.meta.textContent = fmtUsage(usage);
          setStatus("ok", "connected");
        } catch (err) {
          if (err.name === "AbortError") {
            const partial = splitThink(rawContent).content;
            view.bubble.textContent = (partial || "") + "  ⏹ stopped";
            if (partial && messages === thread) {
              messages.push({ role: "assistant", content: partial });
              commitTurn(partial, lastSources);   // Stop still produced an answer
            }
          } else {
            // An error can arrive mid-stream, after text is already on screen.
            // Discarding the view would throw away a partial answer; keep it and
            // report alongside. With nothing rendered, drop the user turn too,
            // or the next send posts two user messages in a row.
            const partial = splitThink(rawContent).content;
            if (partial) {
              view.bubble.textContent = partial;
              view.status.textContent = "Interrupted: " + (err.message || err);
              view.status.hidden = false;
              if (messages === thread) messages.push({ role: "assistant", content: partial });
              commitTurn(partial, lastSources);
            } else {
              view.root.remove();
              if (messages === thread && messages.length &&
                  messages[messages.length - 1].role === "user") messages.pop();
              markError(err.message || String(err));
            }
            setStatus("bad", "error");
          }
        } finally {
          busy = false; sendBtn.disabled = false; stopBtn.hidden = true;
          controller = null;
          if (pendingAutoSend) {
            pendingAutoSend = false;
            if (inputEl.value.trim()) { setTimeout(send, 0); return; }
          }
          inputEl.focus();
        }
      }

      function stop() { if (controller) controller.abort(); }

      function newChat() {
        if (controller) controller.abort();
        currentConvoId = null;
        messages = [];
        pendingImages = []; renderThumbs();
        chatEl.innerHTML = '<div class="wrap"><div class="empty" id="empty">' +
          'Send a message to start chatting with your local model.</div></div>';
        inputEl.focus();
      }


      // ---- Conversation history ----
      // The server owns storage so a thread started on one device continues on
      // another; this side just mirrors each completed message into it.
      let currentConvoId = null;
      let historyOn = false;
      let modelVision = {};   // model name -> can it answer about an image
      let visionDefault = null;  // server's pick: smallest non-OCR vision model
      let ocrAvailable = null;   // an installed transcriber, if there is one

      function when(ts) {
        const secs = Math.max(0, Date.now() / 1000 - ts);
        if (secs < 60) return "just now";
        if (secs < 3600) return Math.floor(secs / 60) + "m ago";
        if (secs < 86400) return Math.floor(secs / 3600) + "h ago";
        if (secs < 604800) return Math.floor(secs / 86400) + "d ago";
        return new Date(ts * 1000).toLocaleDateString();
      }

      async function refreshConversations() {
        if (!historyOn) return;
        let list = [];
        try {
          list = (await (await fetch("api/conversations")).json()).conversations || [];
        } catch (e) { return; }

        convoListEl.innerHTML = "";
        if (!list.length) {
          const p = document.createElement("p");
          p.className = "convo-empty";
          p.textContent = "No saved conversations yet.";
          convoListEl.appendChild(p);
          return;
        }
        for (const convo of list) {
          const row = document.createElement("div");
          row.className = "convo" + (convo.id === currentConvoId ? " active" : "");

          const open = document.createElement("button");
          open.className = "convo-open";
          open.type = "button";
          const title = document.createElement("span");
          title.className = "convo-title";
          title.textContent = convo.title;
          const meta = document.createElement("span");
          meta.className = "convo-meta";
          meta.textContent = when(convo.updated_at) + " · " + convo.message_count + " msg";
          open.appendChild(title); open.appendChild(meta);
          open.addEventListener("click", function () { loadConversation(convo.id); });

          const ren = document.createElement("button");
          ren.className = "convo-act"; ren.type = "button";
          ren.textContent = "✎"; ren.title = "Rename";
          ren.addEventListener("click", async function (e) {
            e.stopPropagation();
            const name = prompt("Rename conversation", convo.title);
            if (!name) return;
            await fetch("api/conversations/" + convo.id, {
              method: "PATCH", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ title: name }) });
            refreshConversations();
          });

          const del = document.createElement("button");
          del.className = "convo-act"; del.type = "button";
          del.textContent = "✕"; del.title = "Delete";
          del.addEventListener("click", async function (e) {
            e.stopPropagation();
            if (!confirm("Delete \\"" + convo.title + "\\"?")) return;
            await fetch("api/conversations/" + convo.id, { method: "DELETE" });
            if (convo.id === currentConvoId) newChat();
            refreshConversations();
          });

          row.appendChild(open); row.appendChild(ren); row.appendChild(del);
          convoListEl.appendChild(row);
        }
      }

      // Create on first use rather than on page load, so idly opening the app
      // does not litter the list with empty conversations.
      async function ensureConversation(firstMessage) {
        if (!historyOn || currentConvoId) return currentConvoId;
        try {
          const resp = await fetch("api/conversations", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: firstMessage || "", model: modelEl.value || null }) });
          currentConvoId = (await resp.json()).id;
        } catch (e) { currentConvoId = null; }
        return currentConvoId;
      }

      async function saveMessage(role, content, images, sources) {
        if (!historyOn || !currentConvoId) return;
        try {
          await fetch("api/conversations/" + currentConvoId + "/messages", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ role: role, content: content,
                                   images: images || null, sources: sources || null }) });
        } catch (e) { /* history is a convenience; never block the chat on it */ }
      }

      async function loadConversation(id) {
        let convo;
        try {
          const resp = await fetch("api/conversations/" + id);
          if (!resp.ok) return;
          convo = await resp.json();
        } catch (e) { return; }

        if (controller) controller.abort();
        currentConvoId = convo.id;
        messages = [];
        chatEl.innerHTML = "";
        pendingImages = []; renderThumbs();

        for (const msg of convo.messages || []) {
          const imgs = (msg.images || []).map(function (b64) {
            return { b64: b64, url: "data:image/jpeg;base64," + b64 };
          });
          if (msg.role === "user") {
            addUser(msg.content, imgs);
            const entry = { role: "user", content: msg.content };
            if (imgs.length) entry.images = imgs.map(function (i) { return i.b64; });
            messages.push(entry);
          } else {
            const view = addAssistant();
            if (msg.content) paintMarkdown(view.bubble, msg.content);
            if (msg.sources) showSources(view, msg.sources);
            messages.push({ role: "assistant", content: msg.content });
          }
        }
        if (convo.model && modelEl.querySelector('option[value="' + convo.model + '"]')) {
          modelEl.value = convo.model;
        }
        closeDrawer();
        refreshConversations();
        scrollDown();
      }

      function openDrawer() { drawerEl.hidden = false; backdropEl.hidden = false; refreshConversations(); }
      function closeDrawer() { drawerEl.hidden = true; backdropEl.hidden = true; }

      // ---- Vision routing ----
      // Attaching an image to a text-only model gets you a refusal or an
      // invention, and remembering to switch by hand every time is a papercut.
      function switchToVisionModel() {
        if (!pendingImages.length) return;
        if (modelVision[modelEl.value]) return;   // it can already see

        // An OCR model on the server can transcribe the image for a text-only
        // model, so there is no need to take the conversation away from it —
        // which matters when the model you are on is the one you actually want
        // (a coder model, mid-debugging, with a screenshot of the stack trace).
        if (ocrAvailable) {
          hintEl.textContent = modelEl.value + " can't see images — " + ocrAvailable +
            " will read the text out of it. Pick a vision model if the image isn't text.";
          return;
        }

        // No transcriber installed: the only way to use the image is a model
        // that can see it. The server's pick is the smallest general vision
        // model; fall back to dropdown order if it isn't installed.
        const options = Array.from(modelEl.options).map(function (o) { return o.value; });
        const candidate = (visionDefault && options.indexOf(visionDefault) >= 0)
          ? visionDefault
          : options.find(function (name) { return modelVision[name]; });
        if (!candidate) {
          hintEl.textContent = "No installed model can read images — pull one (e.g. minicpm-v or glm-ocr).";
          return;
        }
        const previous = modelEl.value;
        modelEl.value = candidate;
        hintEl.textContent = "Switched to " + candidate + " to read the image (was " + previous + ").";
      }

      // ---- Voice input (offline, via /api/transcribe) ----
      let recording = false, mediaStream = null, audioCtx = null, srcNode = null, procNode = null, buffers = [];
      let micRate = 16000, bufferedSamples = 0, micStarting = false;
      // ~10 minutes at 16 kHz; a 16-bit mono WAV of that is ~19 MB, inside the
      // server's 25 MB body cap with room for the header.
      const MAX_SAMPLES = 16000 * 600;

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
        buffers = []; bufferedSamples = 0;
        vadSpeechMs = 0; vadSilenceMs = 0; vadHasSpeech = false;
        const wav = encodeWav(mergeBuffers(captured), micRate, 16000);
        if (wav) transcribeBlob(wav);
      }

      async function toggleMic() {
        if (recording) { await stopMic(); return; }
        if (micStarting) return;   // a second tap while permission is pending
        micStarting = true;
        try { await startMic(); } finally { micStarting = false; }
      }

      async function startMic() {
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
        buffers = []; bufferedSamples = 0; vadReset();
        procNode.onaudioprocess = (e) => {
          const chunk = new Float32Array(e.inputBuffer.getChannelData(0));
          buffers.push(chunk);
          bufferedSamples += chunk.length;
          if (continuousEl.checked) { vadStep(chunk); return; }
          // Push-to-talk: no VAD runs, so the idle stop and the length cap have
          // to be enforced here or a mic left on records until the upload 413s.
          if (bufferedSamples >= MAX_SAMPLES) {
            hintEl.textContent = "Reached the maximum recording length — transcribing.";
            setTimeout(stopMic, 0);
          }
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
        buffers = []; bufferedSamples = 0; vadReset();
        try { await audioCtx.close(); } catch (e) {}
        const wav = encodeWav(mergeBuffers(captured), micRate, 16000);
        if (!wav) { hintEl.textContent = ""; return; }
        await transcribeBlob(wav);
      }

      // Post one utterance for transcription. Runs both for a whole push-to-talk
      // recording and for each pause-delimited chunk while continuous is on, so
      // it must never assume the mic has stopped.
      let transcribeQueue = Promise.resolve();

      function transcribeBlob(wav) {
        // Chain rather than fire-and-forget: two utterances in flight can
        // otherwise resolve out of order and swap words in the box.
        transcribeQueue = transcribeQueue.then(() => sendForTranscription(wav));
        return transcribeQueue;
      }

      async function sendForTranscription(wav) {
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
          if (autoSendEl.checked && busy) {
            // Re-checked in send()'s finally, so the utterance is not stranded.
            pendingAutoSend = true;
            hintEl.textContent = "Waiting for the reply to finish…";
          }
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
      rememberToggle(webEl, "chatWeb", false);

      menuBtn.addEventListener("click", openDrawer);
      document.getElementById("drawerClose").addEventListener("click", closeDrawer);
      backdropEl.addEventListener("click", closeDrawer);
      document.getElementById("drawerNew").addEventListener("click", function () {
        newChat(); closeDrawer();
      });

      sendBtn.addEventListener("click", send);
      stopBtn.addEventListener("click", stop);
      newBtn.addEventListener("click", newChat);
      micBtn.addEventListener("click", toggleMic);
      inputEl.addEventListener("input", autosize);
      inputEl.addEventListener("keydown", (e) => {
        // isComposing / keyCode 229: Enter is accepting an IME candidate, not
        // submitting. Sending here would post the raw romaji.
        if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
          e.preventDefault(); send();
        }
      });

      // Keyboard advice only where there is a keyboard; on a phone it wrapped
      // to four lines and pushed the composer off the screen.
      if (window.matchMedia("(min-width: 641px)").matches) {
        inputEl.placeholder = "Type a message…  (Enter to send, Shift+Enter for a new line)";
      }

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
