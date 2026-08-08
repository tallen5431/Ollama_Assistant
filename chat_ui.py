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
    <!-- Inline, so there is no favicon request to 404 on every page load, and a
         home-screen shortcut on a phone gets an icon rather than a blank page. -->
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%232563eb'/%3E%3Cpath d='M8 11h16M8 16h16M8 21h10' stroke='white' stroke-width='2.6' stroke-linecap='round'/%3E%3C/svg%3E">
    <meta name="theme-color" content="#0b1120">
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
      /* Quiet until you look for it — this sits under every reply. */
      .replycopy { font-size:0.68rem; padding:0.15rem 0.45rem; margin:0.25rem 0.3rem 0;
        background:transparent; color:var(--muted); border:1px solid var(--border);
        border-radius:0.35rem; opacity:0.6; }
      .replycopy:hover { opacity:1; color:var(--text); }
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

      /* A comparison table is wider than a phone more often than not, so it
         scrolls inside its own box. Letting it widen the bubble instead would
         put the whole conversation on a horizontal scrollbar. */
      .tablewrap { overflow-x:auto; margin:0.6rem 0; }
      .bubble.md table { border-collapse:collapse; font-size:0.85rem; min-width:100%; }
      .bubble.md th, .bubble.md td {
        border:1px solid var(--border); padding:0.3rem 0.55rem;
        text-align:left; vertical-align:top;
      }
      .bubble.md th { background:var(--panel2); font-weight:600; white-space:nowrap; }
      .bubble.md tbody tr:nth-child(even) td { background:rgba(255,255,255,0.02); }
      .bubble.md hr { border:0; border-top:1px solid var(--border); margin:0.9rem 0; }

      /* One row that scrolls sideways rather than four rows that eat a phone
         screen — the strip has to stay a single line at ten routines. */
      .chips { display:flex; gap:0.35rem; flex:1 1 auto; min-width:0;
               overflow-x:auto; scrollbar-width:none; }
      .chips::-webkit-scrollbar { display:none; }
      .chip { flex:0 0 auto; white-space:nowrap; font-size:0.78rem;
              padding:0.35rem 0.65rem; border-radius:999px;
              background:var(--panel2); color:var(--muted); }
      .chip.active { background:var(--accent); border-color:var(--accent);
                     color:#fff; font-weight:600; }
      .chip-edit { flex:0 0 auto; padding:0.35rem 0.5rem; font-size:0.8rem; color:var(--muted); }
      .drawer-tabs { display:flex; gap:0.3rem; }
      .tab { font-size:0.85rem; padding:0.25rem 0.6rem; color:var(--muted); }
      .tab.active { color:var(--text); border-color:var(--accent); }
      #convoPane, #routinePane { display:flex; flex-direction:column;
                                 flex:1 1 auto; min-width:0; min-height:0; }
      #routineList { overflow-y:auto; display:flex; flex-direction:column; gap:0.3rem; }
      #routineEdit { display:flex; flex-direction:column; gap:0.35rem; overflow-y:auto; }
      #routineEdit label { font-size:0.72rem; color:var(--muted); margin-top:0.2rem; }
      #routineEdit input, #routineEdit select { width:100%; max-width:none; }
      #routineEdit input { font-family:inherit; font-size:0.9rem;
        background:var(--panel2); color:var(--text);
        border:1px solid var(--border); border-radius:0.5rem; padding:0.4rem 0.6rem; }
      #routineEdit textarea { min-height:9rem; max-height:none; font-size:0.85rem; }
      .routine-acts { display:flex; gap:0.4rem; margin-top:0.5rem; }
      .routine-acts button { flex:1 1 auto; }
      /* Whether a photo's own timestamp actually came through. The failure this
         guards is silent otherwise: with photo details off there is no EXIF, no
         metadata turn, and an answer to half the question with nothing on
         screen to say why. */
      .thumb .stamp.muted { opacity:0.55; }
      .thumb .stamp { position:absolute; bottom:0; left:0; padding:0 0.2rem;
        font-size:0.6rem; line-height:1.3; background:rgba(0,0,0,0.65);
        border-radius:0 0.35rem 0 0; }

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
        #routinebar { flex-wrap:nowrap; }   /* .voicebar wraps; this one scrolls */
        .chip { padding:0.5rem 0.75rem; }   /* a wet thumb needs about 38px */
      }
      .insecure { margin:0 0.2rem 0.5rem; }
      .insecure a { color:var(--accent); word-break:break-all; }
    </style>
  </head>
  <body>
    <div class="backdrop" id="backdrop" hidden></div>
    <aside class="drawer" id="drawer" hidden>
      <div class="drawer-head">
        <div class="drawer-tabs">
          <button id="tabChats" class="tab active" type="button">Chats</button>
          <button id="tabRoutines" class="tab" type="button">Routines</button>
        </div>
        <button id="drawerClose" title="Close">✕</button>
      </div>
      <div id="convoPane">
        <button class="drawer-new" id="drawerNew" type="button">＋ New chat</button>
        <div id="convoList"></div>
      </div>
      <!-- A div, not a form: a form in this page can submit and navigate away.
           Each pane owns its own ＋ button, so there is never a question about
           which section it acts on. -->
      <div id="routinePane" hidden>
        <button class="drawer-new" id="routineNew" type="button">＋ New routine</button>
        <div id="routineList"></div>
        <div id="routineEdit" hidden>
          <label for="rName">Name</label>
          <input id="rName" maxlength="40" placeholder="🚗 Trip">
          <label for="rBody">Prompt</label>
          <!-- maxlength to match the store's cap: the operative sentence in a
               prompt is usually the last one, and cutting it server-side in
               silence changes what the routine does on every later use. -->
          <textarea id="rBody" rows="8" spellcheck="false" maxlength="4000"></textarea>
          <label for="rPhotos">Photos to attach</label>
          <select id="rPhotos">
            <option value="0">No photos</option>
            <option value="1">1 photo</option>
            <option value="2">2 photos</option>
            <option value="3">3 photos</option>
            <option value="4">4 photos</option>
          </select>
          <label for="rMeta">📍 Photo details</label>
          <select id="rMeta">
            <option value="">Leave as it is</option>
            <option value="1">Turn on for this routine</option>
            <option value="0">Turn off for this routine</option>
          </select>
          <label for="rWeb">🌐 Web access</label>
          <select id="rWeb">
            <option value="">Leave as it is</option>
            <option value="1">Turn on for this routine</option>
            <option value="0">Turn off for this routine</option>
          </select>
          <p class="convo-empty" id="routineWarn"></p>
          <div class="routine-acts">
            <button class="primary" id="rSave" type="button">Save</button>
            <button id="rCancel" type="button">Cancel</button>
            <button class="danger" id="rDelete" type="button" hidden>Delete</button>
          </div>
        </div>
      </div>
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
        <div class="voicebar" id="exifbar" hidden>
          <label class="voicebar-check" title="Read the date, camera and GPS position a photo carries, and tell the model. The app re-encodes images, which strips this, so turning it off means the location never leaves your phone at all. Set PHOTO_META=0 on the server to make off the default.">
            <input type="checkbox" id="exif"> 📍 Photo details
          </label>
          <span class="voicebar-label" id="exifnote">date, camera and location from the photo itself</span>
        </div>
        <!-- Above the thumbs and the composer, so the footer reads downwards
             as: pick a routine, attach its photos, write the message. -->
        <div class="voicebar" id="routinebar" hidden>
          <div class="chips" id="routineChips"></div>
          <button class="chip chip-edit" id="routineEditBtn" type="button"
                  title="Add or change a saved prompt">⚙</button>
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
      const exifEl   = document.getElementById("exif");
      const exifBar  = document.getElementById("exifbar");
      // Declared with the element, not with the toggle wiring further down:
      // toAttachment() reads it, and a "let" below its use is a temporal
      // dead zone waiting for someone to call that function earlier.
      let exifOn = false;
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
      const routineBar = document.getElementById("routinebar");
      const routineChipsEl = document.getElementById("routineChips");
      const routineEditBtn = document.getElementById("routineEditBtn");
      const convoPaneEl = document.getElementById("convoPane");
      const routinePaneEl = document.getElementById("routinePane");
      const routineListEl = document.getElementById("routineList");
      const routineEditEl = document.getElementById("routineEdit");
      const routineNewBtn = document.getElementById("routineNew");
      const routineWarnEl = document.getElementById("routineWarn");
      const tabChatsEl = document.getElementById("tabChats");
      const tabRoutinesEl = document.getElementById("tabRoutines");
      const rNameEl  = document.getElementById("rName");
      const rBodyEl  = document.getElementById("rBody");
      const rPhotosEl = document.getElementById("rPhotos");
      const rWebEl   = document.getElementById("rWeb");
      const rMetaEl  = document.getElementById("rMeta");
      const rSaveBtn  = document.getElementById("rSave");
      const rDeleteBtn = document.getElementById("rDelete");

      let messages = [];       // conversation sent to /api/chat for context
      let busy = false;
      let controller = null;   // AbortController for the in-flight stream
      let pendingImages = [];  // [{ b64, url }] attached but not yet sent
      let pendingAutoSend = false;  // an utterance arrived while busy
      let routines = [];       // the saved list, held so a chip tap is instant
      // The routine picked but not yet sent: { routine, web, exif }, where
      // web/exif hold the checkbox values its forcing overwrote, or null when
      // it forced nothing. A routine is a stamp on one turn, not a mode — this
      // is the only state it has, and clearing it is the only thing newChat and
      // loadConversation have to remember. Declared here with exifOn for the
      // same reason: renderThumbs() and addAttachment() read it.
      let pendingRoutine = null;
      let editingRoutineId = null;   // open in the editor; "" for a new one

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
      // Follow the stream only while the user is already at the bottom.
      // Scrolling up to re-read the question during a long answer used to drag
      // you back on every token, and the only way out was to press Stop.
      let stickBottom = true;
      let lastAutoTop = -1;      // where our own last auto-scroll left it
      function atBottom() {
        return chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 80;
      }
      chatEl.addEventListener("scroll", () => {
        // Our own auto-follow fires this event too; reacting to it would reset
        // the flag we are trying to honour. Only a scroll we did not cause
        // counts as the user expressing a preference.
        if (lastAutoTop >= 0 && Math.abs(chatEl.scrollTop - lastAutoTop) < 2) return;
        stickBottom = atBottom();
      }, { passive: true });

      function scrollDown(force) {
        if (force) stickBottom = true;
        if (!stickBottom) return;
        if (scrollPending) return;
        scrollPending = true;
        requestAnimationFrame(() => {
          scrollPending = false;
          // Re-checked here, not only above: a frame was already queued when
          // the user scrolled away, and firing it blindly yanked them back and
          // re-armed the follow — which is what made the first fix ineffective.
          if (!stickBottom) return;
          chatEl.scrollTop = chatEl.scrollHeight;
          lastAutoTop = chatEl.scrollTop;
        });
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

      // ---- Maths written as TeX ----
      // Models reach for LaTeX the moment an answer contains arithmetic, and
      // this page has no maths renderer: no build step and nothing from a CDN,
      // so KaTeX is not on the table for a handful of subtractions. Unwrapping
      // it into plain text gives what a person reads anyway — "68 / 3.13 ≈ 21.7"
      // beats a raw \\frac on a phone, and beats it by a mile unrendered.
      const TEX_SYMBOLS = [
        [/\\\\approx/g, "≈"], [/\\\\times/g, "×"], [/\\\\cdot/g, "·"],
        [/\\\\div/g, "÷"], [/\\\\pm/g, "±"], [/\\\\mp/g, "∓"],
        [/\\\\leq?\\b/g, "≤"], [/\\\\geq?\\b/g, "≥"], [/\\\\neq/g, "≠"],
        [/\\\\rightarrow/g, "→"], [/\\\\to\\b/g, "→"], [/\\\\infty/g, "∞"],
        [/\\\\degree|\\\\deg\\b/g, "°"],
      ];

      function texToText(body) {
        let out = body;
        // \\text{…} and its relatives are pure wrapping; \\frac wants to become a
        // division that reads left to right. Repeated because these nest.
        for (let pass = 0; pass < 4; pass++) {
          out = out.replace(/\\\\(?:text|mathrm|mathbf|mathit|operatorname)\\s*\\{([^{}]*)\\}/g, "$1");
          out = out.replace(/\\\\(?:d|t)?frac\\s*\\{([^{}]*)\\}\\s*\\{([^{}]*)\\}/g, "($1) / ($2)");
          out = out.replace(/\\\\sqrt\\s*\\{([^{}]*)\\}/g, "√($1)");
        }
        for (const pair of TEX_SYMBOLS) out = out.replace(pair[0], pair[1]);
        out = out.replace(/\\\\left|\\\\right/g, "")
                 .replace(/\\\\[,;:!]/g, " ")
                 .replace(/\\\\\\\\/g, " ")
                 .replace(/\\\\([%$&#_{}])/g, "$1")
                 // Brackets the fraction rule added, dropped again where neither
                 // side needs them: "(68) / (3.13)" is worse than "68 / 3.13".
                 .replace(/\\(([^()\\s]+)\\) \\/ \\(([^()\\s]+)\\)/g, "$1 / $2");
        return out.replace(/\\s+/g, " ").trim();
      }

      // Whether a $…$ pair is maths or two prices in one sentence. Both turn up
      // constantly and they look identical to a regex, so this leans on the one
      // convention TeX actually has: the delimiters hug their contents.
      function looksLikeMaths(body) {
        // A backslash settles it — no price contains \\frac or \\approx.
        if (body.indexOf("\\\\") >= 0) return true;
        // "It cost $100,407 and then $5 more" — the span between the two signs
        // is prose, and prose ends in a space. TeX never opens or closes on one.
        if (/^\\s|\\s$/.test(body)) return false;
        // "$5-$10" is a price range: the span ends on the operator rather than
        // on something for it to operate on.
        if (/^[-+*/=<>,^_]|[-+*/=<>,^_]$/.test(body)) return false;
        // Past that it has to actually do something to be maths at all, or
        // "$100,407$" alone would lose its signs on a guess.
        return /[-+*/=<>^_≈±×÷]/.test(body);
      }

      function unTex(t) {
        return t.replace(
          /\\$\\$([\\s\\S]+?)\\$\\$|\\$([^$\\n]+?)\\$|\\\\\\(([\\s\\S]+?)\\\\\\)|\\\\\\[([\\s\\S]+?)\\\\\\]/g,
          function (whole, a, b, c, d) {
            const dollars = a !== undefined ? a : b;
            if (dollars !== undefined) {
              if (!looksLikeMaths(dollars)) return whole;
              return texToText(dollars);
            }
            // \\( \\) and \\[ \\] say what they are; nothing else uses them.
            return texToText(c !== undefined ? c : d);
          });
      }

      // ---- Tables ----
      // A model asked to compare two things reaches for a pipe table almost
      // every time, and without this the whole grid came out as one paragraph
      // of pipes and dashes joined by <br> — which is how the odometer answer
      // arrived: correct numbers, unreadable layout.
      const TABLE_DIVIDER_RE = /^\\s*\\|?(?:\\s*:?-{1,}:?\\s*\\|)+\\s*:?-{1,}:?\\s*\\|?\\s*$/;

      function tableCells(line) {
        let row = line.trim();
        // The outer pipes are optional in the format models actually emit.
        if (row.slice(0, 1) === "|") row = row.slice(1);
        if (row.slice(-1) === "|") row = row.slice(0, -1);
        return row.split("|").map(function (c) { return c.trim(); });
      }

      function alignOf(spec) {
        const left = spec.slice(0, 1) === ":", right = spec.slice(-1) === ":";
        if (left && right) return " style=\\"text-align:center\\"";
        if (right) return " style=\\"text-align:right\\"";
        return "";
      }

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
        out = unTex(out);
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

        for (let ln = 0; ln < lines.length; ln++) {
          const line = lines[ln];
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

          // A thematic break: three or more of the same mark on a line of its
          // own. It arrived as a literal "***" paragraph between two sections.
          if (/^ {0,3}([-*_])(?:\\s*\\1){2,}\\s*$/.test(line)) {
            flushAll();
            out.push("<hr>");
            continue;
          }

          // A pipe table, but only when the next line is the divider — that is
          // what tells a real table from prose that happens to contain a pipe.
          if (line.indexOf("|") >= 0 && ln + 1 < lines.length &&
              TABLE_DIVIDER_RE.test(lines[ln + 1])) {
            const head = tableCells(line);
            const align = tableCells(lines[ln + 1]).map(alignOf);
            const rows = [];
            let at = ln + 2;
            while (at < lines.length && lines[at].trim() && lines[at].indexOf("|") >= 0) {
              rows.push(tableCells(lines[at]));
              at += 1;
            }
            // Only if the divider agrees with the header about the column
            // count; otherwise this is prose and belongs in a paragraph.
            if (align.length === head.length) {
              flushAll();
              const cell = function (tag, text, i) {
                return "<" + tag + (align[i] || "") + ">" + inlineMd(text) + "</" + tag + ">";
              };
              out.push('<div class="tablewrap"><table><thead><tr>' +
                head.map(function (h, i) { return cell("th", h, i); }).join("") +
                "</tr></thead><tbody>" +
                rows.map(function (r) {
                  // Pad or trim to the header, so one short row cannot shear
                  // the rest of the grid sideways.
                  const cells = [];
                  for (let i = 0; i < head.length; i++) cells.push(cell("td", r[i] || "", i));
                  return "<tr>" + cells.join("") + "</tr>";
                }).join("") +
                "</tbody></table></div>");
              ln = at - 1;
              continue;
            }
          }

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
      // Telling someone to press Ctrl+C is only useful if the text is selected.
      function selectText(node) {
        try {
          const range = document.createRange();
          range.selectNodeContents(node);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
        } catch (err) { /* selection is a nicety, never a failure */ }
      }
      const selectCode = selectText;

      // renders as garbage, and re-parsing every chunk is wasted work.
      function paintMarkdown(el, raw) {
        el.innerHTML = renderMarkdown(raw);
        el.classList.add("md");
        el.querySelectorAll("button.copy").forEach(function (btn) {
          btn.addEventListener("click", function () {
            const code = btn.parentElement.querySelector("code");
            // The Clipboard API is [SecureContext]-gated, so it is undefined
            // over plain http to the box by name or IP. Dereferencing it threw
            // before any promise existed, which made the rejection fallback
            // below unreachable — and invisible on localhost, which counts as
            // a trustworthy origin.
            if (!navigator.clipboard) { selectCode(code); btn.textContent = "Press Ctrl+C"; return; }
            navigator.clipboard.writeText(code.textContent).then(
              function () {
                btn.textContent = "Copied";
                setTimeout(function () { btn.textContent = "Copy"; }, 1200);
              },
              function () { selectCode(code); btn.textContent = "Press Ctrl+C"; }
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
        // A closing tag with no opening one: Ollama's deepseek-r1 template opens
        // <think> in the prompt, so the reply starts inside the scratchpad and
        // only the closing tag comes back.
        //
        // The tag must be ALONE ON ITS LINE, which is how a template emits it.
        // Matching it anywhere meant any reply that merely mentioned the tag —
        // "use `</think>` to close it" — had everything before it moved into
        // the reasoning panel and dropped from the answer, and the truncated
        // text was what got written to history. Permanent and invisible.
        //
        // Only when no opening tag was found at all, and only when something
        // follows: a reply that is nothing but scratchpad is better shown whole
        // than blanked.
        if (!thinking) {
          const orphan = content.match(/^[^\\S\\n]*<\\/think>[^\\S\\n]*$/m);
          // Not inside a fenced code block. Ask a coder model what a reasoning
          // model's output looks like and it shows you one — tag on its own
          // line, inside ``` — and treating that as a real terminator threw
          // away the half of the reply that explained it. An odd number of
          // fences before the tag means we are inside one.
          const fenced = orphan &&
            (content.slice(0, orphan.index).match(/^[^\\S\\n]*```/gm) || []).length % 2 === 1;
          if (orphan && orphan.index > 0 && !fenced) {
            const after = content.slice(orphan.index + orphan[0].length);
            if (after.trim()) {
              thinking = content.slice(0, orphan.index);
              content = after;
            }
          }
        }
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
        chatEl.appendChild(wrap); scrollDown(true);   // your own message: always jump
        return wrap;   // so a turn that produced nothing can be rolled back
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
          rm.addEventListener("click", () => {
            pendingImages.splice(i, 1); renderThumbs(); routineProgress();
          });
          cell.appendChild(el); cell.appendChild(rm);
          // What was actually read, on the thumbnail, before you send. The
          // failure this exists for is silent otherwise: "photo details is on"
          // and "this photo has a position in it" are different things, and
          // until now the difference only showed up in the answer.
          const badges = [];
          if (img.meta && img.meta.taken) badges.push(["🕘", "date read from the photo"]);
          const hasPlace = img.meta && typeof img.meta.lat === "number" &&
                           typeof img.meta.lon === "number" &&
                           !(img.meta.lat === 0 && img.meta.lon === 0);
          if (hasPlace) badges.push(["📍", "location read from the photo"]);
          if (badges.length) {
            const stamp = document.createElement("span");
            stamp.className = "stamp";
            stamp.textContent = badges.map(b => b[0]).join("");
            stamp.title = badges.map(b => b[1]).join(", ");
            cell.appendChild(stamp);
          } else if (img.meta) {
            const stamp = document.createElement("span");
            stamp.className = "stamp muted";
            stamp.textContent = "·";
            stamp.title = "this photo carries no date or position";
            cell.appendChild(stamp);
          }
          thumbsEl.appendChild(cell);
        });
      }

      // ---- EXIF ----
      // Read before the canvas re-encode, which is the only chance: drawing an
      // image onto a canvas and reading it back produces clean pixels with no
      // metadata at all, so nothing downstream could recover this.
      //
      // Hand-rolled rather than a library: this is a few hundred bytes at the
      // front of a JPEG, the app ships no build step and loads nothing from a
      // CDN, and the alternative is a dependency for a couple of dozen tags.
      const EXIF_HEAD_BYTES = 256 * 1024;   // EXIF lives at the very front

      function readExif(file) {
        if (!file || !file.slice || !window.DataView) return Promise.resolve(null);
        return file.slice(0, EXIF_HEAD_BYTES).arrayBuffer()
          .then(buf => parseExif(new DataView(buf)))
          .catch(() => null);
      }

      function parseExif(view) {
        if (view.byteLength < 4 || view.getUint16(0) !== 0xffd8) return null;   // not a JPEG
        // Walk the marker segments looking for APP1 with an "Exif\\0\\0" payload.
        let offset = 2;
        while (offset + 4 <= view.byteLength) {
          if (view.getUint8(offset) !== 0xff) return null;   // desynchronised
          let marker = view.getUint8(offset + 1);
          // Any number of 0xFF bytes may pad the gap before a marker, and the
          // standard says to skip them. Treating one as the marker itself read
          // a nonsense segment length and lost the whole file — every tag, not
          // just the padded segment.
          while (marker === 0xff && offset + 2 < view.byteLength) {
            offset += 1;
            marker = view.getUint8(offset + 1);
          }
          if (marker === 0xda || marker === 0xd9) return null;   // image data starts
          if (offset + 4 > view.byteLength) return null;
          const size = view.getUint16(offset + 2);
          if (size < 2) return null;
          if (marker === 0xe1 && offset + 10 <= view.byteLength &&
              view.getUint32(offset + 4) === 0x45786966) {
            return readTiff(view, offset + 10);
          }
          offset += 2 + size;
        }
        return null;
      }

      // How a value that is a ratio of two integers should read to a person.
      function ratio(value, digits) {
        if (typeof value !== "number" || !isFinite(value)) return null;
        return Math.round(value * Math.pow(10, digits)) / Math.pow(10, digits);
      }

      const ORIENTATIONS = {
        1: "upright", 2: "mirrored", 3: "upside down", 4: "mirrored and upside down",
        5: "mirrored and rotated 90° left", 6: "rotated 90° clockwise",
        7: "mirrored and rotated 90° right", 8: "rotated 90° anticlockwise",
      };

      function readTiff(view, base) {
        if (base + 8 > view.byteLength) return null;
        const order = view.getUint16(base);
        if (order !== 0x4949 && order !== 0x4d4d) return null;
        const le = order === 0x4949;                       // "II" is little-endian
        if (view.getUint16(base + 2, le) !== 0x002a) return null;

        const out = {};
        const ifd0 = readIfd(view, base, base + view.getUint32(base + 4, le), le);
        if (!ifd0) return null;

        const make = ifd0[0x010f], model = ifd0[0x0110];
        const camera = [make, model].filter(Boolean).join(" ").trim();
        // "Google Pixel 8" rather than "Google Google Pixel 8": makers often
        // repeat the brand in the model.
        if (camera) out.camera = model && make && model.indexOf(make) === 0 ? model : camera;
        if (ifd0[0x0131]) out.software = String(ifd0[0x0131]).trim();
        if (ifd0[0x013b]) out.artist = String(ifd0[0x013b]).trim();
        if (ifd0[0x8298]) out.copyright = String(ifd0[0x8298]).trim();
        if (ORIENTATIONS[ifd0[0x0112]] && ifd0[0x0112] !== 1) {
          out.orientation = ORIENTATIONS[ifd0[0x0112]];
        }

        if (ifd0[0x8769] !== undefined) {
          const exif = readIfd(view, base, base + ifd0[0x8769], le) || {};
          // DateTimeOriginal is when the shutter fired; DateTime is when the
          // file was last written, which an edit or a copy can change.
          const stamp = exif[0x9003] || exif[0x9004] || ifd0[0x0132];
          if (stamp) out.taken = String(stamp).trim();
          // The one tag that makes a capture time unambiguous. Without it two
          // photos either side of a time-zone change are hours apart in a way
          // nothing downstream can detect.
          const offset = exif[0x9010] || exif[0x9011] || exif[0x9012];
          if (offset) out.offset = String(offset).trim();

          const width = exif[0xa002], height = exif[0xa003];
          if (typeof width === "number" && typeof height === "number") {
            out.width = width;
            out.height = height;
          }
          if (exif[0xa434]) out.lens = String(exif[0xa434]).trim();
          if (typeof exif[0x8827] === "number") out.iso = exif[0x8827];
          const shutter = ratio(exif[0x829a], 6);
          if (shutter) {
            // "1/120" is how a shutter speed is written and read; 0.008333 is
            // the same number and tells nobody anything.
            out.exposure = shutter < 1 ? "1/" + Math.round(1 / shutter) + " s"
                                       : ratio(shutter, 1) + " s";
          }
          const aperture = ratio(exif[0x829d], 1);
          if (aperture) out.aperture = "f/" + aperture;
          const focal = ratio(exif[0x920a], 1);
          if (focal) out.focal = focal + " mm";
          if (typeof exif[0xa405] === "number" && exif[0xa405]) {
            out.focal35 = exif[0xa405] + " mm";
          }
          // Bit 0 of Flash is whether it actually fired; the rest is about
          // return detection and modes, which nobody asks about.
          if (typeof exif[0x9209] === "number") out.flash = (exif[0x9209] & 1) === 1;
        }

        if (ifd0[0x8825] !== undefined) {
          readGps(view, base, base + ifd0[0x8825], le, out);
        }
        return Object.keys(out).length ? out : null;
      }

      function readGps(view, base, at, le, out) {
        const gps = readIfd(view, base, at, le) || {};
        const lat = dms(gps[0x0002], gps[0x0001]);
        const lon = dms(gps[0x0004], gps[0x0003]);
        if (lat !== null && lon !== null) { out.lat = lat; out.lon = lon; }
        if (typeof gps[0x0006] === "number") {
          out.altitude = Math.round(gps[0x0006] * (gps[0x0005] === 1 ? -1 : 1));
        }
        // How far out the fix might be, in metres. Worth having: "at 44.9778,
        // -93.2650" reads as a doorstep and can be a whole block.
        const error = ratio(gps[0x001f], 1);
        if (error) out.accuracy = error;
        const speed = ratio(gps[0x000d], 1);
        if (speed !== null && speed > 0) {
          const unit = { K: "km/h", M: "mph", N: "knots" }[String(gps[0x000c] || "K")[0]];
          out.speed = speed + " " + (unit || "km/h");
        }
        const heading = ratio(gps[0x0011], 0);
        if (heading !== null && gps[0x0011] !== undefined) {
          out.heading = heading + (String(gps[0x0010] || "T")[0] === "M" ? "° magnetic" : "°");
        }
        // GPS records UTC. Alongside a local DateTimeOriginal that is a time
        // zone, which is the one thing an elapsed-time answer needs and the one
        // thing the capture time alone cannot give.
        const time = gps[0x0007], date = gps[0x001d];
        if (Array.isArray(time) && time.length >= 3) {
          const pad = (n) => (n < 10 ? "0" : "") + n;
          const clock = pad(Math.round(time[0])) + ":" + pad(Math.round(time[1])) +
                        ":" + pad(Math.round(time[2]));
          out.utc = date ? String(date).trim() + " " + clock : clock;
        }
      }

      // One IFD as {tag: value}. Only the types these tags actually use.
      function readIfd(view, base, at, le) {
        if (at + 2 > view.byteLength) return null;
        const count = view.getUint16(at, le);
        const out = {};
        for (let i = 0; i < count; i++) {
          const entry = at + 2 + i * 12;
          if (entry + 12 > view.byteLength) break;
          const tag = view.getUint16(entry, le);
          const type = view.getUint16(entry + 2, le);
          const length = view.getUint32(entry + 4, le);
          const width = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 }[type];
          if (!width) continue;
          const bytes = width * length;
          // Values of four bytes or fewer are stored in the entry itself.
          const at2 = bytes <= 4 ? entry + 8 : base + view.getUint32(entry + 8, le);
          if (at2 < 0 || at2 + bytes > view.byteLength) continue;
          out[tag] = readValue(view, at2, type, length, le);
        }
        return out;
      }

      function readValue(view, at, type, length, le) {
        if (type === 2) {                                   // ASCII
          let s = "";
          for (let i = 0; i < length; i++) {
            const c = view.getUint8(at + i);
            if (!c) break;
            s += String.fromCharCode(c);
          }
          return s;
        }
        if (type === 5 || type === 10) {                    // RATIONAL
          const read = (o) => type === 5
            ? view.getUint32(o, le) / (view.getUint32(o + 4, le) || 1)
            : view.getInt32(o, le) / (view.getInt32(o + 4, le) || 1);
          if (length === 1) return read(at);
          const out = [];
          for (let i = 0; i < length; i++) out.push(read(at + i * 8));
          return out;
        }
        if (type === 3) return view.getUint16(at, le);
        if (type === 4 || type === 9) return view.getUint32(at, le);
        if (type === 1 || type === 7) return view.getUint8(at);
        return null;
      }

      // Degrees/minutes/seconds plus a N/S/E/W reference, to a signed decimal.
      function dms(parts, ref) {
        if (!Array.isArray(parts) || parts.length < 3) return null;
        const value = parts[0] + parts[1] / 60 + parts[2] / 3600;
        if (!isFinite(value)) return null;
        const sign = /^[SW]/i.test(String(ref || "")) ? -1 : 1;
        return Math.round(value * sign * 1e6) / 1e6;
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
        // Before the canvas: re-encoding produces clean pixels with no metadata,
        // so this is the only point at which it still exists.
        const meta = exifOn ? await readExif(file) : null;
        const bmp = await loadBitmap(file);
        const canvas = toCanvas(bmp, bmp.width || bmp.naturalWidth,
                                     bmp.height || bmp.naturalHeight, maxDim);
        if (bmp.close) bmp.close();
        const att = encodeAttachment(canvas);
        if (meta) att.meta = meta;
        return att;
      }

      function addAttachment(att) {
        if (pendingImages.length >= 4) { hintEl.textContent = "Up to 4 images per message."; return false; }
        pendingImages.push(att); renderThumbs();
        // Before switchToVisionModel, which writes the same hint line: a model
        // that cannot see the image is the more urgent message and should win.
        routineProgress();
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
        chatEl.appendChild(wrap); scrollDown(true);
        const view = {
          root: wrap,
          bubble: wrap.querySelector(".bubble"),
          status: wrap.querySelector(".webstatus"),
          think: wrap.querySelector("details.think"),
          thinkBody: wrap.querySelector(".think-body"),
          meta: wrap.querySelector(".meta"),
          sources: wrap.querySelector(".sources"),
          raw: "",
        };
        addReplyCopy(view);
        return view;
      }

      // Copying a whole reply had no affordance at all — only individual code
      // blocks did. The raw markdown, not the rendered text, because that is
      // what pastes usefully into notes or an editor.
      function addReplyCopy(view) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "replycopy";
        btn.textContent = "Copy reply";
        btn.hidden = true;
        function flash(label) {
          btn.textContent = label;
          // Always restore. Both failure labels used to stick forever, so the
          // only way back to a usable button was reloading the page.
          setTimeout(() => { btn.textContent = "Copy reply"; }, 1600);
        }
        btn.addEventListener("click", async () => {
          const text = view.raw || view.bubble.textContent || "";
          if (!text) return;
          // Share beats clipboard on a phone: it reaches the apps you would
          // actually paste into, and works without a secure-context clipboard.
          if (navigator.share && window.matchMedia &&
              window.matchMedia("(pointer: coarse)").matches) {
            try { await navigator.share({ text: text }); return; }
            catch (e) {
              // Dismissing the share sheet rejects too. That is not a failure
              // to report — falling through to "select and copy" made cancel
              // look like an error.
              if (e && e.name === "AbortError") return;
            }
          }
          if (!navigator.clipboard) { selectText(view.bubble); flash("Press Ctrl+C"); return; }
          try {
            await navigator.clipboard.writeText(text);
            flash("Copied");
          } catch (e) { selectText(view.bubble); flash("Press Ctrl+C"); }
        });
        view.meta.parentElement.insertBefore(btn, view.meta.nextSibling);
        view.copyBtn = btn;
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
        chatEl.appendChild(wrap); scrollDown(true);   // an error must not be missed
      }

      async function loadModels() {
        try {
          const resp = await fetch("api/models");
          const data = await resp.json();
          // The 502 this endpoint returns when Ollama is unreachable is a
          // perfectly well-formed JSON body, so parsing it without checking
          // read as "connected, zero models" — green dot, a picker seeded with
          // the configured default, and on a phone (#statusText is hidden
          // there) the dot was the only signal and it was lying.
          if (!resp.ok || data.error) {
            modelEl.innerHTML = "";
            const o = document.createElement("option");
            o.textContent = data.default || "(no models)"; o.value = data.default || "";
            modelEl.appendChild(o);
            setStatus("bad", "no connection");
            hintEl.textContent = data.error ||
              "Could not reach the model server. Check Ollama is running and OLLAMA_HOST is set.";
            return;
          }
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
          // Read BEFORE emptying the <select> — clearing its options resets
          // .value to "", which made the whole "keep what is selected" branch
          // below dead code and let a tab resume switch models anyway.
          const current = modelEl.value;
          modelEl.innerHTML = "";
          if (!list.length) {
            const o = document.createElement("option");
            o.textContent = data.default || "(no models found)"; o.value = data.default || "";
            modelEl.appendChild(o);
          } else {
            // Whatever was selected wins: this runs again on every tab resume,
            // and rebuilding from the remembered value silently discarded the
            // model a loaded conversation had restored, or one an image
            // auto-switch had chosen — both assign modelEl.value without
            // firing "change". Then this device's remembered pick, then the
            // server's default, which is one value for every device.
            const preferred =
              list.indexOf(current) >= 0 ? current
              : list.indexOf(remembered("model")) >= 0 ? remembered("model")
              : data.default;
            for (const name of list) {
              const o = document.createElement("option");
              o.value = name; o.textContent = name;
              if (name === preferred) o.selected = true;
              modelEl.appendChild(o);
            }
          }
          setStatus("ok", "connected");
          // Not over a live warning: this now runs on every tab resume, and
          // overwriting the hint wiped the "history is not being saved" notice
          // — which, because it only fires once, then never came back. That is
          // the silent failure the notice exists to prevent.
          if (!historyBroken) {
            hintEl.textContent = list.length ? (list.length + " model(s) available") : "";
          }
        } catch (err) {
          setStatus("bad", "no connection");
          hintEl.textContent = "Could not reach the model server. Check Ollama is running and OLLAMA_HOST is set.";
        }
      }

      async function checkVoice() {
        try {
          const data = await (await fetch("api/health")).json();
          if (data.web) webBar.hidden = false;
          exifBar.hidden = false;
          // The deployment picks the default, not the page: on a box only you
          // can reach this is plainly useful on, and PHOTO_META=0 turns it
          // round. A browser that has used the toggle keeps its own answer.
          // Not while a routine is armed: this runs again when the tab is
          // resumed, and it would otherwise lower exifOn back underneath a
          // routine between picking it and attaching the photos.
          if (!pendingRoutine && !chosen("chatExif") &&
              typeof data.photo_meta === "boolean") {
            exifEl.checked = data.photo_meta;
            exifOn = data.photo_meta;
          }
          if (typeof data.image_turns === "number") KEEP_IMAGE_TURNS = data.image_turns;
          if (data.history) {
            historyOn = true; menuBtn.hidden = false;
            refreshConversations(); refreshRoutines();
          }
          // A default naming a model that has been deleted or renamed since is
          // otherwise a silent failure on every turn until someone works it out.
          if (data.ollama_reachable && data.model_count && !data.default_installed) {
            hintEl.textContent = "The configured default model (" + data.default_model +
              ") is not installed — pick one from the dropdown.";
          }
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
          // This now re-runs on every tab resume, so a mid-session language
          // pick would be rebuilt away. Keep what is selected, then what this
          // device chose before, then the server's default.
          const wanted = voiceSel.value || remembered("voice") || data.default;
          voiceSel.innerHTML = "";
          const seen = new Set();
          for (const m of (data.available || [])) {
            seen.add(m.id);
            voiceSel.appendChild(new Option(m.label, m.id, false, m.id === wanted));
          }
          for (const m of (data.catalog || [])) {
            if (m.downloaded || seen.has(m.id)) continue;
            const o = new Option(m.label + " — " + m.size + " ⬇", m.id);
            o.dataset.download = "1";
            voiceSel.appendChild(o);
          }
          voiceBar.hidden = voiceSel.options.length === 0;
          // On a fresh install nothing is downloaded, so every option is a
          // catalog entry and the browser auto-selects the first — which fires
          // no "change", so the download the picker relies on never happens and
          // the mic can only return an error. Say what to do instead.
          const chosen = voiceSel.selectedOptions[0];
          if (chosen && chosen.dataset.download === "1") {
            hintEl.textContent = "Pick a language to download it — the mic needs one first.";
          }
        } catch (e) { /* leave picker hidden */ }
      }

      // Downloading a not-yet-present language happens on selection so the mic
      // is ready before you speak.
      voiceSel.addEventListener("change", async () => {
        remember("voice", voiceSel.value);
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

      // Feature-detected, like isSecureContext elsewhere: Wake Lock is
      // Chromium-and-Safari-only and secure-context gated, and it is a nicety —
      // never let its absence, or a rejection, break a turn.
      let wakeLock = null;
      let wakeLockPending = false;
      // Bumped by every release. A request that resolves after its turn has
      // already ended — the visibilitychange handler firing just as the reply
      // lands — would otherwise store a sentinel that nothing goes on to
      // release, and the phone screen stays on indefinitely.
      let wakeLockEpoch = 0;
      async function acquireWakeLock() {
        // The in-flight guard matters for continuous voice, where turn N's
        // release and turn N+1's request overlap: without it a late release
        // event from the old sentinel nulled the handle for the new one, and
        // the phone screen then stayed on with nothing able to release it.
        if (wakeLock || wakeLockPending || !navigator.wakeLock) return;
        wakeLockPending = true;
        const epoch = wakeLockEpoch;
        try {
          const sentinel = await navigator.wakeLock.request("screen");
          if (epoch !== wakeLockEpoch) {
            // Released while we were asking. Hand it straight back.
            try { sentinel.release(); } catch (e) {}
            return;
          }
          sentinel.addEventListener("release", () => {
            if (wakeLock === sentinel) wakeLock = null;   // only its own handle
          });
          wakeLock = sentinel;
        } catch (e) { /* a nicety; never break a turn over it */ }
        finally { wakeLockPending = false; }
      }
      function releaseWakeLock() {
        wakeLockEpoch += 1;
        const sentinel = wakeLock;
        wakeLock = null;
        if (!sentinel) return;
        try { sentinel.release(); } catch (e) {}
      }
      document.addEventListener("visibilitychange", () => {
        // The browser drops the lock when the page is hidden; take it back if
        // the turn is still running when we come into view again.
        if (document.visibilityState === "visible" && busy) acquireWakeLock();
      });

      // Send image bytes only for the most recent turn that has any. They were
      // re-uploaded in full on every subsequent turn — measured at a 400 KB
      // body for a message whose text was 714 bytes — over a phone connection,
      // for a model that had already read them. The server applies the same
      // rule; this just stops the bytes crossing the network at all.
      // Server-configured (CHAT_IMAGE_TURNS), not hardcoded: the browser strips
      // before the request leaves the phone, so a hardcoded 1 here left the
      // documented knob with nothing to keep — raising it did nothing at all.
      let KEEP_IMAGE_TURNS = 1;
      function withRecentImages(list) {
        const out = list.slice();
        let kept = 0;
        for (let i = out.length - 1; i >= 0; i--) {
          if (!out[i] || !out[i].images || !out[i].images.length) continue;
          kept += 1;
          if (kept > KEEP_IMAGE_TURNS) {
            const copy = Object.assign({}, out[i]);
            delete copy.images;
            delete copy.image_meta;   // it describes photos that are no longer here
            out[i] = copy;
          }
        }
        return out;
      }

      async function send() {
        const text = inputEl.value.trim();
        const images = pendingImages.slice();
        // Reports whether the turn committed — not merely that it started. A
        // caller that armed something for this turn (the routine guard) must
        // not spend it on a refusal, and must not spend it on a turn that was
        // rolled back either: the message comes back to the composer, so the
        // routine has to still be holding its toggles and its photo count for
        // the retry the app has just invited.
        if ((!text && !images.length) || busy) return false;
        let went = true, queued = false;
        busy = true; sendBtn.disabled = true; stopBtn.hidden = false;
        pendingImages = []; renderThumbs();
        const userView = addUser(text, images);
        // Ollama takes images as bare base64 alongside the text, not as a data URL.
        const userMsg = { role: "user", content: text };
        if (images.length) {
          userMsg.images = images.map(img => img.b64);
          // Parallel to images[], so the server can pair them up. Only sent
          // when at least one photo actually carried any.
          const meta = images.map(img => img.meta || null);
          // Checked here as well as at attach time: turning the toggle off
          // after attaching, or picking a routine that turns it off, has to
          // mean the details do not go — not just that new photos skip them.
          if (exifOn && meta.some(Boolean)) userMsg.image_meta = meta;
        }
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
        // Nothing was produced, so nothing is recorded — and you get your
        // message back to edit and resend. Leaving the bubble on screen without
        // it reaching the store meant the phone and the desktop disagreed about
        // the conversation, and the next send posted two user turns in a row.
        // Returns whether the message was actually handed back, so the caller
        // does not claim it was when it wasn't.
        function rollBackTurn() {
          // Abandoned on purpose: newChat() and loadConversation() abort the
          // request, and pushing the discarded message and its photos into the
          // composer of a *different* conversation is not a rescue, it is a
          // surprise. The view and messages[] are already guarded this way.
          if (messages !== thread) return false;

          if (messages.length && messages[messages.length - 1] === userMsg) messages.pop();
          if (userView) userView.remove();
          // Attachments come back either way. A photo of an error on a screen
          // that has since changed cannot be retaken, and dropping it silently
          // because the composer happened to be occupied was the worst outcome
          // of the three.
          if (images.length && !pendingImages.length) { pendingImages = images.slice(); renderThumbs(); }
          // The text only if the composer is still empty — the user may have
          // started typing the next message while this one was in flight, and
          // overwriting that would be its own small disaster.
          if (inputEl.value.trim()) return false;
          inputEl.value = text;
          autosize();
          return true;
        }

        let turnSaved = false;
        async function commitTurn(reply, sources) {
          if (turnSaved || messages !== thread) return;
          turnSaved = true;
          await ensureConversation(text);
          await saveMessage("user", text, userMsg.images || null, null,
                            userMsg.image_meta || null);
          if (reply) await saveMessage("assistant", reply, null, sources || null);
          refreshConversations();
        }
        controller = new AbortController();
        let rawContent = "", thinkingField = "", started = false, usage = null, lastSources = null;

        // Between Send and the first token there was no feedback at all. A cold
        // 30b over Tailscale is half a minute of a motionless "…" against a
        // five-minute timeout, which is what makes people give up and reload.
        // A grounding status line takes the slot over permanently when it
        // arrives, since it says something more useful than a stopwatch.
        // Keep the screen on for the turn. A phone that locks mid-generation
        // suspends the tab, and the answer the server already produced is lost
        // when it resumes. Re-acquired on resume, since the lock is dropped
        // whenever the page is hidden.
        await acquireWakeLock();

        const waitingSince = Date.now();
        let waitTimer = setInterval(() => {
          if (started || view.statusOwned) return;
          const secs = Math.round((Date.now() - waitingSince) / 1000);
          if (secs < 3) return;   // don't flicker on a warm model
          view.status.textContent = "waiting for " + (modelEl.value || "the model") +
                                    "… " + secs + "s";
          view.status.hidden = false;
        }, 1000);
        function stopWaitTimer() {
          if (waitTimer) { clearInterval(waitTimer); waitTimer = null; }
        }

        try {
          const resp = await fetch("api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model: modelEl.value || undefined,
              messages: withRecentImages(messages),
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
              // Web-grounding progress, emitted before the model starts. While
              // one is on screen it says more than a stopwatch would, so the
              // counter stays out of the way — but an empty status *clears*
              // the line, which is what the server sends when the planner
              // declines, and the wait that follows is the longest one there
              // is. Ownership tracks the line's current state rather than
              // latching, so "searching…" then "" hands the line back.
              if (obj.status !== undefined) {
                view.statusOwned = !!obj.status;
                view.status.textContent = obj.status;
                view.status.hidden = !obj.status;
                scrollDown();
                continue;
              }
              if (obj.sources) { lastSources = obj.sources; showSources(view, obj.sources); scrollDown(); continue; }
              // /api/chat nests thinking under "message", the same as content.
              // Reading only the top-level field — the /api/generate shape —
              // meant that on any Ollama new enough to stream reasoning
              // natively, the panel never opened and the bubble sat on its "…"
              // placeholder for the whole scratchpad, looking hung.
              const think = (obj.message && obj.message.thinking) || obj.thinking || "";
              if (think) thinkingField += think;
              const piece = (obj.message && obj.message.content) || obj.content || "";
              if (piece) rawContent += piece;
              if (obj.done) usage = obj;

              const { content, thinking } = splitThink(rawContent);
              const allThink = thinkingField + thinking;
              if (piece || think) {
                started = true;
                stopWaitTimer();
                if (!view.statusOwned) view.status.hidden = true;
                view.bubble.textContent = content || "…";
              }
              if (allThink) { view.think.hidden = false; view.thinkBody.textContent = allThink; }
              scrollDown();
            }
          }
          const finalContent = splitThink(rawContent).content;
          view.status.hidden = true;
          if (finalContent) {
            paintMarkdown(view.bubble, finalContent);
            view.raw = finalContent;
            if (view.copyBtn) view.copyBtn.hidden = false;
            if (messages === thread) messages.push({ role: "assistant", content: finalContent });
            commitTurn(finalContent, lastSources);
            if (usage) view.meta.textContent = fmtUsage(usage);
          } else {
            // The request completed but the model said nothing. Recording the
            // question without an answer left the store disagreeing with the
            // screen and made the next send post two user turns; roll the whole
            // turn back and hand the message back so it can be retried.
            view.root.remove();
            went = false;
            const returned = rollBackTurn();
            markError("The model returned an empty reply." +
                      (returned ? " Your message is back in the box." : ""));
          }
          setStatus("ok", "connected");
        } catch (err) {
          if (err.name === "AbortError") {
            const partial = splitThink(rawContent).content;
            if (partial) {
              view.bubble.textContent = partial + "  ⏹ stopped";
              if (messages === thread) messages.push({ role: "assistant", content: partial });
              commitTurn(partial, lastSources);   // Stop still produced an answer
            } else {
              // Stopped before any visible text — which for a reasoning model is
              // the whole scratchpad, i.e. exactly when Stop gets pressed. The
              // turn used to stay on screen and in messages[] while never being
              // written, so the two devices diverged and the next send posted
              // two user roles.
              view.root.remove();
              went = false;
              // Only claim the message came back if it did: with the composer
              // already holding something else it is not restored, and saying
              // otherwise sent people looking for a photo that had been dropped.
              if (rollBackTurn()) {
                hintEl.textContent = "Stopped before the model replied — your message is back in the box.";
              }
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
              went = false;
              rollBackTurn();
              markError(err.message || String(err));
            }
            setStatus("bad", "error");
          }
        } finally {
          stopWaitTimer();
          releaseWakeLock();
          busy = false; sendBtn.disabled = false; stopBtn.hidden = true;
          controller = null;
          if (pendingAutoSend) {
            pendingAutoSend = false;
            // trySend, not send: a dictated follow-up while a routine is
            // armed still owes it its photos.
            //
            // Not `return` — a return inside finally replaces the value the
            // function was about to give back, so this reported every queued
            // dictation as a turn that never went and the routine stayed armed
            // with its own message already sent.
            if (inputEl.value.trim()) queued = true;
          }
          if (queued) setTimeout(trySend, 0);
          else inputEl.focus();
        }
        return went;
      }

      function stop() { if (controller) controller.abort(); }

      function newChat() {
        if (controller) controller.abort();
        currentConvoId = null;
        messages = [];
        pendingImages = []; renderThumbs();
        clearRoutine();
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
        showHistoryCost();
      }

      // What history is costing on disk. Attached images are stored with their
      // message, so an image-heavy history grows quickly and there was no way
      // to see that short of looking at the file.
      async function showHistoryCost() {
        try {
          const s = await (await fetch("api/conversations/stats")).json();
          if (!s || typeof s.bytes !== "number") return;
          const mb = s.bytes / (1024 * 1024);
          const size = mb >= 1 ? mb.toFixed(1) + " MB" : Math.round(s.bytes / 1024) + " KB";
          const note = document.createElement("p");
          note.className = "convo-empty";
          note.textContent = s.messages + " messages · " + size + " on disk";
          convoListEl.appendChild(note);
        } catch (e) { /* a footnote is never worth an error */ }
      }

      // Create on first use rather than on page load, so idly opening the app
      // does not litter the list with empty conversations.
      // History failing is not worth blocking the chat over — but it is worth
      // saying. Neither of these checked resp.ok, so a 503 from a full or
      // corrupt database resolved normally, the catch never ran, and the
      // conversation list quietly stopped growing while chatting looked fine.
      let historyBroken = false;
      function noteHistoryBroken(detail) {
        if (historyBroken) return;
        historyBroken = true;
        hintEl.textContent = detail ||
          "Conversations are no longer being saved — the history database could not be written.";
      }

      async function ensureConversation(firstMessage) {
        if (!historyOn || currentConvoId) return currentConvoId;
        try {
          const resp = await fetch("api/conversations", {
            method: "POST", headers: { "Content-Type": "application/json" },
            // A routine names its thread. Titling from the message would
            // give every trip the same 60-character prefix of the same
            // prompt, and the drawer becomes a wall of identical rows.
            body: JSON.stringify({
              title: (pendingRoutine && pendingRoutine.routine.name) || firstMessage || "",
              model: modelEl.value || null }) });
          const data = await resp.json();
          if (!resp.ok || !data.id) { noteHistoryBroken(data.error); currentConvoId = null; }
          else { currentConvoId = data.id; historyBroken = false; }
        } catch (e) { noteHistoryBroken(); currentConvoId = null; }
        return currentConvoId;
      }

      async function saveMessage(role, content, images, sources, meta) {
        if (!historyOn || !currentConvoId) return;
        try {
          const resp = await fetch("api/conversations/" + currentConvoId + "/messages", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ role: role, content: content,
                                   images: images || null, sources: sources || null,
                                   image_meta: meta || null }) });
          if (!resp.ok) {
            const data = await resp.json().catch(function () { return {}; });
            noteHistoryBroken(data.error);
          } else { historyBroken = false; }
        } catch (e) { noteHistoryBroken(); }
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
        clearRoutine();

        for (const msg of convo.messages || []) {
          const imgs = (msg.images || []).map(function (b64) {
            return { b64: b64, url: "data:image/jpeg;base64," + b64 };
          });
          if (msg.role === "user") {
            addUser(msg.content, imgs);
            const entry = { role: "user", content: msg.content };
            if (imgs.length) entry.images = imgs.map(function (i) { return i.b64; });
            if (imgs.length && msg.image_meta) entry.image_meta = msg.image_meta;
            messages.push(entry);
          } else {
            const view = addAssistant();
            if (msg.content) {
              paintMarkdown(view.bubble, msg.content);
              view.raw = msg.content;      // a reopened thread copies too
              if (view.copyBtn) view.copyBtn.hidden = false;
            }
            if (msg.sources) showSources(view, msg.sources);
            messages.push({ role: "assistant", content: msg.content });
          }
        }
        if (convo.model && modelEl.querySelector('option[value="' + convo.model + '"]')) {
          modelEl.value = convo.model;
        }
        closeDrawer();
        refreshConversations();
        scrollDown(true);
      }

      function openDrawer(pane) {
        drawerEl.hidden = false; backdropEl.hidden = false;
        showPane(pane === "routines" ? "routines" : "chats");
      }
      function closeDrawer() { drawerEl.hidden = true; backdropEl.hidden = true; }

      // ---- Routines ----
      // A saved prompt you tap instead of typing. Picking one drops its text
      // into the composer, ticks the toggles it declares, and holds Send back
      // until its photos are attached — which is the whole point: two odometer
      // photos have to ride on ONE message, because only the newest
      // image-bearing turn keeps its payload (withRecentImages here,
      // keep_recent_images on the server), and their timestamps have to be read
      // before the canvas re-encode strips them.

      async function refreshRoutines() {
        if (!historyOn) { routineBar.hidden = true; return; }
        try {
          const resp = await fetch("api/routines");
          if (!resp.ok) return;
          const data = await resp.json();
          routines = Array.isArray(data.routines) ? data.routines : [];
        } catch (e) { return; }
        // Deleted on another device while it was armed here: drop it rather
        // than leaving a lit chip that refers to nothing.
        if (pendingRoutine &&
            !routines.some(r => r.id === pendingRoutine.routine.id)) clearRoutine();
        routineBar.hidden = false;
        renderRoutineChips();
      }

      function renderRoutineChips() {
        routineChipsEl.innerHTML = "";
        if (!routines.length) {
          // The discovery path on a fresh install: one chip that opens the
          // drawer, where the starters are one more tap away.
          const add = document.createElement("button");
          add.type = "button"; add.className = "chip";
          add.textContent = "＋ Add a routine";
          add.addEventListener("click", () => openDrawer("routines"));
          routineChipsEl.appendChild(add);
          return;
        }
        const armed = pendingRoutine && pendingRoutine.routine.id;
        for (const saved of routines) {
          // The armed chip is drawn from the object the guard is actually
          // using, not from the refreshed list. Editing a routine while it is
          // armed otherwise had the chip reading "1/3" off the new record
          // while Send let it through on the old count of 1.
          const routine = saved.id === armed ? pendingRoutine.routine : saved;
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "chip" + (routine.id === armed ? " active" : "");
          // textContent, never innerHTML: a routine name is stored text that
          // anyone who can reach this app could have written.
          let label = routine.name;
          if (routine.id === armed && routine.photos) {
            label += " · " + Math.min(pendingImages.length, routine.photos) +
                     "/" + routine.photos;
          }
          chip.textContent = label;
          chip.addEventListener("click", () => pickRoutine(routine));
          routineChipsEl.appendChild(chip);
        }
      }

      function forceToggle(el, value) {
        // Assigned, never dispatched: rememberToggle listens for "change", and
        // a routine forcing a toggle for one turn must not overwrite what this
        // browser chose. Same rule as switchToVisionModel setting modelEl.value.
        el.checked = value;
      }

      function pickRoutine(routine) {
        // Tapping the lit chip puts everything back — the undo for a mis-tap.
        if (pendingRoutine && pendingRoutine.routine.id === routine.id) {
          clearRoutine();
          return;
        }
        clearRoutine();
        pendingRoutine = { routine: routine, web: null, exif: null };
        if (routine.web !== null && webEl.checked !== routine.web) {
          pendingRoutine.web = webEl.checked;
          forceToggle(webEl, routine.web);
        }
        if (routine.photo_meta !== null && exifEl.checked !== routine.photo_meta) {
          pendingRoutine.exif = exifEl.checked;
          forceToggle(exifEl, routine.photo_meta);
          exifOn = routine.photo_meta;
        }
        // Into the box, not straight onto the wire: you read it, you can edit
        // it ("this was the rental, it reads km"), and what the model gets is
        // exactly what the bubble shows and what the store keeps. A routine
        // body is saved text that anyone who can reach this app can change, so
        // the moment it is visible before it is sent is the mitigation that
        // matters most.
        const typed = inputEl.value.trim();
        pendingRoutine.inserted = routine.body;
        inputEl.value = typed ? routine.body + "\\n\\n" + typed : routine.body;
        autosize();
        renderRoutineChips();
        routineProgress();
        // Attaching first and picking afterwards is too late for the photo's
        // own timestamp: toAttachment() reads it before the canvas re-encode
        // and there is no second chance. Say so while re-attaching still fixes it.
        if (pendingImages.length && routine.photo_meta === true &&
            !pendingImages.some(img => img.meta && img.meta.taken)) {
          hintEl.textContent = "Those photos were read without their date — remove them and attach again.";
        }
        inputEl.focus();
      }

      function clearRoutine() {
        if (!pendingRoutine) return;
        if (pendingRoutine.web !== null) forceToggle(webEl, pendingRoutine.web);
        if (pendingRoutine.exif !== null) {
          forceToggle(exifEl, pendingRoutine.exif);
          exifOn = pendingRoutine.exif;
        }
        // Take the body back out of the composer, but only while it is still
        // there verbatim — the whole point of putting it in a box was that you
        // can edit it, and edited text is yours. Without this, picking a second
        // routine stacked the two prompts on top of each other.
        const inserted = pendingRoutine.inserted;
        if (inserted && inputEl.value.indexOf(inserted) === 0) {
          let rest = inputEl.value.slice(inserted.length);
          if (rest.slice(0, 2) === "\\n\\n") rest = rest.slice(2);
          inputEl.value = rest;
          autosize();
        }
        pendingRoutine = null;
        renderRoutineChips();
      }

      function routineGap(routine, have) {
        // "At least": three photos for a two-photo routine is your business,
        // one is the silently-wrong-answer case this exists to prevent.
        if (!routine || !routine.photos) return 0;
        return Math.max(0, routine.photos - have);
      }

      function routineProgress() {
        if (!pendingRoutine) return;
        renderRoutineChips();
        const routine = pendingRoutine.routine;
        const gap = routineGap(routine, pendingImages.length);
        if (gap) {
          hintEl.textContent = routine.name + " needs " + routine.photos +
            (routine.photos === 1 ? " photo — " : " photos — ") +
            pendingImages.length + " attached.";
        } else if (routine.photos && hintEl.textContent.indexOf(" attached.") > 0) {
          // Only our own message. hintEl is shared, and blanking it outright
          // wiped the latched "history is not being saved" warning the moment
          // the second photo landed.
          hintEl.textContent = "";
        }
      }

      async function trySend() {
        const routine = pendingRoutine && pendingRoutine.routine;
        if (routineGap(routine, pendingImages.length)) {
          routineProgress();
          return;
        }
        // Only when the turn actually went. send() refuses an empty message
        // and one that arrives while a stream is still running; disarming
        // there put the toggles back and dropped the count while the message
        // was still sitting in the composer waiting to be sent.
        if (await send()) clearRoutine();
      }

      // ---- Routines: the drawer ----

      function showPane(which) {
        const wantRoutines = which === "routines";
        convoPaneEl.hidden = wantRoutines;
        routinePaneEl.hidden = !wantRoutines;
        tabChatsEl.classList.toggle("active", !wantRoutines);
        tabRoutinesEl.classList.toggle("active", wantRoutines);
        if (wantRoutines) { closeRoutineEditor(); renderRoutineList(); }
        else refreshConversations();
      }

      function renderRoutineList() {
        routineListEl.innerHTML = "";
        if (!routines.length) {
          const note = document.createElement("p");
          note.className = "convo-empty";
          note.textContent = "A routine is a prompt you save once and tap instead of typing.";
          routineListEl.appendChild(note);
          const add = document.createElement("button");
          add.type = "button"; add.className = "drawer-new";
          add.textContent = "＋ Add the starter routines";
          add.addEventListener("click", addStarters);
          routineListEl.appendChild(add);
          return;
        }
        for (const routine of routines) {
          const row = document.createElement("div");
          row.className = "convo";
          const open = document.createElement("button");
          open.type = "button"; open.className = "convo-open";
          const title = document.createElement("span");
          title.className = "convo-title";
          title.textContent = routine.name;
          const meta = document.createElement("span");
          meta.className = "convo-meta";
          const bits = [];
          if (routine.photos) bits.push(routine.photos + (routine.photos === 1 ? " photo" : " photos"));
          if (routine.photo_meta !== null) bits.push("📍 " + (routine.photo_meta ? "on" : "off"));
          if (routine.web !== null) bits.push("🌐 " + (routine.web ? "on" : "off"));
          meta.textContent = bits.join(" · ") || "no photos";
          open.appendChild(title); open.appendChild(meta);
          open.addEventListener("click", () => openRoutineEditor(routine));
          row.appendChild(open);
          routineListEl.appendChild(row);
        }
      }

      function openRoutineEditor(routine) {
        editingRoutineId = routine ? routine.id : "";
        rNameEl.value = routine ? routine.name : "";
        rBodyEl.value = routine ? routine.body : "";
        rPhotosEl.value = String(routine ? routine.photos : 0);
        rMetaEl.value = routine && routine.photo_meta !== null ? (routine.photo_meta ? "1" : "0") : "";
        rWebEl.value = routine && routine.web !== null ? (routine.web ? "1" : "0") : "";
        rDeleteBtn.hidden = !routine;
        routineListEl.hidden = true;
        routineNewBtn.hidden = true;
        routineEditEl.hidden = false;
        routineWarnUpdate();
        rNameEl.focus();
      }

      function closeRoutineEditor() {
        editingRoutineId = null;
        routineEditEl.hidden = true;
        routineListEl.hidden = false;
        routineNewBtn.hidden = false;
      }

      function routineWarnUpdate() {
        // Said where the choice is made rather than discovered later: with
        // photos attached and web access forced on, the photo's own position
        // goes to the search planner, whose queries leave the machine.
        const risky = Number(rPhotosEl.value) > 0 && rWebEl.value === "1";
        routineWarnEl.textContent = risky
          ? "With photos attached and web access on, a photo's coordinates are sent to the search planner, whose queries go to a search engine. WEB_SHARE_LOCATION=0 stops that."
          : "";
      }

      let savingRoutine = false;
      async function saveRoutine() {
        // editingRoutineId is not set until after the await, so two taps on a
        // slow link both took the create branch and made two of the routine.
        if (savingRoutine) return;
        const name = rNameEl.value.trim();
        const body = rBodyEl.value.trim();
        if (!name || !body) {
          routineWarnEl.textContent = "A routine needs a name and a prompt.";
          return;
        }
        const tri = (value) => (value === "" ? null : value === "1");
        const payload = { name: name, body: body,
                          photos: Number(rPhotosEl.value) || 0,
                          web: tri(rWebEl.value), photo_meta: tri(rMetaEl.value) };
        savingRoutine = true;
        rSaveBtn.disabled = true;
        try {
          const resp = await fetch(
            editingRoutineId ? "api/routines/" + editingRoutineId : "api/routines",
            { method: editingRoutineId ? "PATCH" : "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload) });
          if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            routineWarnEl.textContent = data.error || "Could not save that.";
            return;
          }
          // What came back is what will be used. The store truncates, and a
          // routine that quietly does something other than what is written in
          // the box is worse than one that refuses to save.
          const saved = await resp.json().catch(() => ({}));
          const kept = saved.routine || saved;
          if (kept && typeof kept.body === "string" && kept.body.length < body.length) {
            rBodyEl.value = kept.body;
            routineWarnEl.textContent = "Saved, but the prompt was trimmed to " +
              kept.body.length + " characters.";
            return;
          }
        } catch (e) {
          routineWarnEl.textContent = "Could not reach the server.";
          return;
        } finally {
          savingRoutine = false;
          rSaveBtn.disabled = false;
        }
        closeRoutineEditor();
        await refreshRoutines();
        renderRoutineList();
      }

      async function deleteRoutine() {
        if (!editingRoutineId) return;
        if (!window.confirm("Delete this routine?")) return;
        const id = editingRoutineId;
        // Armed and then deleted from under itself: put the toggles back before
        // the chip it belongs to stops existing.
        try {
          const resp = await fetch("api/routines/" + id, { method: "DELETE" });
          if (!resp.ok) {
            routineWarnEl.textContent = "Could not delete that.";
            return;
          }
        } catch (e) {
          routineWarnEl.textContent = "Could not reach the server.";
          return;
        }
        // Only once it is actually gone. Disarming first left a failed delete
        // with the routine alive, its toggles reverted and the editor stuck.
        if (pendingRoutine && pendingRoutine.routine.id === id) clearRoutine();
        closeRoutineEditor();
        await refreshRoutines();
        renderRoutineList();
      }

      async function addStarters() {
        try {
          await fetch("api/routines/starters", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: "{}" });
        } catch (e) { return; }
        await refreshRoutines();
        renderRoutineList();
      }

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
          // trySend, not send: dictation is still a send, and a routine armed
          // for this turn is still owed its photos.
          if (autoSendEl.checked && !busy) { trySend(); return; }
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

      // Remembered per device, because the right answer differs per device:
      // the phone and the desktop want different models. Storage can throw in a
      // locked-down browser, so every access is guarded.
      function remembered(key) {
        try { return localStorage.getItem("chat." + key) || ""; } catch (e) { return ""; }
      }
      function remember(key, value) {
        try { localStorage.setItem("chat." + key, value); } catch (e) {}
      }
      // Only an explicit choice is stored. switchToVisionModel() and
      // loadConversation() set modelEl.value programmatically, which does not
      // fire "change" — so an automatic switch never overwrites your pick.
      modelEl.addEventListener("change", () => remember("model", modelEl.value));

      // Remember the voice toggles — they describe your hardware and habits, not
      // this visit. Headphones defaults ON (see the checkbox in the markup);
      // only an explicit stored choice overrides it.
      // Whether this browser has ever touched a toggle, so a server-supplied
      // default can apply to a fresh browser without overriding a real choice.
      function chosen(key) {
        try { return localStorage.getItem(key) !== null; } catch (e) { return false; }
      }

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
      // Starts off and is turned on by /api/health if the server says so —
      // rather than the other way round, so a deployment that wants this off
      // never has a window in which a photo's position is read anyway.
      rememberToggle(exifEl, "chatExif", false);
      exifOn = exifEl.checked;
      exifEl.addEventListener("change", () => { exifOn = exifEl.checked; });
      // Reaching for a toggle while a routine is armed is a real choice, so
      // drop it from the restore rather than putting it back afterwards.
      exifEl.addEventListener("change", () => {
        if (pendingRoutine) pendingRoutine.exif = null;
      });
      webEl.addEventListener("change", () => {
        if (pendingRoutine) pendingRoutine.web = null;
      });

      // An arrow function, not the handler by reference: openDrawer now
      // takes a pane name, and passing it the MouseEvent would read as one.
      menuBtn.addEventListener("click", () => openDrawer("chats"));
      routineEditBtn.addEventListener("click", () => openDrawer("routines"));
      tabChatsEl.addEventListener("click", () => showPane("chats"));
      tabRoutinesEl.addEventListener("click", () => showPane("routines"));
      routineNewBtn.addEventListener("click", () => openRoutineEditor(null));
      rSaveBtn.addEventListener("click", saveRoutine);
      document.getElementById("rCancel").addEventListener("click", closeRoutineEditor);
      rDeleteBtn.addEventListener("click", deleteRoutine);
      rPhotosEl.addEventListener("change", routineWarnUpdate);
      rWebEl.addEventListener("change", routineWarnUpdate);
      document.getElementById("drawerClose").addEventListener("click", closeDrawer);
      backdropEl.addEventListener("click", closeDrawer);
      document.getElementById("drawerNew").addEventListener("click", function () {
        newChat(); closeDrawer();
      });

      sendBtn.addEventListener("click", trySend);
      stopBtn.addEventListener("click", stop);
      newBtn.addEventListener("click", newChat);
      micBtn.addEventListener("click", toggleMic);
      inputEl.addEventListener("input", autosize);
      inputEl.addEventListener("keydown", (e) => {
        // isComposing / keyCode 229: Enter is accepting an IME candidate, not
        // submitting. Sending here would post the raw romaji.
        if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
          e.preventDefault(); trySend();
        }
      });

      // Keyboard advice only where there is a keyboard; on a phone it wrapped
      // to four lines and pushed the composer off the screen.
      if (window.matchMedia("(min-width: 641px)").matches) {
        inputEl.placeholder = "Type a message…  (Enter to send, Shift+Enter for a new line)";
      }

      // A tab resumed after a night in a pocket had checked the server exactly
      // once, when it was opened — so a dead dot stayed dead and a model pulled
      // since never appeared. Re-check on resume, but never mid-turn.
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState !== "visible" || busy) return;
        loadModels();
        checkVoice();
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
