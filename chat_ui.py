#!/usr/bin/env python3
"""Single-page chat UI for the Ollama Chat app.

Kept in its own module so ``app.py`` stays focused on routing. The page is a
self-contained HTML document (inline CSS + JS, no build step, no CDN) so it
works offline behind the server manager and on a phone. It talks to this app's
own ``/api/models``, ``/api/chat`` (streaming), ``/api/health`` and
``/api/transcribe`` endpoints.

``_PAGE`` is a **raw** string, and must stay one. Without the ``r`` Python
reads the escapes before the browser ever sees them, and the ones it
understands — ``\\b \\n \\t \\f \\v \\1`` — are consumed silently: ``\\b`` in a
JavaScript regex becomes a backspace character, ``\\1`` becomes a control
character, and the regex quietly stops matching. Three bugs of exactly that
shape shipped before this was made raw. ``test_ui_shell`` now scans the
rendered page for stray control characters and parses every script block, so
the mistake cannot be made silently again.
"""

from __future__ import annotations

import html

_PAGE = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>__TITLE__</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <!-- Inline, so there is no favicon request to 404 on every page load, and a
         home-screen shortcut on a phone gets an icon rather than a blank page. -->
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%232563eb'/%3E%3Cpath d='M8 11h16M8 16h16M8 21h10' stroke='white' stroke-width='2.6' stroke-linecap='round'/%3E%3C/svg%3E">
    <link rel="manifest" href="manifest.webmanifest">
    <meta name="color-scheme" content="light dark">
    <meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
    <meta name="theme-color" content="#0a0e17" media="(prefers-color-scheme: dark)">
    <!-- Before the stylesheet, on purpose: a saved theme applied from the main
         script at the end of <body> paints the other one first, and a white
         flash at 2am is the whole reason someone chose dark. -->
    <script>
      try {
        const saved = localStorage.getItem("theme");
        if (saved === "light" || saved === "dark")
          document.documentElement.setAttribute("data-theme", saved);
      } catch (e) {}
    </script>
    <style>
      /* ------------------------------------------------------------------
         Tokens. Two themes from one set of names: the light values are the
         defaults and the dark ones are swapped in under a media query, so a
         phone that follows the system at night needs no script to have run.
         data-theme on <html> overrides both, for the times the system is
         wrong about what you want.
         ------------------------------------------------------------------ */
      :root {
        color-scheme:light;
        --bg:#f4f6fb; --surface:#ffffff; --surface2:#eef1f8; --surface3:#e2e8f4;
        --border:#d8dfec; --border-soft:#e7ecf5;
        --text:#101828; --muted:#5b6b85; --faint:#8a99b3;
        --accent:#2563eb; --accent-hover:#1d4ed8; --accent-soft:#e5edff;
        --on-accent:#ffffff;
        --danger:#dc2626; --ok:#16a34a;
        --assistant:#ffffff; --assistant-border:#e2e8f4;
        --code-bg:#f6f8fc;
        --shadow1:0 1px 2px rgba(16,24,40,0.06);
        --shadow2:0 12px 32px rgba(16,24,40,0.14);
        --shadow-bubble:0 1px 1px rgba(16,24,40,0.04);
        /* Four steps, used everywhere. The page had eleven ad-hoc sizes. */
        --fs-xs:0.72rem; --fs-sm:0.8125rem; --fs-md:0.9375rem; --fs-lg:1.0625rem;
        --r1:0.5rem; --r2:0.75rem; --r3:1.1rem; --pill:999px;
      }
      :root[data-theme="dark"] { color-scheme:dark; }
      @media (prefers-color-scheme: dark) { :root { color-scheme:dark; } }
      /* One block, applied by either route. */
      :root[data-theme="dark"],
      :root:not([data-theme="light"]) {
        --dark-bg:#0a0e17; --dark-surface:#121a29; --dark-surface2:#1a2434;
        --dark-surface3:#233044; --dark-border:#26334a; --dark-border-soft:#1c2637;
        --dark-text:#e8edf7; --dark-muted:#9aa8c0; --dark-faint:#69788f;
        --dark-accent:#3b82f6; --dark-accent-hover:#60a5fa;
        --dark-accent-soft:rgba(59,130,246,0.16);
        --dark-danger:#f05252; --dark-ok:#22c55e;
        --dark-assistant:#16202f; --dark-assistant-border:#25324a;
        --dark-code-bg:#0c1320;
      }
      @media (prefers-color-scheme: dark) {
        :root:not([data-theme="light"]) {
          --bg:var(--dark-bg); --surface:var(--dark-surface);
          --surface2:var(--dark-surface2); --surface3:var(--dark-surface3);
          --border:var(--dark-border); --border-soft:var(--dark-border-soft);
          --text:var(--dark-text); --muted:var(--dark-muted); --faint:var(--dark-faint);
          --accent:var(--dark-accent); --accent-hover:var(--dark-accent-hover);
          --accent-soft:var(--dark-accent-soft);
          --danger:var(--dark-danger); --ok:var(--dark-ok);
          --assistant:var(--dark-assistant);
          --assistant-border:var(--dark-assistant-border);
          --code-bg:var(--dark-code-bg);
          --shadow1:0 1px 2px rgba(0,0,0,0.4);
          --shadow2:0 16px 40px rgba(0,0,0,0.5);
          --shadow-bubble:none;
        }
      }
      :root[data-theme="dark"] {
        --bg:var(--dark-bg); --surface:var(--dark-surface);
        --surface2:var(--dark-surface2); --surface3:var(--dark-surface3);
        --border:var(--dark-border); --border-soft:var(--dark-border-soft);
        --text:var(--dark-text); --muted:var(--dark-muted); --faint:var(--dark-faint);
        --accent:var(--dark-accent); --accent-hover:var(--dark-accent-hover);
        --accent-soft:var(--dark-accent-soft);
        --danger:var(--dark-danger); --ok:var(--dark-ok);
        --assistant:var(--dark-assistant);
        --assistant-border:var(--dark-assistant-border);
        --code-bg:var(--dark-code-bg);
        --shadow1:0 1px 2px rgba(0,0,0,0.4);
        --shadow2:0 16px 40px rgba(0,0,0,0.5);
        --shadow-bubble:none;
      }

      * { box-sizing:border-box; }
      /* Smooth scrolling and sliding panels are decoration; someone who has
         asked their system for less motion has usually asked for a reason. */
      @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
          animation-duration:0.01ms !important; animation-iteration-count:1 !important;
          transition-duration:0.01ms !important; scroll-behavior:auto !important;
        }
      }
      /* Author display rules outrank the UA [hidden] rule, so every
         flex element here would ignore .hidden without this. */
      [hidden] { display:none !important; }
      html, body { height:100%; margin:0; }
      body {
        background:var(--bg); color:var(--text);
        font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        font-size:var(--fs-md);
        /* A row, not a column: the drawer is a real column of the layout on a
           desktop and an overlay on a phone, and this is what lets it be both
           without the script knowing which. */
        display:flex; flex-direction:row; height:100dvh;
        -webkit-tap-highlight-color:transparent;
      }
      .pane { flex:1 1 auto; min-width:0; display:flex; flex-direction:column;
              height:100dvh; }
      /* One ring everywhere, in the accent colour. The browser default is
         orange, which fights a blue app on every focused control. */
      :focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
      .sprite { display:none; }
      .i { width:1.15em; height:1.15em; flex:0 0 auto; fill:none;
           stroke:currentColor; stroke-width:1.85; stroke-linecap:round;
           stroke-linejoin:round; }
      .i.solid { fill:currentColor; stroke:none; }

      /* ---- Header ------------------------------------------------------- */
      header {
        display:flex; align-items:center; gap:0.6rem; flex-wrap:nowrap;
        padding:0.55rem 0.9rem; background:var(--surface);
        border-bottom:1px solid var(--border-soft);
      }
      /* The conversation you are in, not the name of the app — you always know
         which app you opened, and after twenty saved threads you often do not
         know which one is on screen. */
      #chatTitle {
        font-size:var(--fs-md); margin:0; font-weight:600; white-space:nowrap;
        overflow:hidden; text-overflow:ellipsis; min-width:0;
      }
      .head-title { display:flex; flex-direction:column; min-width:0; }
      .status { display:flex; align-items:center; gap:0.35rem;
                font-size:var(--fs-xs); color:var(--muted); }
      .dot { width:0.5rem; height:0.5rem; border-radius:50%; background:var(--faint);
             box-shadow:0 0 0 3px transparent; transition:box-shadow 0.2s; }
      .dot.ok { background:var(--ok); box-shadow:0 0 0 3px rgba(34,197,94,0.15); }
      .dot.bad { background:var(--danger); box-shadow:0 0 0 3px rgba(240,82,82,0.15); }
      .spacer { flex:1 1 auto; }

      /* ---- Controls ----------------------------------------------------- */
      select, button, input, textarea {
        font-family:inherit; font-size:var(--fs-sm);
        background:var(--surface2); color:var(--text);
        border:1px solid var(--border); border-radius:var(--r1);
        padding:0.42rem 0.6rem;
      }
      select { max-width:16rem; }
      button { cursor:pointer; display:inline-flex; align-items:center;
               justify-content:center; gap:0.35rem; line-height:1.2;
               transition:background 0.12s, border-color 0.12s, color 0.12s; }
      button:hover:not(:disabled) { background:var(--surface3); }
      button.primary { background:var(--accent); border-color:var(--accent);
                       color:var(--on-accent); font-weight:600; }
      button.primary:hover:not(:disabled) { background:var(--accent-hover);
                                            border-color:var(--accent-hover); }
      button.danger { background:var(--danger); border-color:var(--danger);
                      color:#fff; font-weight:600; }
      button.danger:hover:not(:disabled) { filter:brightness(1.08);
                                           background:var(--danger); }
      button:disabled { opacity:0.5; cursor:default; }
      /* Square, icon-only, quiet until you point at it. */
      .iconbtn { width:2.15rem; height:2.15rem; padding:0; color:var(--muted);
                 background:transparent; border-color:transparent; }
      .iconbtn:hover:not(:disabled) { background:var(--surface2); color:var(--text); }
      .iconbtn .i { width:1.15rem; height:1.15rem; }

      /* ---- Conversation -------------------------------------------------- */
      /* No scroll-behavior:smooth here. Following a stream assigns scrollTop
         on every token, and an animated assignment fires scroll events part of
         the way there — which atBottom() reads as the user having scrolled
         away, so the follow switches itself off mid-reply. */
      #chat {
        flex:1 1 auto; overflow-y:auto; padding:1.5rem 1rem 0.5rem;
        display:flex; flex-direction:column; gap:0.9rem;
      }
      .wrap { width:100%; max-width:48rem; margin:0 auto; }
      .msg { display:flex; }
      .msg.user { justify-content:flex-end; }
      .col { display:flex; flex-direction:column; max-width:min(85%,40rem); }
      .msg.user .col { align-items:flex-end; }
      .bubble { padding:0.7rem 0.9rem; border-radius:var(--r3);
        white-space:pre-wrap; word-wrap:break-word; line-height:1.55;
        font-size:var(--fs-md); box-shadow:var(--shadow-bubble); }
      .msg.user .bubble { background:var(--accent); color:var(--on-accent);
        border-bottom-right-radius:0.35rem; }
      .msg.assistant .bubble { background:var(--assistant);
        border:1px solid var(--assistant-border); border-bottom-left-radius:0.35rem; }
      .msg.error .bubble { background:color-mix(in srgb, var(--danger) 12%, var(--surface));
        border:1px solid var(--danger); color:var(--danger); }
      .role { font-size:var(--fs-xs); text-transform:uppercase; letter-spacing:0.05em;
              color:var(--faint); margin:0 0.4rem 0.25rem; font-weight:600; }
      .meta { font-size:var(--fs-xs); color:var(--faint); margin:0.3rem 0.4rem 0; }
      /* Quiet until you look for it — this sits under every reply. */
      .replycopy { align-self:flex-start; font-size:var(--fs-xs);
        padding:0.2rem 0.5rem; margin:0.3rem 0.4rem 0; background:transparent;
        color:var(--faint); border:1px solid var(--border-soft);
        border-radius:var(--r1); opacity:0.75; }
      .replycopy:hover { opacity:1; color:var(--text); }
      .sources { font-size:var(--fs-xs); color:var(--muted); margin:0.35rem 0.4rem 0; }
      .sources a { color:var(--muted); text-decoration:underline; }
      .sources a:hover { color:var(--text); }
      .webstatus { font-size:var(--fs-xs); color:var(--muted); font-style:italic;
        margin:0 0.4rem 0.3rem; }
      details.think {
        margin:0 0 0.45rem; background:var(--surface2);
        border:1px solid var(--border-soft); border-radius:var(--r2);
        padding:0.15rem 0.6rem;
      }
      details.think summary { cursor:pointer; font-size:var(--fs-xs);
        color:var(--muted); padding:0.3rem 0; }
      .think-body { white-space:pre-wrap; font-size:var(--fs-sm); color:var(--muted);
        border-top:1px solid var(--border-soft); padding:0.45rem 0; margin-top:0.2rem; }
      /* The same shape as the thinking panel, because it is the same kind of
         thing: available, folded, and never in the way. */
      .steps-body { border-top:1px solid var(--border-soft); padding:0.45rem 0;
                    margin-top:0.2rem; font-size:var(--fs-xs); }
      .steprow { display:flex; flex-wrap:wrap; align-items:baseline; gap:0.4rem;
                 padding:0.2rem 0; color:var(--muted); }
      .steprow + .steprow { border-top:1px solid var(--border-soft); }
      .steprow b { color:var(--text); font-weight:600; flex:0 0 auto; }
      .steprow span { flex:1 1 12rem; min-width:0; }
      /* Closed it is a word at the end of the line; open it takes the row, so
         the block it reveals is bounded by the bubble rather than running off
         the side of it. */
      .stepmore { flex:0 0 auto; min-width:0; max-width:100%; }
      .stepmore[open] { flex:1 1 100%; }
      .stepmore summary { cursor:pointer; color:var(--accent); }
      .stepmore pre { white-space:pre-wrap; word-break:break-word; margin:0.35rem 0 0;
        padding:0.5rem 0.6rem; background:var(--code-bg); color:var(--muted);
        border:1px solid var(--border-soft); border-radius:var(--r1);
        max-height:18rem; overflow:auto; width:100%; box-sizing:border-box; }

      /* ---- The empty conversation ---------------------------------------
         A whole screen used to hold one grey sentence. It is the only moment
         the app can say what it is for, and the routines are the answer most
         of the time, so they are the thing on offer. */
      .empty { display:flex; flex-direction:column; align-items:center;
               text-align:center; gap:0.4rem; padding:2.5rem 0.5rem 1rem;
               color:var(--muted); }
      .empty-mark { width:3rem; height:3rem; border-radius:var(--r2);
        background:var(--accent); color:#fff; display:flex; align-items:center;
        justify-content:center; margin-bottom:0.5rem; box-shadow:var(--shadow1); }
      .empty-mark .i { width:1.6rem; height:1.6rem; }
      .empty h2 { margin:0; font-size:var(--fs-lg); color:var(--text);
                  font-weight:650; }
      .empty p { margin:0; font-size:var(--fs-sm); max-width:26rem; }
      .empty-routines { display:flex; flex-wrap:wrap; gap:0.5rem;
        justify-content:center; margin-top:1.2rem; }
      .startcard {
        display:flex; flex-direction:column; align-items:flex-start; gap:0.15rem;
        text-align:left; min-width:9.5rem; max-width:14rem;
        padding:0.7rem 0.85rem; border-radius:var(--r2);
        background:var(--surface); border:1px solid var(--border);
        box-shadow:var(--shadow1);
      }
      .startcard:hover { border-color:var(--accent); background:var(--surface); }
      .startcard b { font-size:var(--fs-sm); font-weight:600; color:var(--text); }
      .startcard span { font-size:var(--fs-xs); color:var(--muted);
        overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
        max-width:100%; }

      /* Sits over the conversation, clear of the composer, and only appears
         once you have scrolled away from the bottom. */
      .chatarea { position:relative; flex:1 1 auto; min-height:0; display:flex; }
      /* Filled rather than quiet: it sits over the reply it is offering to
         take you to, and a muted circle on a dark bubble was barely there. */
      .tolatest { position:absolute; bottom:0.6rem; left:50%;
        transform:translateX(-50%);
        z-index:5; width:2.3rem; height:2.3rem; padding:0; border-radius:50%;
        background:var(--accent); border:1px solid var(--accent);
        color:var(--on-accent); box-shadow:var(--shadow2); }
      .tolatest:hover { background:var(--accent-hover);
                        border-color:var(--accent-hover); }

      /* ---- Footer and composer ------------------------------------------- */
      footer { background:var(--surface); border-top:1px solid var(--border-soft);
               padding:0.55rem 1rem 0.7rem; }
      /* The card is the composer: the box you type in, the photos riding with
         it and the buttons that act on it are one object, not three stacked
         rows that happen to be adjacent. */
      .composer-card {
        background:var(--surface2); border:1px solid var(--border);
        border-radius:var(--r3); padding:0.5rem 0.55rem 0.45rem;
        transition:border-color 0.15s, box-shadow 0.15s;
      }
      .composer-card:focus-within { border-color:var(--accent);
        box-shadow:0 0 0 3px var(--accent-soft); }
      .composer { display:flex; flex-wrap:wrap; align-items:center; gap:0.35rem; }
      textarea {
        flex:1 1 100%; order:-1; min-width:0; resize:none;
        font-size:var(--fs-md); background:transparent; color:var(--text);
        border:0; border-radius:0; padding:0.35rem 0.4rem 0.45rem;
        max-height:40vh; min-height:2.2rem; line-height:1.5;
      }
      textarea:focus { outline:none; }
      textarea::placeholder { color:var(--faint); }
      .composer button { flex:0 0 auto; white-space:nowrap; }
      .composer .iconbtn { width:2.3rem; height:2.3rem; }
      /* Both get the rule: whichever is on screen sits at the right-hand end. */
      #send, #stop { margin-left:auto; padding:0.5rem 1.1rem;
                     border-radius:var(--r2); font-size:var(--fs-sm); }
      #mic.rec { background:var(--danger); border-color:var(--danger); color:#fff;
                 animation:pulse 1.2s infinite; }
      #mic .i-stop, #mic.rec .i-mic { display:none; }
      #mic.rec .i-stop { display:block; }
      @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.6;} }

      /* ---- The rows above the composer ------------------------------------ */
      .voicebar { display:flex; align-items:center; gap:0.4rem;
                  margin-bottom:0.45rem; flex-wrap:wrap; }
      .voicebar-label { font-size:var(--fs-xs); color:var(--faint); white-space:nowrap; }
      /* A pill that reads as on or off at a glance. The checkbox is still a
         real checkbox — it is the label that got the styling, because the
         label is what a thumb lands on. */
      .voicebar-check {
        display:flex; align-items:center; gap:0.35rem; cursor:pointer;
        font-size:var(--fs-xs); color:var(--muted); white-space:nowrap;
        padding:0.3rem 0.65rem; border-radius:var(--pill);
        background:var(--surface2); border:1px solid var(--border);
        user-select:none;
      }
      .voicebar-check:hover { border-color:var(--faint); }
      .voicebar-check input { accent-color:var(--accent); margin:0;
                              width:0.85rem; height:0.85rem; }
      .voicebar-check:has(input:checked) { background:var(--accent-soft);
        border-color:var(--accent); color:var(--text); font-weight:600; }
      .voicebar-check:has(input:focus-visible) { outline:2px solid var(--accent);
        outline-offset:2px; }
      .thumbs { display:flex; gap:0.4rem; flex-wrap:wrap; margin-bottom:0.45rem; }
      .thumb { position:relative; width:3.4rem; height:3.4rem; border-radius:var(--r1);
        overflow:hidden; border:1px solid var(--border); cursor:pointer; }
      .thumb img { width:100%; height:100%; object-fit:cover; display:block; }
      .thumb button {
        position:absolute; top:0; right:0; padding:0 0.3rem; font-size:0.7rem;
        line-height:1.4; border:0; border-radius:0 0 0 var(--r1);
        background:rgba(0,0,0,0.7); color:#fff;
      }
      .thumb button:hover { background:rgba(0,0,0,0.85); }
      .thumb .stamp { position:absolute; bottom:0; left:0; padding:0 0.25rem;
        font-size:0.6rem; line-height:1.4; background:rgba(0,0,0,0.7); color:#fff;
        border-radius:0 var(--r1) 0 0; }
      /* Whether a photo's own timestamp actually came through. The failure this
         guards is silent otherwise: with photo details off there is no EXIF, no
         metadata turn, and an answer to half the question with nothing on
         screen to say why. */
      .thumb .stamp.muted { opacity:0.6; }
      .hint { color:var(--faint); font-size:var(--fs-xs); margin:0.4rem 0.3rem 0;
              min-height:1rem; }
      .insecure { margin:0 0.2rem 0.5rem; }
      .insecure a { color:var(--accent); word-break:break-all; }
      #voiceModel, #micDevice { flex:0 1 auto; min-width:0; max-width:11rem;
                    font-size:var(--fs-xs); padding:0.35rem 0.4rem; }
      /* A bar that fills with the input level and a mark that sticks at the
         loudest thing heard, so both "nothing is arriving" and "this is
         clipping" are visible at a glance rather than inferred from a bad
         transcription. */
      .level { position:relative; flex:0 0 auto; width:4.5rem; height:0.45rem;
        border-radius:var(--pill); background:var(--surface3); overflow:hidden; }
      .level-bar { position:absolute; inset:0 auto 0 0; width:0;
        background:var(--ok); transition:width 0.06s linear; }
      .level-bar.hot { background:var(--danger); }
      .level-peak { position:absolute; top:0; bottom:0; width:2px;
        background:var(--text); opacity:0.7; left:0; }

      .bubble img { max-width:min(320px,100%); border-radius:var(--r1);
        margin-bottom:0.35rem; display:block; }
      /* Rendered markdown. white-space:normal because the renderer supplies the
         structure; pre-wrap would double every blank line. */
      .bubble.md { white-space:normal; }
      .bubble.md > *:first-child { margin-top:0; }
      .bubble.md > *:last-child { margin-bottom:0; }
      .bubble.md p { margin:0.55rem 0; }
      .bubble.md h3, .bubble.md h4, .bubble.md h5, .bubble.md h6 {
        margin:0.9rem 0 0.4rem; font-size:var(--fs-md); font-weight:650; }
      .bubble.md ul, .bubble.md ol { margin:0.55rem 0; padding-left:1.3rem; }
      .bubble.md li { margin:0.2rem 0; }
      .bubble.md blockquote { margin:0.55rem 0; padding:0.1rem 0 0.1rem 0.8rem;
        border-left:3px solid var(--border); color:var(--muted); }
      .bubble.md a { color:var(--accent); }
      .bubble.md code { background:var(--surface2); padding:0.1rem 0.32rem;
        border-radius:0.3rem; font-size:0.88em;
        font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
      .msg.user .bubble.md code { background:rgba(255,255,255,0.18); }
      .msg.user .bubble.md a { color:#fff; }
      .code { position:relative; margin:0.65rem 0; }
      .code pre { margin:0; padding:0.75rem 0.85rem; overflow-x:auto;
        background:var(--code-bg); border:1px solid var(--border);
        border-radius:var(--r2); }
      .code pre code { background:none; padding:0; font-size:0.84em; line-height:1.5; }
      .code pre[data-lang]::before { content:attr(data-lang); position:absolute;
        top:0.4rem; left:0.85rem; font-size:0.62rem; color:var(--faint);
        text-transform:uppercase; letter-spacing:0.07em; }
      .code pre[data-lang] { padding-top:1.5rem; }
      .code .copy { position:absolute; top:0.35rem; right:0.4rem; font-size:0.66rem;
        padding:0.18rem 0.45rem; background:var(--surface); color:var(--muted);
        border:1px solid var(--border); border-radius:var(--r1); opacity:0.8; }
      .code .copy:hover { opacity:1; color:var(--text); }

      /* A comparison table is wider than a phone more often than not, so it
         scrolls inside its own box. Letting it widen the bubble instead would
         put the whole conversation on a horizontal scrollbar. */
      .tablewrap { overflow-x:auto; margin:0.65rem 0; }
      .bubble.md table { border-collapse:collapse; font-size:var(--fs-sm);
                         min-width:100%; }
      .bubble.md th, .bubble.md td {
        border:1px solid var(--border); padding:0.35rem 0.6rem;
        text-align:left; vertical-align:top;
      }
      .bubble.md th { background:var(--surface2); font-weight:600; white-space:nowrap; }
      .bubble.md tbody tr:nth-child(even) td { background:var(--surface2); }
      .bubble.md hr { border:0; border-top:1px solid var(--border); margin:1rem 0; }

      /* ---- Drawer -------------------------------------------------------- */
      .backdrop { position:fixed; inset:0; background:rgba(3,7,18,0.55);
                  z-index:20; backdrop-filter:blur(2px); }
      .drawer {
        position:fixed; top:0; left:0; bottom:0; z-index:21; width:min(21rem,86vw);
        transition:width 0.14s ease;
        background:var(--surface); border-right:1px solid var(--border);
        display:flex; flex-direction:column; padding:0.7rem;
        box-shadow:var(--shadow2);
      }
      /* Wide enough for a sidebar and there is no reason to dim the app to
         show a list of its own conversations: the drawer becomes a column of
         the layout, and the ☰ button collapses it rather than dismissing it. */
      @media (min-width: 1024px) {
        .drawer { position:static; height:100dvh; box-shadow:none;
                  flex:0 0 auto; }
      }
      .drawer-head { display:flex; align-items:center; justify-content:space-between;
        gap:0.4rem; margin-bottom:0.6rem; }
      .brand { display:flex; align-items:center; gap:0.5rem; min-width:0;
               padding:0.15rem 0.2rem 0.5rem; }
      .brand .mark { width:1.6rem; height:1.6rem; border-radius:0.45rem;
        background:var(--accent); color:#fff; display:flex; align-items:center;
        justify-content:center; flex:0 0 auto; }
      .brand .mark .i { width:0.95rem; height:0.95rem; }
      .brand h1 { font-size:var(--fs-md); margin:0; font-weight:650;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      /* Capped: the tabs fill a 21rem drawer, but Records widens it to 56rem
         and three words spread across that read as a navigation bar for the
         whole app rather than a switch between three lists. */
      .drawer-tabs { display:flex; gap:0.2rem; background:var(--surface2);
        border:1px solid var(--border-soft); border-radius:var(--pill);
        padding:0.2rem; flex:1 1 auto; min-width:0; max-width:22rem; }
      .tab { font-size:var(--fs-xs); padding:0.35rem 0.7rem; color:var(--muted);
        background:transparent; border:0; border-radius:var(--pill);
        flex:1 1 0; min-width:0; }
      .tab:hover:not(.active) { background:var(--surface3); }
      .tab.active { color:var(--text); background:var(--surface);
        border:0; font-weight:600; box-shadow:var(--shadow1); }
      /* Never lit, because choosing it leaves the drawer rather than switching
         what the drawer is showing. The arrow is what says so. */
      .tab-link { letter-spacing:0.01em; }
      .drawer-new { width:100%; margin-bottom:0.6rem; font-weight:600;
        background:var(--accent); border-color:var(--accent);
        color:var(--on-accent); padding:0.55rem; border-radius:var(--r2); }
      .drawer-new:hover { background:var(--accent-hover); }
      /* The search box reads as one control: the magnifier and the clear
         button live inside the field's own border rather than beside it. */
      .searchbar { display:flex; align-items:center; gap:0.4rem; margin-bottom:0.5rem;
        padding:0 0.5rem; background:var(--surface2);
        border:1px solid var(--border); border-radius:var(--pill); }
      .searchbar:focus-within { border-color:var(--accent);
        box-shadow:0 0 0 3px var(--accent-soft); }
      .searchbar .i { color:var(--faint); width:1rem; height:1rem; }
      .searchbar input { flex:1 1 auto; min-width:0; background:transparent;
        border:0; padding:0.5rem 0; font-size:var(--fs-sm); }
      .searchbar input:focus { outline:none; }
      .searchbar input::-webkit-search-cancel-button { display:none; }
      .searchbar .iconbtn { width:1.6rem; height:1.6rem; }
      .searchhead { font-size:var(--fs-xs); color:var(--faint); font-weight:600;
        text-transform:uppercase; letter-spacing:0.05em;
        margin:0.7rem 0.3rem 0.2rem; }
      .searchhead:first-child { margin-top:0.1rem; }
      /* Two lines of context under a title, wrapped rather than truncated —
         the matched phrase is the reason the row is there. */
      .convo-meta.hit { white-space:normal; line-height:1.4;
        display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
        overflow:hidden; }
      .hitrecord { border:1px solid var(--border-soft); border-radius:var(--r1);
        margin-bottom:0.25rem; }
      .storage { margin:0.6rem 0.3rem 0.2rem; }
      .storage-act { align-self:flex-start; margin:0.1rem 0.3rem 0.4rem; }
      .expired { font-style:italic; }
      #convoList { overflow-y:auto; display:flex; flex-direction:column; gap:0.25rem; }
      .convo { display:flex; align-items:stretch; gap:0.15rem; border-radius:var(--r1); }
      .convo-open {
        flex:1 1 auto; min-width:0; text-align:left; display:flex;
        flex-direction:column; align-items:flex-start; gap:0.1rem;
        padding:0.45rem 0.55rem; background:transparent; border-color:transparent;
      }
      .convo-open:hover { background:var(--surface2); }
      .convo.active .convo-open { background:var(--accent-soft);
        border-color:var(--accent); }
      .convo-title { font-size:var(--fs-sm); overflow:hidden;
        text-overflow:ellipsis; white-space:nowrap; max-width:100%; }
      .convo-meta { font-size:var(--fs-xs); color:var(--faint); }
      .convo-act { flex:0 0 auto; padding:0 0.45rem; font-size:var(--fs-xs);
        color:var(--faint); background:transparent; border-color:transparent; }
      .convo-act:hover { color:var(--text); background:var(--surface2); }
      .convo-empty { color:var(--muted); font-size:var(--fs-sm); margin:0.5rem 0.2rem;
        line-height:1.5; }
      /* Thirty threads is a wall of titles. Days give it somewhere to land. */
      .daymark { font-size:var(--fs-xs); color:var(--faint); font-weight:600;
        text-transform:uppercase; letter-spacing:0.05em;
        margin:0.7rem 0.3rem 0.15rem; position:sticky; top:0;
        background:var(--surface); padding:0.15rem 0; }
      .daymark:first-child { margin-top:0; }

      /* One row that scrolls sideways rather than four rows that eat a phone
         screen — the strip has to stay a single line at ten routines. */
      .chips { display:flex; gap:0.35rem; flex:0 1 auto; min-width:0;
               overflow-x:auto; scrollbar-width:none; }
      .chips::-webkit-scrollbar { display:none; }
      .chip { flex:0 0 auto; white-space:nowrap; font-size:var(--fs-xs);
              padding:0.4rem 0.75rem; border-radius:var(--pill);
              background:var(--surface2); color:var(--muted); }
      .chip:hover:not(.active) { background:var(--surface3); color:var(--text); }
      .chip.active { background:var(--accent); border-color:var(--accent);
                     color:var(--on-accent); font-weight:600; }
      .chip-edit { flex:0 0 auto; padding:0.4rem 0.55rem; color:var(--muted); }
      #convoPane, #routinePane { display:flex; flex-direction:column;
                                 flex:1 1 auto; min-width:0; min-height:0; }
      #routineList { overflow-y:auto; display:flex; flex-direction:column; gap:0.25rem; }
      #routineEdit { display:flex; flex-direction:column; gap:0.3rem; overflow-y:auto; }
      #routineEdit label { font-size:var(--fs-xs); color:var(--muted);
        margin-top:0.35rem; font-weight:600; }
      #routineEdit input, #routineEdit select, #routineEdit textarea {
        width:100%; max-width:none; }
      #routineEdit textarea { min-height:9rem; max-height:none;
        font-size:var(--fs-sm); background:var(--surface2);
        border:1px solid var(--border); border-radius:var(--r1);
        padding:0.5rem 0.6rem; order:0; flex:none; }
      .routine-acts { display:flex; gap:0.4rem; margin-top:0.7rem; }
      .routine-acts button { flex:1 1 auto; padding:0.5rem; }

      /* A sheet from the bottom rather than a side drawer: it is reached from
         the composer, which is where a thumb already is on a phone. On a
         desktop the same panel floats clear of the edges instead. */
      .sheet {
        position:fixed; left:0; right:0; bottom:0; z-index:22; max-height:70vh;
        overflow-y:auto; background:var(--surface);
        border-top:1px solid var(--border);
        border-radius:var(--r3) var(--r3) 0 0; padding:0.8rem 0.95rem 1.2rem;
        box-shadow:var(--shadow2);
      }
      @media (min-width: 1024px) {
        .sheet { left:auto; right:1.5rem; bottom:1.5rem; width:min(30rem,42vw);
          border:1px solid var(--border); border-radius:var(--r3); }
      }
      .metarow { display:flex; gap:0.6rem; padding:0.32rem 0;
                 border-bottom:1px solid var(--border-soft); font-size:var(--fs-sm); }
      .metarow:last-child { border-bottom:0; }
      .metaname { flex:0 0 9rem; color:var(--muted); }
      .metarow a { color:var(--accent); word-break:break-all; }

      /* ---- Records ------------------------------------------------------- */
      /* Its own view: the whole pane, scrolling on its own, with the composer
         out of the way. On a phone that is 390px of table instead of 367px of
         drawer over a conversation you cannot see anyway. */
      .recordsview { flex:1 1 auto; min-height:0; overflow:auto;
                     padding:1rem 1rem 1.5rem; }
      body.records .chatarea, body.records footer { display:none !important; }
      .recordsview .wrap { max-width:none; }
      .exportbar { display:flex; gap:0.4rem; align-items:center; margin:0.3rem 0 0.6rem; }
      .exportbar a { text-decoration:none; }
      .recordfilter { flex-wrap:wrap; margin-bottom:0.4rem; }
      /* min-width:100% rather than width:100%: fill a wide drawer when four
         columns would otherwise huddle at the left edge, while still being
         allowed to grow past it and scroll inside .tablewrap when a dozen
         columns cannot fit. These sit above the narrow-screen block below
         because that block has to override them, and both selectors weigh the
         same — so the later one wins. */
      #recordList table { border-collapse:collapse; min-width:100%;
                          font-size:var(--fs-xs); }
      #recordList th, #recordList td { border:1px solid var(--border);
        padding:0.35rem 0.5rem; text-align:left; vertical-align:top; }
      #recordList th { background:var(--surface2); font-weight:600;
                       white-space:nowrap; }

      /* Narrow enough and a table stops being readable at any width — the
         header alone wraps to two lines per column. Each record becomes its own
         block of labelled lines instead, which is the same data in the shape a
         phone can actually show. data-label carries the heading down. */
      @media (max-width: 720px) {
        #recordList .tablewrap { overflow-x:visible; }
        #recordList table, #recordList tbody, #recordList tr, #recordList td {
          display:block; width:auto; min-width:0;
        }
        #recordList thead { display:none; }
        #recordList tr { border:1px solid var(--border); border-radius:var(--r2);
          margin-bottom:0.5rem; padding:0.4rem 0.6rem; position:relative;
          background:var(--surface2); }
        #recordList td { border:0; padding:0.22rem 0; display:flex; gap:0.6rem; }
        #recordList td::before { content:attr(data-label); flex:0 0 7.5rem;
          color:var(--muted); }
        #recordList td:empty { display:none; }
        /* The delete button belongs in the corner of its own card. */
        #recordList td[data-label=""] { position:absolute; top:0.3rem; right:0.4rem;
          padding:0; }
        #recordList td[data-label=""]::before { content:none; }
      }

      /* A line under the reply saying what was kept, so a record appearing is
         visible when it happens rather than discovered in a drawer later. */
      .kept { font-size:var(--fs-xs); color:var(--muted); margin:0.3rem 0 0 0.3rem;
              cursor:pointer; }
      .kept:hover { color:var(--text); }
      #recordList { overflow:auto; }
      #recordList .editable { min-width:5rem; cursor:text; }
      #recordList .editable:focus { outline:1px solid var(--accent); }
      /* A value that was standardised on the way in. Quiet on purpose — this
         is almost every cell in a healthy log, so anything louder would read
         as a column of warnings. Hover (or long-press) says what it was. */
      #recordList .tidied { border-bottom:1px dotted var(--faint); }
      /* Worked out here rather than read off the answer. Marked because an
         empty one is information — it means nothing was recorded to work it
         out from — and an unmarked blank just looks like a failure. */
      #recordList .derived { color:var(--muted); font-style:italic; }
      #recordList .derived:empty::after { content:"—"; opacity:0.5; }
      /* The declaration box is prose-ish, so it gets prose-ish room. */
      #rRecord { min-height:4.5rem; resize:vertical; font-family:inherit; }
      .rechint code { font-size:0.9em; background:var(--surface2);
                      padding:0.05rem 0.25rem; border-radius:0.2rem; }
      #recordList a.drawer-new { text-align:center; text-decoration:none;
                                 padding:0.45rem; color:var(--on-accent);
                                 display:flex; align-items:center;
                                 justify-content:center; }

      /* While the keyboard is up, everything above the composer goes away —
         see fitFooter(). Not display:none on the composer itself: the point is
         to give the conversation back its screen, not to hide what you are
         typing into. */
      body.typing #voicebar, body.typing #togglebar,
      body.typing #routinebar, body.typing #insecureNote { display:none; }
      body.typing .hint { display:none; }

      /* Phones: reclaim vertical space and stop the header wrapping. */
      @media (max-width: 640px) {
        header { gap:0.4rem; padding:0.45rem 0.6rem; flex-wrap:nowrap; }
        .model-label { display:none; }          /* the dropdown speaks for itself */
        #statusText { display:none; }           /* the coloured dot already says it */
        #model { max-width:8.5rem; font-size:var(--fs-xs); padding:0.35rem 0.4rem; }
        #newChat span { display:none; }         /* the ＋ is the whole button here */
        #newChat { padding:0; }
        #chat { padding:1rem 0.7rem 0.4rem; }
        footer { padding:0.45rem 0.6rem 0.6rem; }
        .col { max-width:92%; }
        textarea { font-size:1rem; }            /* < 1rem makes iOS zoom on focus */
        .composer { gap:0.3rem; }
        button.primary { padding:0.5rem 0.9rem; }
        .voicebar { gap:0.35rem; margin-bottom:0.4rem; }
        #webnote { display:none; }   /* the tooltip covers it */
        #routinebar, #togglebar { flex-wrap:nowrap; overflow-x:auto;
                                   scrollbar-width:none; }
        #togglebar::-webkit-scrollbar { display:none; }
        #togglebar .voicebar-check { flex:0 0 auto; white-space:nowrap; }
        #exifnote, #webnote { display:none; }   /* the tooltips carry these */
        .empty { padding:1.5rem 0.5rem 0.5rem; }
        .startcard { min-width:8rem; }
        /* Every control here measured under 44px, which is the minimum every
           platform guideline asks for; the checkbox rows were 14px tall. The
           padding is on the label rather than the box, because the label is
           what a thumb actually lands on. */
        .chip { padding:0.6rem 0.8rem; min-height:44px; display:flex; align-items:center; }
        .voicebar-check { min-height:44px; }
        .composer button { min-height:44px; padding:0.5rem 0.6rem; }
        .composer .iconbtn { width:44px; height:44px; }
        /* Measured at 34px square before this: .iconbtn is sized for a
           mouse, and the header is three of them in a row on a phone. */
        #menu, #theme, #newChat { min-height:40px; min-width:40px;
                                  width:40px; height:40px; }
      }

      /* The button shows the theme it would switch you TO, which is the one
         thing about a theme toggle people read without being told. */
      #theme .i-sun { display:none; }
      @media (prefers-color-scheme: dark) {
        :root:not([data-theme="light"]) #theme .i-moon { display:none; }
        :root:not([data-theme="light"]) #theme .i-sun { display:block; }
      }
      :root[data-theme="dark"] #theme .i-moon { display:none; }
      :root[data-theme="dark"] #theme .i-sun { display:block; }
      :root[data-theme="light"] #theme .i-moon { display:block; }
      :root[data-theme="light"] #theme .i-sun { display:none; }
    </style>
  </head>
  <body>
    <!-- One sprite for every icon-only control. Inline SVG rather than emoji:
         emoji are drawn by the platform in full colour at whatever weight it
         likes, so a paperclip, a camera and a screenshot button sat side by
         side in three different visual styles. These inherit currentColor and
         one stroke width, so a row of them reads as one set of controls. -->
    <svg class="sprite" aria-hidden="true" focusable="false">
      <symbol id="i-menu" viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16"/></symbol>
      <symbol id="i-close" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></symbol>
      <symbol id="i-plus" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></symbol>
      <symbol id="i-clip" viewBox="0 0 24 24"><path d="M20.5 11.6l-8.4 8.4a5 5 0 0 1-7.1-7.1l8.5-8.5a3.5 3.5 0 0 1 4.9 4.9l-8.4 8.4a2 2 0 0 1-2.8-2.8l7.8-7.8"/></symbol>
      <symbol id="i-camera" viewBox="0 0 24 24"><path d="M3 9a2 2 0 0 1 2-2h2l1.4-2h7.2L17 7h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><circle cx="12" cy="13.5" r="3.4"/></symbol>
      <symbol id="i-shot" viewBox="0 0 24 24"><path d="M9 3H5a2 2 0 0 0-2 2v4M15 3h4a2 2 0 0 1 2 2v4M9 21H5a2 2 0 0 1-2-2v-4M15 21h4a2 2 0 0 0 2-2v-4"/></symbol>
      <symbol id="i-mic" viewBox="0 0 24 24"><path d="M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></symbol>
      <symbol id="i-stop" viewBox="0 0 24 24"><rect x="6.5" y="6.5" width="11" height="11" rx="2"/></symbol>
      <!-- Two sheets, the usual "copy" shorthand. On a phone this button opens
           the share sheet rather than the clipboard, but the meaning people
           read off the icon — "take this text away with me" — is the same. -->
      <symbol id="i-copy" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></symbol>
      <symbol id="i-eraser" viewBox="0 0 24 24"><path d="M4 16l8-8 6 6-5 5H7z"/><path d="M9 21h11"/></symbol>
      <symbol id="i-sliders" viewBox="0 0 24 24"><path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2.2"/><circle cx="9" cy="17" r="2.2"/></symbol>
      <symbol id="i-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"/></symbol>
      <symbol id="i-moon" viewBox="0 0 24 24"><path d="M20.5 14.4A8.6 8.6 0 0 1 9.6 3.5a8.6 8.6 0 1 0 10.9 10.9z"/></symbol>
      <symbol id="i-back" viewBox="0 0 24 24"><path d="M19 12H5M11 6l-6 6 6 6"/></symbol>
      <symbol id="i-down" viewBox="0 0 24 24"><path d="M12 5v14M6 13l6 6 6-6"/></symbol>
      <symbol id="i-search" viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/></symbol>
      <symbol id="i-spark" viewBox="0 0 24 24"><path d="M12 3l2.1 5.4L19.5 10l-5.4 2.1L12 17.5l-2.1-5.4L4.5 10l5.4-1.6z"/><path d="M18.5 16.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/></symbol>
    </svg>
    <div class="backdrop" id="backdrop" hidden></div>
    <aside class="drawer" id="drawer" hidden>
      <!-- The app names itself here, once. The bar over the conversation says
           which conversation you are in, which is the thing you cannot work
           out by looking. -->
      <div class="brand">
        <span class="mark"><svg class="i" aria-hidden="true"><use href="#i-spark"></use></svg></span>
        <h1>__TITLE__</h1>
      </div>
      <div class="drawer-head">
        <div class="drawer-tabs">
          <button id="tabChats" class="tab active" type="button">Chats</button>
          <button id="tabRoutines" class="tab" type="button">Routines</button>
          <button id="tabRecords" class="tab tab-link" type="button" title="Open the records log">Records ↗</button>
        </div>
        <button id="drawerClose" class="iconbtn" title="Close" type="button"><svg class="i" aria-hidden="true"><use href="#i-close"></use></svg></button>
      </div>
      <div id="convoPane">
        <button class="drawer-new" id="drawerNew" type="button">＋ New chat</button>
        <!-- A month in and the list is titles you no longer recognise. This
             looks inside the messages and the records, not just the titles. -->
        <div class="searchbar">
          <svg class="i" aria-hidden="true"><use href="#i-search"></use></svg>
          <input id="convoSearch" type="search" autocomplete="off"
                 placeholder="Search chats and records" aria-label="Search chats and records">
          <button id="searchClear" class="iconbtn" type="button" title="Clear the search" hidden><svg class="i" aria-hidden="true"><use href="#i-close"></use></svg></button>
        </div>
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
          <label for="rRecord">Keep a record of each run</label>
          <!-- A box rather than a line, because a field can now say what it
               holds and how it is worked out, and those do not fit on one
               line three at a time. A bare list of names still works and
               still means what it always did. -->
          <textarea id="rRecord" rows="3" placeholder="distance, elapsed, average speed
&#10;or, one per line:&#10;Total earnings: money&#10;Distance traveled: distance&#10;Earnings per mile = Total earnings / Distance traveled"></textarea>
          <p class="hint rechint">A name on its own is read from the answer.
            <code>name: kind</code> says what it holds — money, distance, speed,
            duration, timestamp, number.
            <code>name = a / b</code> is worked out here rather than by the
            model, and is left empty when there is nothing to work it out from.</p>
          <p class="convo-empty">Field names, comma separated.
            After every run the model restates its own answer as these, and the
            row lands in Records. Leave empty to keep nothing.</p>
          <p class="convo-empty" id="routineWarn"></p>
          <div class="routine-acts">
            <button class="primary" id="rSave" type="button">Save</button>
            <button id="rCancel" type="button">Cancel</button>
            <button class="danger" id="rDelete" type="button" hidden>Delete</button>
          </div>
        </div>
      </div>
    </aside>
    <aside class="sheet" id="photometa" hidden>
      <div class="drawer-head">
        <strong>📍 Photo details</strong>
        <button id="metaClose" class="iconbtn" title="Close" type="button"><svg class="i" aria-hidden="true"><use href="#i-close"></use></svg></button>
      </div>
      <div id="metaBody"></div>
    </aside>

    <div class="pane">
      <header>
        <button id="menu" class="iconbtn" title="Saved conversations" type="button" hidden><svg class="i" aria-hidden="true"><use href="#i-menu"></use></svg></button>
        <button id="backToChat" class="iconbtn" title="Back to the conversation" type="button" hidden><svg class="i" aria-hidden="true"><use href="#i-back"></use></svg></button>
        <div class="head-title"><h2 id="chatTitle">New chat</h2></div>
        <div class="spacer"></div>
        <div class="status"><span class="dot" id="dot"></span><span id="statusText">connecting…</span></div>
        <label class="status model-label" for="model">Model</label>
        <select id="model" title="Choose which local model answers"></select>
        <button id="theme" class="iconbtn" type="button" title="Switch between light and dark"><svg class="i i-sun" aria-hidden="true"><use href="#i-sun"></use></svg><svg class="i i-moon" aria-hidden="true"><use href="#i-moon"></use></svg></button>
        <button id="newChat" title="Start a new conversation" type="button"><svg class="i" aria-hidden="true"><use href="#i-plus"></use></svg><span>New chat</span></button>
      </header>

      <!-- The button is anchored to this, not to the pane: #chat scrolls, so a
           child of it would scroll away, and the pane's bottom edge is under
           the composer. -->
      <!-- Records is a place you go, not something you pick: a table you read,
           filter, correct and export, and the only part of the app that wants
           the whole window. In a 21rem drawer it was eight columns in 320px;
           in a 56rem one it pushed the conversation it came from off the
           screen. So it gets the pane instead, and the composer goes away
           while it is open — there is nothing to type at. -->
      <section id="recordPane" class="recordsview" hidden>
        <div id="recordList"></div>
      </section>

      <div class="chatarea">
        <main id="chat">
          <div class="wrap"><div class="empty" id="empty"></div></div>
        </main>
        <!-- Scrolling up during a long reply stops the view following it,
             which is right — but then getting back meant flicking down through
             everything that arrived while you were reading. -->
        <button id="toLatest" class="tolatest" type="button" title="Jump to the latest" hidden>
          <svg class="i" aria-hidden="true"><use href="#i-down"></use></svg>
        </button>
      </div>

      <footer>
        <div class="wrap">
          <div class="voicebar" id="voicebar" hidden>
            <span class="voicebar-label">🎙 Voice</span>
            <!-- Which microphone, because the browser's default is frequently
                 not the one you are speaking into: a laptop has the webcam's,
                 a desktop has whatever the monitor came with, and neither is
                 the headset. Labels only exist once permission has been
                 granted, so this fills in properly after the first recording. -->
            <select id="micDevice" title="Which microphone to record from"></select>
            <select id="voiceModel" title="Speech recognition language"></select>
            <!-- Reading replies out is independent of dictation: this appears
                 wherever the browser has voices, with or without vosk. -->
            <select id="ttsVoice" title="Which voice reads replies aloud" hidden></select>
            <!-- Whether anything is actually arriving, and whether it is
                 clipping. "It transcribed wrongly" and "it heard nothing" look
                 identical without this. -->
            <span class="level" id="micLevel" hidden aria-hidden="true">
              <span class="level-bar" id="micLevelBar"></span>
              <span class="level-peak" id="micLevelPeak"></span>
            </span>
          </div>
          <p class="hint insecure" id="insecureNote" hidden></p>
          <!-- One row, not two. Each toggle hides independently — web access has
               a server-side kill switch — but two single-checkbox rows cost 48px
               of a 844px phone before anything has been typed. -->
          <div class="voicebar" id="togglebar">
            <label class="voicebar-check" id="webbar" hidden title="Let this app read the web for you: a link in your message is fetched, and otherwise the model is asked whether a search would help. Sources are listed under the reply.">
              <input type="checkbox" id="web"> 🌐 Web
            </label>
            <label class="voicebar-check" id="exifbar" hidden title="Read the date, camera and GPS position a photo carries, and tell the model. The app re-encodes images, which strips this, so turning it off means the location never leaves your phone at all. Set PHOTO_META=0 on the server to make off the default.">
              <input type="checkbox" id="exif"> 📍 Photo details
            </label>
            <!-- Named for what it does, not for when to use it. Called
                 "Headphones" it invited everyone not wearing any to untick it,
                 which turns echo cancelling back on — and that is the thing
                 that was making dictation worse on a phone. -->
            <label class="voicebar-check" id="speakbar" hidden title="Read each reply aloud as it finishes. The speaker under any reply reads that one on demand, whether this is on or not.">
              <input type="checkbox" id="speak"> 🔊 Speak replies
            </label>
            <label class="voicebar-check voicesetting" title="On by default, and usually the better setting. With it off, the browser's echo cancelling ducks the mic whenever audio is playing — it mutes you over music and clips quiet speech. Untick it only if you get an echo from laptop speakers.">
              <input type="checkbox" id="headset" checked> 🎙 Raw mic
            </label>
            <label class="voicebar-check voicesetting" title="Send as soon as speech is transcribed, instead of waiting for you to press Send.">
              <input type="checkbox" id="autosend"> ⚡ Auto-send
            </label>
            <label class="voicebar-check voicesetting" title="Keep the mic open and treat a pause in speech as the end of a message. With Auto-send on, this runs a whole conversation from one tap.">
              <input type="checkbox" id="continuous"> 🔁 Continuous
            </label>
            <span class="voicebar-label" id="webnote">links you paste are read; the model decides when to search</span>
          </div>
          <!-- Above the thumbs and the composer, so the footer reads downwards
               as: pick a routine, attach its photos, write the message. -->
          <div class="voicebar" id="routinebar" hidden>
            <div class="chips" id="routineChips"></div>
            <button class="chip chip-edit" id="routineEditBtn" type="button"
                    title="Add or change a saved prompt"><svg class="i" aria-hidden="true"><use href="#i-sliders"></use></svg></button>
          </div>
          <!-- The box you type in, the photos riding with the message and the
               buttons that act on it are one card. They were three rows that
               happened to be adjacent, which is why the footer read as clutter. -->
          <div class="composer-card">
            <div class="thumbs" id="thumbs" hidden></div>
            <div class="composer">
              <textarea id="input" rows="1" placeholder="Type a message…"></textarea>
              <button id="attach" class="iconbtn" type="button" title="Attach an image (needs a vision model)"><svg class="i" aria-hidden="true"><use href="#i-clip"></use></svg></button>
              <button id="camera" class="iconbtn" type="button" title="Take a photo" hidden><svg class="i" aria-hidden="true"><use href="#i-camera"></use></svg></button>
              <button id="shot" class="iconbtn" type="button" title="Capture a screenshot to analyse" hidden><svg class="i" aria-hidden="true"><use href="#i-shot"></use></svg></button>
              <button id="mic" class="iconbtn" type="button" title="Speak (offline transcription)" hidden><svg class="i i-mic" aria-hidden="true"><use href="#i-mic"></use></svg><svg class="i i-stop" aria-hidden="true"><use href="#i-stop"></use></svg></button>
              <!-- Take the text somewhere else instead of sending it here.
                   Dictating a message for an app with no voice input is a
                   whole use of this page on its own, and before these two the
                   only ways out of the box were the Send button and selecting
                   the text by hand on a phone keyboard. Both appear only with
                   something in the box, so an empty composer is unchanged. -->
              <button id="copyOut" class="iconbtn" type="button" hidden title="Copy this text — or share it straight to another app"><svg class="i" aria-hidden="true"><use href="#i-copy"></use></svg></button>
              <button id="clearOut" class="iconbtn" type="button" hidden title="Empty the box"><svg class="i" aria-hidden="true"><use href="#i-eraser"></use></svg></button>
              <button class="primary" id="send" type="button">Send</button>
              <button class="danger" id="stop" type="button" hidden>Stop</button>
            </div>
          </div>
          <input type="file" id="file" accept="image/*" multiple hidden>
          <input type="file" id="cameraFile" accept="image/*" capture="environment" hidden>
          <p class="hint" id="hint"></p>
        </div>
      </footer>
    </div>

    <script>
      const chatEl   = document.getElementById("chat");
      const inputEl  = document.getElementById("input");
      const sendBtn  = document.getElementById("send");
      const stopBtn  = document.getElementById("stop");
      const newBtn   = document.getElementById("newChat");
      const micBtn   = document.getElementById("mic");
      const copyOutBtn  = document.getElementById("copyOut");
      const clearOutBtn = document.getElementById("clearOut");
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
      const chatTitleEl = document.getElementById("chatTitle");
      const themeBtn = document.getElementById("theme");
      const backBtn = document.getElementById("backToChat");
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
      const metaEl    = document.getElementById("photometa");
      const metaBodyEl = document.getElementById("metaBody");
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
      const rRecordEl = document.getElementById("rRecord");
      const recordPaneEl = document.getElementById("recordPane");
      const recordListEl = document.getElementById("recordList");
      const tabRecordsEl = document.getElementById("tabRecords");
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
        syncComposerTools();
      }

      // Copy and clear are only meaningful with something to act on, and an
      // empty composer is the state this page spends most of its life in — so
      // they appear with the first character and go again when the box empties.
      // Hung off autosize() rather than the input event because dictation sets
      // .value directly, which fires no input event: the button would not have
      // appeared for the one case it was built for.
      function syncComposerTools() {
        const has = !!inputEl.value.trim();
        if (copyOutBtn) copyOutBtn.hidden = !has;
        if (clearOutBtn) clearOutBtn.hidden = !has;
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
        showToLatest();
      }, { passive: true });

      // The button and the follow are the same state seen twice: it is offered
      // exactly when the view has stopped following, and taking it puts the
      // follow back on.
      const toLatestEl = document.getElementById("toLatest");
      function showToLatest() {
        toLatestEl.hidden = stickBottom || chatEl.scrollHeight <= chatEl.clientHeight + 8;
      }
      toLatestEl.addEventListener("click", () => scrollDown(true));

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
          showToLatest();
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
      const MD_SLOT = new RegExp(MD_SENTINEL + "(\\d+)" + MD_SENTINEL, "g");

      // ---- Maths written as TeX ----
      // Models reach for LaTeX the moment an answer contains arithmetic, and
      // this page has no maths renderer: no build step and nothing from a CDN,
      // so KaTeX is not on the table for a handful of subtractions. Unwrapping
      // it into plain text gives what a person reads anyway — "68 / 3.13 ≈ 21.7"
      // beats a raw \frac on a phone, and beats it by a mile unrendered.
      const TEX_SYMBOLS = [
        [/\\approx/g, "≈"], [/\\times/g, "×"], [/\\cdot/g, "·"],
        [/\\div/g, "÷"], [/\\pm/g, "±"], [/\\mp/g, "∓"],
        [/\\leq?\b/g, "≤"], [/\\geq?\b/g, "≥"], [/\\neq/g, "≠"],
        [/\\rightarrow/g, "→"], [/\\to\b/g, "→"], [/\\infty/g, "∞"],
        [/\\degree|\\deg\b/g, "°"],
      ];

      function texToText(body) {
        let out = body;
        // \text{…} and its relatives are pure wrapping; \frac wants to become a
        // division that reads left to right. Repeated because these nest.
        for (let pass = 0; pass < 4; pass++) {
          out = out.replace(/\\(?:text|mathrm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}/g, "$1");
          out = out.replace(/\\(?:d|t)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1) / ($2)");
          out = out.replace(/\\sqrt\s*\{([^{}]*)\}/g, "√($1)");
        }
        for (const pair of TEX_SYMBOLS) out = out.replace(pair[0], pair[1]);
        out = out.replace(/\\left|\\right/g, "")
                 .replace(/\\[,;:!]/g, " ")
                 .replace(/\\\\/g, " ")
                 .replace(/\\([%$&#_{}])/g, "$1")
                 // Brackets the fraction rule added, dropped again where neither
                 // side needs them: "(68) / (3.13)" is worse than "68 / 3.13".
                 .replace(/\(([^()\s]+)\) \/ \(([^()\s]+)\)/g, "$1 / $2");
        return out.replace(/\s+/g, " ").trim();
      }

      // Whether a $…$ pair is maths or two prices in one sentence. Both turn up
      // constantly and they look identical to a regex, so this leans on the one
      // convention TeX actually has: the delimiters hug their contents.
      function looksLikeMaths(body) {
        // A backslash settles it — no price contains \frac or \approx.
        if (body.indexOf("\\") >= 0) return true;
        // "It cost $100,407 and then $5 more" — the span between the two signs
        // is prose, and prose ends in a space. TeX never opens or closes on one.
        if (/^\s|\s$/.test(body)) return false;
        // "$5-$10" is a price range: the span ends on the operator rather than
        // on something for it to operate on.
        if (/^[-+*/=<>,^_]|[-+*/=<>,^_]$/.test(body)) return false;
        // Past that it has to actually do something to be maths at all, or
        // "$100,407$" alone would lose its signs on a guess.
        return /[-+*/=<>^_≈±×÷]/.test(body);
      }

      function unTex(t) {
        return t.replace(
          /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$|\\\(([\s\S]+?)\\\)|\\\[([\s\S]+?)\\\]/g,
          function (whole, a, b, c, d) {
            const dollars = a !== undefined ? a : b;
            if (dollars !== undefined) {
              if (!looksLikeMaths(dollars)) return whole;
              return texToText(dollars);
            }
            // \( \) and \[ \] say what they are; nothing else uses them.
            return texToText(c !== undefined ? c : d);
          });
      }

      // ---- Tables ----
      // A model asked to compare two things reaches for a pipe table almost
      // every time, and without this the whole grid came out as one paragraph
      // of pipes and dashes joined by <br> — which is how the odometer answer
      // arrived: correct numbers, unreadable layout.
      const TABLE_DIVIDER_RE = /^\s*\|?(?:\s*:?-{1,}:?\s*\|)+\s*:?-{1,}:?\s*\|?\s*$/;

      function tableCells(line) {
        let row = line.trim();
        // The outer pipes are optional in the format models actually emit.
        if (row.slice(0, 1) === "|") row = row.slice(1);
        if (row.slice(-1) === "|") row = row.slice(0, -1);
        return row.split("|").map(function (c) { return c.trim(); });
      }

      function alignOf(spec) {
        const left = spec.slice(0, 1) === ":", right = spec.slice(-1) === ":";
        if (left && right) return " style=\"text-align:center\"";
        if (right) return " style=\"text-align:right\"";
        return "";
      }

      function inlineMd(t) {
        // Lift code spans out before anything else runs. Applying the emphasis
        // and link rules to inlineMd's own output means markdown *inside* a code
        // span gets eaten: `*args` rendered as an italic "args", and two spans
        // each containing `*` mis-nested tags across both of them.
        const spans = [];
        let out = t.replace(/`([^`\n]+)`/g, function (_, code) {
          spans.push(code);
          return MD_SENTINEL + (spans.length - 1) + MD_SENTINEL;
        });
        out = unTex(out);
        // Emphasis needs a non-space character inside both delimiters, as
        // CommonMark requires. Without that, prose like "SELECT * FROM a …
        // SELECT * FROM b" turns everything between two unrelated asterisks
        // into <em> and deletes both asterisks from the page.
        out = out
          .replace(/\*\*([^\s*][^*\n]*[^\s*]|[^\s*])\*\*/g, "<strong>$1</strong>")
          .replace(/(^|[^*\w])\*([^\s*][^*\n]*[^\s*]|[^\s*])\*/g, "$1<em>$2</em>")
          .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
                   '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        return out.replace(MD_SLOT, function (_, i) {
          return "<code>" + spans[Number(i)] + "</code>";
        });
      }

      // Walk the text line by line rather than deciding a whole blank-line
      // chunk's type at once. The chunk approach needed a blank line before
      // every construct, so the very common "## Summary\nText" and
      // "Here are steps:\n- one" rendered their markers literally.
      function renderMarkdown(raw) {
        // Split on any line ending, not just \n. A stray \r survives a plain
        // split and then defeats every block pattern below, because "." does not
        // match a carriage return and "$" without the m flag only matches
        // end-of-input — so CRLF output rendered its markers literally.
        const lines = esc(raw).split(/\r\n|\r|\n/);
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
          const fenceMatch = line.match(/^\s*```(.*)$/);
          if (fence) {
            if (fenceMatch) {
              emitCode(fence.lang, fence.body.join("\n"));
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
            const solo = line.match(/^\s*```(.*?)```\s*$/);
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
          if (/^ {0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
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

          const heading = line.match(/^ {0,3}(#{1,6})\s+(.*)$/);
          if (heading) {
            flushAll();
            const level = Math.min(6, heading[1].length + 2);
            out.push("<h" + level + ">" + inlineMd(heading[2].trim()) + "</h" + level + ">");
            continue;
          }

          const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
          if (bullet) {
            flushPara(); flushQuote();
            if (!list || list.tag !== "ul") { flushList(); list = { tag: "ul", items: [], start: 1 }; }
            list.items.push(bullet[1]);
            continue;
          }

          const numbered = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
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

          const quoted = line.match(/^\s*&gt;\s?(.*)$/);
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
        if (fence) emitCode(fence.lang, fence.body.join("\n"));
        flushAll();
        return out.join("");
      }

      // Painted once the reply completes, not per token: a half-arrived fence
      // Telling someone to press Ctrl+C is only useful if the text is selected.
      function selectText(node) {
        try {
          // A form field keeps its text in .value, not in child nodes, so a
          // Range over its contents selects nothing at all — and "Press Ctrl+C"
          // would then copy an empty selection while looking like it worked.
          if (node && typeof node.select === "function") {
            node.focus();
            node.select();
            return;
          }
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

      // What the turn actually did, before the model was asked anything: which
      // photo details were read, what was searched for, what came back, what
      // was put in front of it. Collapsed, because it is only ever wanted when
      // an answer looks wrong — and then it is the only thing that helps.
      function addStep(view, entry) {
        if (!view || !view.stepsBody || !entry || !entry.step) return;
        view.steps.hidden = false;
        const row = document.createElement("div");
        row.className = "steprow";
        const name = document.createElement("b");
        name.textContent = entry.step;
        row.appendChild(name);
        if (entry.detail) {
          const detail = document.createElement("span");
          // textContent throughout: a step can carry a page title, a search
          // query or a model name, none of which this app wrote.
          detail.textContent = entry.detail;
          row.appendChild(detail);
        }
        // The long parts — the text handed to the model, the URLs — fold away
        // again, so the panel stays a list of one-liners until you want more.
        const extras = [];
        if (entry.text) extras.push(entry.text);
        if (entry.system && entry.system.length) extras.push(entry.system.join("\n\n---\n\n"));
        if (entry.urls && entry.urls.length) extras.push(entry.urls.join("\n"));
        if (extras.length) {
          const more = document.createElement("details");
          more.className = "stepmore";
          const sum = document.createElement("summary");
          sum.textContent = "show";
          const body = document.createElement("pre");
          body.textContent = extras.join("\n\n");
          more.appendChild(sum); more.appendChild(body);
          row.appendChild(more);
        }
        view.stepsBody.appendChild(row);
      }

      // Split assistant text into visible content + inline <think> reasoning.
      function splitThink(raw) {
        let content = "", thinking = "", last = 0, m;
        const re = /<think>([\s\S]*?)(<\/think>|$)/g;
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
          const orphan = content.match(/^[^\S\n]*<\/think>[^\S\n]*$/m);
          // Not inside a fenced code block. Ask a coder model what a reasoning
          // model's output looks like and it shows you one — tag on its own
          // line, inside ``` — and treating that as a real terminator threw
          // away the half of the reply that explained it. An odd number of
          // fences before the tag means we are inside one.
          const fenced = orphan &&
            (content.slice(0, orphan.index).match(/^[^\S\n]*```/gm) || []).length % 2 === 1;
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
          el.title = "what was read from this photo";
          el.addEventListener("click", () => showPhotoDetails(img));
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
                           typeof img.meta.lon === "number";
          if (hasPlace) badges.push(["📍", "location read from the photo"]);
          else if (img.meta && img.meta.gpsBlock) {
            badges.push(["📍̸", "the camera tried but had no GPS fix"]);
          }
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
        // Walk the marker segments looking for APP1 with an "Exif\0\0" payload.
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
          // Say that the camera tried. A block with no fix in it is the usual
          // answer for a photo taken in a garage, and it is a different thing
          // from a camera that never records position at all — which is the
          // difference between "wait for a fix" and "turn the setting on".
          out.gpsBlock = true;
          readGps(view, base, base + ifd0[0x8825], le, out);
        }
        const facts = Object.keys(out).filter(k => k !== "gpsBlock");
        return facts.length ? out : null;
      }

      function readGps(view, base, at, le, out) {
        const gps = readIfd(view, base, at, le) || {};
        const lat = dms(gps[0x0002], gps[0x0001]);
        const lon = dms(gps[0x0004], gps[0x0003]);
        if (lat !== null && lon !== null && !(lat === 0 && lon === 0)) {
          out.lat = lat;
          out.lon = lon;
        }
        if (typeof gps[0x0006] === "number") {
          out.altitude = Math.round(gps[0x0006] * (gps[0x0005] === 1 ? -1 : 1));
        }
        // How far out the fix might be, in metres. Worth having: "at 44.9778,
        // -93.2650" reads as a doorstep and can be a whole block.
        const error = ratio(gps[0x001f], 1);
        if (error) out.accuracy = error;
        // Dilution of precision: the geometry of the satellites at the time,
        // which is the other half of "how far out might this be". Under 2 is a
        // good fix, over 5 is one to distrust — and a phone that has just woken
        // up indoors writes a position with a DOP of 20 without complaint.
        const dop = ratio(gps[0x000b], 1);
        if (dop) out.dop = dop;
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

      // Why a photo carries no position, said in words at the moment it is
      // attached. A badge is only useful to someone who already knows to look
      // for it; silence here read as "the app is broken" for long enough to be
      // worth a sentence.
      function noteMissingPlace(att) {
        if (!exifOn) return;
        if (att.meta && typeof att.meta.lat === "number") return;
        if (!att.meta) {
          hintEl.textContent = "No photo details in that file — screenshots have " +
            "none, and neither does anything re-encoded on the way here.";
          return;
        }
        const had = att.meta.taken ? "That photo has its date but " : "That photo has ";
        hintEl.textContent = att.meta.gpsBlock
          ? had + "no GPS fix — the camera asked and did not get one. " +
            "Indoors is the usual reason."
          : had + "no location. The camera did not record one: check location " +
            "is on for the camera app.";
      }

      // ---- What was read, on the phone ----
      // Tapping a thumbnail shows every fact pulled out of that file. The
      // question this answers — "does my photo actually have a position in
      // it?" — otherwise needs a terminal and a diagnostic script, which is a
      // silly thing to need while standing next to the car.
      const META_ROWS = [
        ["taken", "Taken"],
        ["offset", "Time zone"],
        ["utc", "GPS clock (UTC)"],
        ["accuracy", "Position error", (v) => "± " + v + " m"],
        ["dop", "Fix quality", (v) => v + " DOP"],
        ["altitude", "Altitude", (v) => v + " m"],
        ["speed", "Speed"],
        ["heading", "Facing"],
        ["camera", "Camera"],
        ["lens", "Lens"],
        ["exposure", "Exposure"],
        ["aperture", "Aperture"],
        ["focal", "Focal length"],
        ["focal35", "35 mm equivalent"],
        ["iso", "ISO"],
        ["flash", "Flash", (v) => (v ? "fired" : "did not fire")],
        ["orientation", "Camera held"],
        ["software", "Written by"],
        ["artist", "Artist"],
        ["copyright", "Copyright"],
      ];

      function showPhotoDetails(img) {
        metaBodyEl.innerHTML = "";
        const meta = img.meta;
        const add = (label, value, href) => {
          const row = document.createElement("div");
          row.className = "metarow";
          const name = document.createElement("span");
          name.className = "metaname";
          name.textContent = label;
          row.appendChild(name);
          if (href) {
            const link = document.createElement("a");
            link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer";
            link.textContent = value;
            row.appendChild(link);
          } else {
            const val = document.createElement("span");
            val.textContent = value;      // never innerHTML: this is file content
            row.appendChild(val);
          }
          metaBodyEl.appendChild(row);
        };

        if (!exifOn) {
          add("Photo details", "turned off — nothing was read from this file");
        } else if (!meta) {
          add("Photo details", "none in this file");
          add("Why", "screenshots carry none, and neither does anything " +
                     "re-encoded on the way here");
        } else {
          if (typeof meta.lat === "number" && typeof meta.lon === "number") {
            const here = meta.lat.toFixed(6) + ", " + meta.lon.toFixed(6);
            add("Position", here,
                "https://www.openstreetmap.org/?mlat=" + meta.lat +
                "&mlon=" + meta.lon + "#map=16");
          } else if (meta.gpsBlock) {
            add("Position", "none — the camera asked for a fix and did not get one");
          } else {
            add("Position", "none — the camera did not record one");
          }
          if (typeof meta.width === "number" && typeof meta.height === "number") {
            add("Size as taken", meta.width + " × " + meta.height);
          }
          for (const row of META_ROWS) {
            const value = meta[row[0]];
            if (value === undefined || value === null || value === "") continue;
            add(row[1], row[2] ? row[2](value) : String(value));
          }
        }
        metaEl.hidden = false;
        backdropEl.hidden = false;
      }

      function hidePhotoDetails() {
        metaEl.hidden = true;
        // The drawer uses the same backdrop, so only take it away if it is ours.
        if (drawerEl.hidden || railed()) backdropEl.hidden = true;
      }

      function addAttachment(att) {
        if (pendingImages.length >= 4) { hintEl.textContent = "Up to 4 images per message."; return false; }
        pendingImages.push(att); renderThumbs();
        noteMissingPlace(att);
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
            '<details class="think steps" hidden><summary>Show what it did</summary>' +
              '<div class="steps-body"></div></details>' +
            '<div class="bubble" aria-live="polite" aria-atomic="false">…</div>' +
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
          think: wrap.querySelector("details.think:not(.steps)"),
          thinkBody: wrap.querySelector(".think-body"),
          steps: wrap.querySelector("details.steps"),
          stepsBody: wrap.querySelector(".steps-body"),
          meta: wrap.querySelector(".meta"),
          sources: wrap.querySelector(".sources"),
          raw: "",
        };
        addReplyCopy(view);
        return view;
      }

      // Getting text out of this page and into something else. Three routes,
      // because no single one works everywhere:
      //
      //   1. The share sheet, on a touch device. This is the good one: it
      //      reaches WhatsApp, Signal, Messages — the apps you would actually
      //      paste into — without a clipboard round trip, and it works on a
      //      plain-HTTP page where navigator.clipboard does not exist at all.
      //   2. The clipboard, on a desktop or where sharing is unavailable.
      //   3. Selecting the text and saying which keys to press, when the page
      //      is not a secure context. Nothing else is possible there, and an
      //      inert button that looks like it worked is worse than a hint.
      //
      // `onDone` is told which one happened so the caller can say so. Silence
      // is the enemy here: a copy that did nothing looks exactly like a copy
      // that worked, and the user finds out in the other app.
      async function shareOrCopy(text, selectable, onDone) {
        if (!text) return false;
        if (navigator.share && window.matchMedia &&
            window.matchMedia("(pointer: coarse)").matches) {
          try { await navigator.share({ text: text }); onDone("shared"); return true; }
          catch (e) {
            // Dismissing the share sheet rejects too. That is not a failure to
            // report — falling through to "select and copy" made cancel look
            // like an error.
            if (e && e.name === "AbortError") { onDone("cancelled"); return false; }
          }
        }
        if (!navigator.clipboard) {
          if (selectable) selectText(selectable);
          onDone("manual");
          return false;
        }
        try {
          await navigator.clipboard.writeText(text);
          onDone("copied");
          return true;
        } catch (e) {
          if (selectable) selectText(selectable);
          onDone("manual");
          return false;
        }
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
          await shareOrCopy(text, view.bubble, (how) => {
            if (how === "copied") flash("Copied");
            else if (how === "manual") flash("Press Ctrl+C");
          });
        });
        view.meta.parentElement.insertBefore(btn, view.meta.nextSibling);
        view.copyBtn = btn;
        addReplySpeak(view);
      }

      // On demand, whatever the toggle says: wanting one reply read out is not
      // the same as wanting all of them read out, and it is the more common of
      // the two.
      function addReplySpeak(view) {
        if (!ttsReady()) return;
        const btn = document.createElement("button");
        btn.type = "button";
        // Its own class as well: the label changes to "Stop reading" while it
        // is going, so the label is not something to find it by.
        btn.className = "replycopy replyspeak";
        btn.textContent = "🔊 Read aloud";
        btn.hidden = true;
        btn.addEventListener("click", () => speakReply(view));
        view.copyBtn.parentElement.insertBefore(btn, view.copyBtn.nextSibling);
        view.speakBtn = btn;
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
          // The dictation toggles sit in the same row now, so they need their
          // own gate: without vosk they are three settings for a button that
          // is not there.
          for (const el of document.querySelectorAll(".voicesetting")) {
            el.hidden = !data.voice;
          }
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

      function mb(bytes) {
        const n = (bytes || 0) / (1024 * 1024);
        return (n >= 100 ? n.toFixed(0) : n.toFixed(1)) + " MB";
      }

      // Downloading a not-yet-present language happens on selection so the mic
      // is ready before you speak.
      voiceSel.addEventListener("change", async () => {
        remember("voice", voiceSel.value);
        const opt = voiceSel.selectedOptions[0];
        if (!opt || opt.dataset.download !== "1") return;
        const id = opt.value;
        voiceSel.disabled = true; micBtn.disabled = true;
        const name = opt.textContent.replace(" ⬇", "");
        hintEl.textContent = "Downloading " + name + "… (one-time)";
        try {
          // NDJSON, like the chat: a percent line as it goes and one final
          // object. The large English model is 1.8 GB, and a silent ten-minute
          // request looks exactly like a broken one.
          const resp = await fetch("api/voice/download", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id }),
          });
          if (!resp.ok || !resp.body) {
            const j = await resp.json().catch(() => ({}));
            throw new Error(j.error || ("HTTP " + resp.status));
          }
          const reader = resp.body.getReader();
          const dec = new TextDecoder();
          let buf = "", done = null;
          for (;;) {
            const step = await reader.read();
            if (step.done) break;
            buf += dec.decode(step.value, { stream: true });
            let at;
            while ((at = buf.indexOf("\n")) >= 0) {
              const raw = buf.slice(0, at).trim();
              buf = buf.slice(at + 1);
              if (!raw) continue;
              let msg;
              try { msg = JSON.parse(raw); } catch (e) { continue; }
              if (msg.error) throw new Error(msg.error);
              if (msg.percent !== undefined) {
                hintEl.textContent = "Downloading " + name + "… " +
                  (msg.total ? msg.percent + "%  (" + mb(msg.downloaded) + " of " +
                               mb(msg.total) + ")"
                             : mb(msg.downloaded));
              } else { done = msg; }
            }
          }
          if (!done) throw new Error("the download ended without finishing");
          hintEl.textContent = done.already
            ? (done.label || id) + " is already downloaded."
            : "Ready — " + (done.label || id) + " downloaded.";
          await loadVoiceModels();
          voiceSel.value = id;
        } catch (e) { hintEl.textContent = "Download failed: " + (e.message || e); }
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
        // Asking the next question is the clearest possible sign you are done
        // listening to the answer to the last one.
        stopSpeaking();
        let went = true, queued = false;
        // Hidden, not merely disabled: a greyed-out Send next to a red Stop
        // is two buttons for one decision.
        busy = true; sendBtn.disabled = true; sendBtn.hidden = true;
        stopBtn.hidden = false;
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
        // How many messages the server has for this thread right now — the
        // turn about to start is not among them, since the server writes the
        // question and the answer together when generation ends. If this
        // connection drops, that is the number the recovery has to see the
        // stored thread grow past. Without it, "the thread ends in an
        // assistant message" was already true of the *previous* turn, so a
        // dropped follow-up concluded on its first poll that the reply had
        // landed, re-read the thread, and wiped the question just asked off
        // the screen while the real reply was still being generated.
        const storedBefore = messages.length;
        messages.push(userMsg);
        inputEl.value = ""; autosize();
        // The thread has to exist before the stream starts, because the server
        // is the one that writes the turn down now and it needs somewhere to
        // put it. A turn that then produces nothing leaves an empty thread, so
        // remember whether this call is what created it — see dropIfEmpty().
        const hadThread = !!currentConvoId;
        await ensureConversation(text);
        const createdThread = !hadThread && !!currentConvoId;

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

        // The server writes the turn — see _keep_turn in app.py — because it
        // is still there when the tab is not. This end only has to catch up
        // with the list, and to take away a thread that was created for a turn
        // which then produced nothing.
        function commitTurn() {
          if (messages !== thread) return;
          // Safe to read straight away: the server writes the turn before it
          // closes the stream, so by the time this runs it is already there.
          refreshConversations();
        }

        async function dropIfEmpty() {
          if (!createdThread || !currentConvoId) return;
          try {
            const resp = await fetch("api/conversations/" + currentConvoId);
            if (resp.ok && !((await resp.json()).messages || []).length) {
              await fetch("api/conversations/" + currentConvoId, { method: "DELETE" });
              if (messages === thread) currentConvoId = null;
              refreshConversations();
            }
          } catch (e) { /* a tidy-up is never worth an error on screen */ }
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
              // Where to write this turn down when it is finished — including
              // when this tab is not the thing that finishes it.
              conversation_id: currentConvoId || undefined,
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
            while ((nl = buf.indexOf("\n")) >= 0) {
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
              if (obj.debug) { addStep(view, obj.debug); scrollDown(); continue; }
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
            if (view.speakBtn) view.speakBtn.hidden = false;
            if (messages === thread) messages.push({ role: "assistant", content: finalContent });
            commitTurn();
            // On completion, not while streaming: chunking a half-arrived
            // sentence reads it wrong, and re-reading the corrected version
            // reads it twice.
            if (speakEl.checked) speakReply(view);
            if (usage) view.meta.textContent = fmtUsage(usage);
          } else {
            // The request completed but the model said nothing. Recording the
            // question without an answer left the store disagreeing with the
            // screen and made the next send post two user turns; roll the whole
            // turn back and hand the message back so it can be retried.
            view.root.remove();
            went = false;
            const returned = rollBackTurn();
            dropIfEmpty();
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
              commitTurn();   // Stop still produced an answer, and the server kept it
            } else {
              // Stopped before any visible text — which for a reasoning model is
              // the whole scratchpad, i.e. exactly when Stop gets pressed. The
              // turn used to stay on screen and in messages[] while never being
              // written, so the two devices diverged and the next send posted
              // two user roles.
              view.root.remove();
              went = false;
              dropIfEmpty();
              // Only claim the message came back if it did: with the composer
              // already holding something else it is not restored, and saying
              // otherwise sent people looking for a photo that had been dropped.
              if (rollBackTurn()) {
                hintEl.textContent = "Stopped before the model replied — your message is back in the box.";
              }
            }
          } else if (messages === thread && currentConvoId) {
            // The connection went, not the turn.
            //
            // Reported: tabbing out while the model was thinking came back as
            // "network error", with the question pulled off the screen and
            // dropped into the composer — and the empty thread deleted behind
            // it. All three were wrong, because the server does not stop when
            // this connection does: generation is detached on purpose and the
            // finished turn is written to the thread either way. The reply was
            // arriving; nothing was listening, and then the evidence was
            // tidied away.
            //
            // So: keep the question, keep whatever arrived, say what happened,
            // and go and collect the answer.
            view.bubble.textContent = splitThink(rawContent).content || "…";
            view.status.textContent = "Lost the connection — the reply is still "
              + "being written. Waiting for it…";
            view.status.hidden = false;
            setStatus("bad", "reconnecting");
            waitForTurn(currentConvoId, view, err, storedBefore);
          } else {
            // No thread to collect it from — an unsaved conversation, or one
            // the user has already navigated away from. An error can still
            // arrive after text is on screen, and discarding the view would
            // throw away a partial answer; keep it and report alongside. With
            // nothing rendered, drop the user turn too, or the next send posts
            // two user messages in a row.
            const partial = splitThink(rawContent).content;
            if (partial) {
              view.bubble.textContent = partial;
              view.status.textContent = "Interrupted: " + (err.message || err);
              view.status.hidden = false;
              if (messages === thread) messages.push({ role: "assistant", content: partial });
              commitTurn();
            } else {
              view.root.remove();
              went = false;
              rollBackTurn();
              dropIfEmpty();
              markError(err.message || String(err));
            }
            setStatus("bad", "error");
          }
        } finally {
          stopWaitTimer();
          releaseWakeLock();
          busy = false; sendBtn.disabled = false; sendBtn.hidden = false;
          stopBtn.hidden = true;
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

      // ---- Collecting a turn whose connection went ----
      // Only ever one of these: a second would fight the first over the same
      // thread. Kept so a new chat or another conversation can call it off.
      let waiting = null;

      function stopWaiting() {
        if (!waiting) return;
        clearTimeout(waiting.timer);
        waiting = null;
      }

      // Give up after this long. The server's own turn has a timeout well
      // inside it, so anything still missing by now is not coming.
      const WAIT_FOR_TURN_MS = 6 * 60 * 1000;

      function waitForTurn(convoId, view, err, storedBefore) {
        stopWaiting();
        waiting = { convoId: convoId, view: view, err: err, timer: 0, tries: 0,
                    storedBefore: storedBefore || 0,
                    until: Date.now() + WAIT_FOR_TURN_MS };
        lookForTurn();
      }

      async function lookForTurn() {
        const mine = waiting;
        // Whatever was being waited for has been left behind: a new chat, a
        // different conversation, a Stop. Not an error, just no longer wanted.
        // The conversation id is the whole test — newChat() and
        // loadConversation() both change it and both call stopWaiting(), and
        // `thread` is a local of the send() this outlived.
        if (!mine || currentConvoId !== mine.convoId) return;

        let landed = false, running = false;
        try {
          const resp = await fetch("api/conversations/" + mine.convoId);
          if (resp.ok) {
            const msgs = (await resp.json()).messages || [];
            // Grown *past what was already there*, not merely ending in an
            // assistant message. On the first turn of a thread those are the
            // same test; on every turn after it they are not, and the weaker
            // one was satisfied by the previous turn's answer before this one
            // had been written at all.
            landed = msgs.length > mine.storedBefore
                     && msgs[msgs.length - 1].role === "assistant";
          }
          // Only worth asking if the answer is not already there — and it is
          // asked second on purpose, because a turn that finishes between the
          // two calls would otherwise read as "not running, nothing written".
          if (!landed) {
            const st = await fetch("api/chat/status?conversation_id="
                                   + encodeURIComponent(mine.convoId));
            running = st.ok && (await st.json()).running === true;
          }
        } catch (e) {
          // Still no network. That is not an answer either way, so keep
          // waiting rather than concluding the turn died.
          running = true;
        }
        if (waiting !== mine) return;      // called off while those were in flight

        if (landed) {
          stopWaiting();
          setStatus("ok", "connected");
          // Re-read the whole thread rather than patching this one bubble: the
          // reply comes back with its thinking, its steps and its sources, and
          // this is the same path that shows a phone's turn on the desktop.
          // One way of rendering a stored turn, not two.
          await loadConversation(mine.convoId);
          refreshConversations();
          return;
        }
        if (!running || Date.now() > mine.until) {
          stopWaiting();
          view_status(mine.view, "Lost the connection, and the reply did not survive it"
                      + (mine.err && mine.err.message ? " (" + mine.err.message + ")" : "")
                      + ". Your question is still here — send it again.");
          setStatus("bad", "error");
          return;
        }
        // Back off, but start quickly and never go past a few seconds: a blip
        // on the desk resolves in under a second, and the far end of this is
        // someone watching an empty bubble waiting for their answer.
        mine.tries++;
        mine.timer = setTimeout(lookForTurn,
                                Math.min(700 * Math.pow(1.6, mine.tries), 5000));
      }

      function view_status(view, text) {
        view.status.textContent = text;
        view.status.hidden = false;
      }

      // Coming back to the tab is the moment most likely to work — the phone
      // has just been unlocked and the radio is awake. Look straight away
      // rather than sitting out the rest of a five-second backoff.
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState !== "visible" || !waiting) return;
        clearTimeout(waiting.timer);
        waiting.tries = 0;
        lookForTurn();
      });

      function stop() {
        stopSpeaking();
        stopWaiting();
        // The server keeps generating after this tab stops listening — that is
        // the whole point — so aborting here alone would leave a 30b model
        // running for another minute. keepalive, because Stop is also what
        // gets pressed on the way out of the page.
        if (currentConvoId) {
          fetch("api/chat/cancel", {
            method: "POST", keepalive: true,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_id: currentConvoId }),
          }).catch(() => {});
        }
        if (controller) controller.abort();
      }

      function newChat() {
        if (controller) controller.abort();
        stopSpeaking();
        stopWaiting();
        if (recordsOpen()) closeRecords();
        currentConvoId = null;
        messages = [];
        pendingImages = []; renderThumbs();
        clearRoutine();
        chatEl.innerHTML = '<div class="wrap"><div class="empty" id="empty"></div></div>';
        paintEmpty();
        setChatTitle("");
        inputEl.focus();
      }

      // The bar over the conversation names the conversation. Falling back to
      // the app's own name told you what you already knew; after twenty saved
      // threads, which one is on screen is the thing you cannot work out by
      // looking at it.
      // Remembered separately, because Records borrows the bar and has to give
      // it back to the right name.
      let currentTitle = "";
      function setChatTitle(text) {
        if (!recordsOpen()) currentTitle = text || "";
        chatTitleEl.textContent = text || "New chat";
        chatTitleEl.title = text || "";
      }

      // A whole screen used to hold one grey sentence. It is the only moment
      // the app gets to say what it is for, and on this app the answer is
      // usually a routine — so the routines are what it offers.
      function paintEmpty() {
        const host = document.getElementById("empty");
        if (!host) return;
        host.innerHTML = "";
        const mark = document.createElement("div");
        mark.className = "empty-mark";
        mark.innerHTML = '<svg class="i" aria-hidden="true">' +
                         '<use href="#i-spark"></use></svg>';
        host.appendChild(mark);
        const head = document.createElement("h2");
        head.textContent = "Ask your local model anything";
        host.appendChild(head);
        const sub = document.createElement("p");
        sub.textContent = "Nothing leaves your own hardware. Attach a photo and " +
          "it can read what is in it — and, with photo details on, when and " +
          "where it was taken.";
        host.appendChild(sub);
        if (!routines.length) return;
        const grid = document.createElement("div");
        grid.className = "empty-routines";
        // Four: enough to show what a routine is, few enough to stay one row
        // on a phone. The rest are a tap away on the strip above the composer.
        for (const routine of routines.slice(0, 4)) {
          const card = document.createElement("button");
          card.type = "button"; card.className = "startcard";
          const name = document.createElement("b");
          // textContent, never innerHTML: a routine name is stored text.
          name.textContent = routine.name;
          const note = document.createElement("span");
          note.textContent = routine.photos
            ? routine.photos + (routine.photos === 1 ? " photo" : " photos")
            : "Saved prompt";
          card.appendChild(name); card.appendChild(note);
          card.addEventListener("click", () => { pickRoutine(routine); inputEl.focus(); });
          grid.appendChild(card);
        }
        host.appendChild(grid);
      }


      // ---- Conversation history ----
      // The server owns storage so a thread started on one device continues on
      // another; this side just mirrors each completed message into it.
      let currentConvoId = null;
      let historyOn = false;
      let modelVision = {};   // model name -> can it answer about an image
      let visionDefault = null;  // server's pick: smallest non-OCR vision model
      let ocrAvailable = null;   // an installed transcriber, if there is one

      // Today / Yesterday / a weekday for the last week / a date. Thirty
      // threads is otherwise a wall of titles with nowhere for the eye to land.
      function dayOf(ts) {
        const then = new Date(ts * 1000), now = new Date();
        const midnight = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const days = Math.floor((midnight - new Date(
          then.getFullYear(), then.getMonth(), then.getDate())) / 86400000);
        if (days <= 0) return "Today";
        if (days === 1) return "Yesterday";
        if (days < 7) return then.toLocaleDateString(undefined, { weekday: "long" });
        if (then.getFullYear() === now.getFullYear()) {
          return then.toLocaleDateString(undefined, { month: "long", day: "numeric" });
        }
        return then.toLocaleDateString(undefined, { year: "numeric", month: "long" });
      }

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
        // A save, a delete or a finished turn all refresh the list. With a
        // search on screen that would silently replace the results with every
        // conversation, which reads as the search having failed.
        if (searchQuery()) { runSearch(); return; }
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
        let lastDay = null;
        for (const convo of list) {
          if (convo.id === currentConvoId) setChatTitle(convo.title);
          // The list is newest first, so the days come out in order and each
          // one only has to be announced when it changes.
          const day = dayOf(convo.updated_at);
          if (day !== lastDay) {
            lastDay = day;
            const mark = document.createElement("p");
            mark.className = "daymark";
            mark.textContent = day;
            convoListEl.appendChild(mark);
          }
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
            if (!confirm("Delete \"" + convo.title + "\"?")) return;
            await fetch("api/conversations/" + convo.id, { method: "DELETE" });
            if (convo.id === currentConvoId) newChat();
            refreshConversations();
          });

          row.appendChild(open); row.appendChild(ren); row.appendChild(del);
          convoListEl.appendChild(row);
        }
        showHistoryCost();
      }

      // A photo that has passed the retention cutoff. The message and the
      // reply are still there; saying the picture is gone is better than a
      // gap where one used to be.
      function expiredPhotoNote(count) {
        const note = document.createElement("p");
        note.className = "meta expired";
        note.textContent = count === 1
          ? "📷 the photo has passed the keep-for period"
          : "📷 " + count + " photos have passed the keep-for period";
        return note;
      }

      // ---- Reading replies aloud ----
      // The browser's own voices, which means nothing to download and nothing
      // to install — and, unlike everything else here, one part of the app not
      // running on your own hardware. The text never leaves the device; the
      // voices are the operating system's. A local backend can take over
      // behind speakText() without any of the controls moving.
      const speakEl = document.getElementById("speak");
      const speakBarEl = document.getElementById("speakbar");
      const ttsVoiceEl = document.getElementById("ttsVoice");
      const synth = window.speechSynthesis;
      let ttsVoices = [];
      // Which reply is being read, so its own button can say "Stop" and a new
      // turn can silence the old one.
      let speaking = null;

      function ttsReady() { return !!(synth && window.SpeechSynthesisUtterance); }

      function loadTtsVoices() {
        if (!ttsReady()) return;
        ttsVoices = synth.getVoices() || [];
        if (!ttsVoices.length) return;   // Chrome fills these in asynchronously
        const want = ttsVoiceEl.value || remembered("ttsVoice");
        ttsVoiceEl.innerHTML = "";
        // The page's language first: a browser typically ships thirty voices
        // and twenty-eight of them are not the language you are reading in.
        const here = (navigator.language || "en").slice(0, 2).toLowerCase();
        const sorted = ttsVoices.slice().sort((a, b) => {
          const mine = v => (v.lang || "").slice(0, 2).toLowerCase() === here ? 0 : 1;
          return mine(a) - mine(b) || (a.name || "").localeCompare(b.name || "");
        });
        for (const voice of sorted) {
          const opt = document.createElement("option");
          opt.value = voice.name;
          // textContent: a voice name comes from the operating system.
          opt.textContent = voice.name + (voice.lang ? " (" + voice.lang + ")" : "");
          ttsVoiceEl.appendChild(opt);
        }
        if (want && sorted.some(v => v.name === want)) ttsVoiceEl.value = want;
        ttsVoiceEl.hidden = false;
      }

      // Markdown as something worth listening to. A reply is written to be
      // read: fences, pipes and asterisks are layout, and a voice reciting
      // them is unlistenable. Code is skipped rather than spelled out —
      // "async function paint open bracket" helps nobody — but it is named, so
      // you know something was there.
      function speechText(raw) {
        let out = String(raw || "");
        out = out.replace(/```[\s\S]*?```/g, function (block) {
          // An empty fence is nothing to announce.
          return block.replace(/```/g, "").trim() ? " (code block) " : " ";
        });
        out = out.replace(/`([^`\n]+)`/g, "$1");
        // Tables are layout too; read the cells as a list rather than
        // reciting the pipes. The outer pipes go first, or every row is read
        // as "comma Leg comma Miles comma".
        out = out.replace(/^\s*\|?[\s:|-]+\|[\s:|-]*$/gm, " ");
        // Row by row, so one ends before the next begins: run together they
        // came out as "Leg, Miles Out, 41 Back, 39".
        out = out.replace(/^([^\n]*\|[^\n]*)$/gm, function (row) {
          const cells = row.replace(/^[ \t]*\|/, "").replace(/\|[ \t]*$/, "")
                           .replace(/\s*\|\s*/g, ", ").trim();
          return cells ? cells + "." : " ";
        });
        out = out.replace(/!\[[^\]]*\]\([^)]*\)/g, " ");
        out = out.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");
        out = out.replace(/^\s{0,3}#{1,6}\s+/gm, "");
        out = out.replace(/^\s{0,3}>\s?/gm, "");
        // A rule before a bullet: "- - -" is a rule, and the bullet rule would
        // eat the first dash and leave the other two to be read out.
        out = out.replace(/^\s{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/gm, " ");
        out = out.replace(/^\s*[-*+]\s+/gm, "");
        out = out.replace(/\*\*([^*]+)\*\*/g, "$1").replace(/\*([^*]+)\*/g, "$1");
        out = out.replace(/[ \t]+/g, " ");
        // A paragraph break is a full stop to a listener — but only where
        // there is not one already, or a heading is read "Trip summary..".
        out = out.replace(/([^.!?:,;])\s*\n{2,}\s*/g, "$1. ");
        return out.replace(/\s*\n+\s*/g, " ").replace(/\s+/g, " ").trim();
      }

      // Chrome stops speaking after roughly fifteen seconds of one utterance,
      // so a long reply is queued a sentence at a time. It also reads better:
      // the queue can be cancelled between sentences rather than only at the
      // end.
      function speechChunks(text) {
        const parts = [];
        for (const piece of text.split(/(?<=[.!?…])\s+|\n+/)) {
          const line = piece.trim();
          if (!line) continue;
          // Still too long for one utterance: break on the next best thing.
          if (line.length <= 220) { parts.push(line); continue; }
          let rest = line;
          while (rest.length > 220) {
            let at = rest.lastIndexOf(", ", 220);
            if (at < 60) at = rest.lastIndexOf(" ", 220);
            if (at < 60) at = 220;
            parts.push(rest.slice(0, at).trim());
            rest = rest.slice(at).trim();
          }
          if (rest) parts.push(rest);
        }
        return parts;
      }

      function stopSpeaking() {
        if (!ttsReady()) return;
        const was = speaking;
        speaking = null;
        try { synth.cancel(); } catch (e) {}
        if (was && was.button) was.button.textContent = "🔊 Read aloud";
      }

      function speakReply(view) {
        if (!ttsReady()) return;
        // The same button stops it: a reply being read is the one thing you
        // want to interrupt, and hunting for a separate control to do it is
        // the wrong answer.
        if (speaking && speaking.view === view) { stopSpeaking(); return; }
        stopSpeaking();
        const text = speechText(view.raw || view.bubble.textContent || "");
        if (!text) return;
        const chunks = speechChunks(text);
        if (!chunks.length) return;
        const chosen = ttsVoices.find(v => v.name === ttsVoiceEl.value);
        const mine = { view: view, button: view.speakBtn };
        speaking = mine;
        if (mine.button) mine.button.textContent = "◼ Stop reading";
        chunks.forEach((part, i) => {
          const say = new SpeechSynthesisUtterance(part);
          if (chosen) { say.voice = chosen; say.lang = chosen.lang; }
          if (i === chunks.length - 1) {
            // Only the last one clears the state, and only if it is still ours
            // — a new reply may have taken over while this was queued.
            say.onend = () => { if (speaking === mine) stopSpeaking(); };
          }
          say.onerror = () => { if (speaking === mine) stopSpeaking(); };
          try { synth.speak(say); } catch (e) { stopSpeaking(); }
        });
      }

      // ---- Search ----
      // Everything you have said and everything a routine wrote down. A month
      // in, the conversation list is a column of titles you no longer
      // recognise, and the thing you actually remember is a word from inside
      // the answer.
      const searchEl = document.getElementById("convoSearch");
      const searchClearEl = document.getElementById("searchClear");
      let searchTimer = null;

      function searchQuery() { return searchEl.value.trim(); }

      function onSearchInput() {
        searchClearEl.hidden = !searchEl.value;
        // Debounced: this scans the messages table, and one query per
        // keystroke would run it eight times to answer the eighth.
        if (searchTimer) clearTimeout(searchTimer);
        searchTimer = setTimeout(runSearch, 180);
      }

      async function runSearch() {
        const query = searchQuery();
        if (!query) { refreshConversations(); return; }
        let found;
        try {
          const resp = await fetch("api/search?q=" + encodeURIComponent(query));
          if (!resp.ok) return;
          found = await resp.json();
        } catch (e) { return; }
        // Typing moved on while this was in flight; that answer is stale.
        if (searchQuery() !== query) return;
        paintSearch(found, query);
      }

      function paintSearch(found, query) {
        convoListEl.innerHTML = "";
        const chats = found.conversations || [], kept = found.records || [];
        if (!chats.length && !kept.length) {
          const none = document.createElement("p");
          none.className = "convo-empty";
          none.textContent = "Nothing matches " + JSON.stringify(query) + ".";
          convoListEl.appendChild(none);
          return;
        }
        if (chats.length) {
          convoListEl.appendChild(searchHeading(
            chats.length === 1 ? "1 conversation" : chats.length + " conversations"));
        }
        for (const convo of chats) {
          const row = document.createElement("div");
          row.className = "convo" + (convo.id === currentConvoId ? " active" : "");
          const open = document.createElement("button");
          open.className = "convo-open"; open.type = "button";
          const title = document.createElement("span");
          title.className = "convo-title"; title.textContent = convo.title;
          const line = document.createElement("span");
          line.className = "convo-meta hit";
          // textContent, never innerHTML: this is a slice of a stored message.
          line.textContent = convo.snippet || when(convo.updated_at);
          open.appendChild(title); open.appendChild(line);
          open.addEventListener("click", () => loadConversation(convo.id));
          row.appendChild(open);
          convoListEl.appendChild(row);
        }
        if (kept.length) {
          convoListEl.appendChild(searchHeading(
            kept.length === 1 ? "1 record" : kept.length + " records"));
          for (const record of kept) {
            const row = document.createElement("button");
            row.className = "convo-open hitrecord"; row.type = "button";
            const name = document.createElement("span");
            name.className = "convo-title";
            name.textContent = record.routine_name;
            const vals = document.createElement("span");
            vals.className = "convo-meta hit";
            vals.textContent = Object.keys(record.fields)
              .map(k => k + ": " + record.fields[k]).join(" · ");
            row.appendChild(name); row.appendChild(vals);
            // The record's own pane can filter and export; this only has to
            // get you there.
            row.addEventListener("click", () => showPane("records"));
            convoListEl.appendChild(row);
          }
        }
      }

      function searchHeading(text) {
        const head = document.createElement("p");
        head.className = "searchhead";
        head.textContent = text;
        return head;
      }

      function clearSearch() {
        searchEl.value = "";
        searchClearEl.hidden = true;
        if (searchTimer) { clearTimeout(searchTimer); searchTimer = null; }
        refreshConversations();
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
          note.className = "convo-empty storage";
          note.textContent = s.messages + " messages · " + size + " on disk";
          convoListEl.appendChild(note);

          // Photos are nearly all of that number, and what is worth having a
          // year later is the reading that came off the photo rather than the
          // photo. Saying so where the number is, is the only place it means
          // anything.
          const days = s.photo_keep_days;
          if (typeof days !== "number") return;
          const policy = document.createElement("p");
          policy.className = "convo-empty storage";
          policy.textContent = days > 0
            ? "Photos are kept for " + (days === 1 ? "a day" : days + " days") +
              "; the text stays for good."
            : "Photos are kept for good (PHOTO_KEEP_DAYS=0).";
          convoListEl.appendChild(policy);
          if (days <= 0) return;
          const now = document.createElement("button");
          now.type = "button"; now.className = "chip storage-act";
          now.textContent = "Free up space now";
          now.title = "Drop stored photos older than the cutoff. Every word is kept.";
          now.addEventListener("click", () => forgetPhotosNow(now, days));
          convoListEl.appendChild(now);
        } catch (e) { /* a footnote is never worth an error */ }
      }

      async function forgetPhotosNow(btn, days) {
        if (!confirm("Drop stored photos older than " + days + " days?\n\n" +
                     "Every message, reply and record is kept — only the " +
                     "pictures go, and only from conversations older than that.")) return;
        btn.disabled = true; btn.textContent = "Working…";
        try {
          const resp = await fetch("api/photos/forget", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ days: days }) });
          const done = await resp.json();
          hintEl.textContent = done.messages
            ? "Freed " + Math.round((done.bytes || 0) / 1024) + " KB from " +
              done.messages + (done.messages === 1 ? " photo." : " photos.")
            : "Nothing was old enough to drop.";
        } catch (e) {
          hintEl.textContent = "Could not free space just now.";
        }
        refreshConversations();
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

      async function loadConversation(id) {
        let convo;
        try {
          const resp = await fetch("api/conversations/" + id);
          if (!resp.ok) return;
          convo = await resp.json();
        } catch (e) { return; }

        if (controller) controller.abort();
        stopSpeaking();
        stopWaiting();
        if (recordsOpen()) closeRecords();
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
            const row = addUser(msg.content, imgs);
            // The photo has passed the keep-for period but its own record of
            // when and where it was taken did not, because that is a different
            // column and it is the part still worth having. Say so rather than
            // leaving a message that reads as if nothing was ever attached.
            const col = row && row.querySelector(".col");
            if (!imgs.length && msg.image_meta && msg.image_meta.length && col) {
              col.appendChild(expiredPhotoNote(msg.image_meta.length));
            }
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
            if (view.speakBtn) view.speakBtn.hidden = false;
            }
            // Both panels are stored with the reply now. They were live stream
            // state only, so a thread opened on the other device had the answer
            // and no way to see how it got there — which is exactly when you
            // go looking.
            if (msg.thinking) {
              view.thinkBody.textContent = msg.thinking;
              view.think.hidden = false;
            }
            for (const entry of msg.steps || []) addStep(view, entry);
            if (msg.sources) showSources(view, msg.sources);
            messages.push({ role: "assistant", content: msg.content });
          }
        }
        if (convo.model && modelEl.querySelector('option[value="' + convo.model + '"]')) {
          modelEl.value = convo.model;
        }
        setChatTitle(convo.title);
        dismissDrawer();
        refreshConversations();
        scrollDown(true);
      }

      // Wide enough for a sidebar and the drawer is a column of the layout
      // rather than something laid over it: no dimming, and picking a
      // conversation does not make the list you picked it from disappear.
      const wideScreen = window.matchMedia("(min-width: 1024px)");
      function railed() { return wideScreen.matches; }

      function openDrawer(pane) {
        drawerEl.hidden = false; backdropEl.hidden = railed();
        showPane(pane === "routines" || pane === "records" ? pane : "chats");
      }
      function closeDrawer() { drawerEl.hidden = true; backdropEl.hidden = true; }
      // Only when it is in the way. On a rail it is furniture, not a dialog.
      function dismissDrawer() { if (!railed()) closeDrawer(); }

      wideScreen.addEventListener("change", (e) => {
        // Narrowing with the rail open would leave it covering the
        // conversation with nothing dimmed to say it is over the top.
        if (e.matches) backdropEl.hidden = true;
        else if (!drawerEl.hidden) closeDrawer();
      });

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
        paintEmpty();
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
        inputEl.value = typed ? routine.body + "\n\n" + typed : routine.body;
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
          if (rest.slice(0, 2) === "\n\n") rest = rest.slice(2);
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
        if (!(await send())) return;
        clearRoutine();
        // Read back out of messages[] rather than out of send(), which is
        // covered by the node tests slice by slice and is not worth widening
        // for this. The last turn is the assistant one that just committed.
        const last = messages[messages.length - 1];
        if (last && last.role === "assistant") {
          const bubbles = chatEl.querySelectorAll(".msg.assistant");
          const root = bubbles.length ? bubbles[bubbles.length - 1] : null;
          keepRecord(routine, last.content, root ? { root: root } : null);
        }
      }

      // ---- Records ----
      // What a routine run wrote down. The paragraph is the answer; this is the
      // thing you still want in a year — 68 miles, 3 h 08 min, on the 7th of
      // August. Kept as fields so it can be a table, a CSV, or a row in
      // whatever the owner already runs.

      let records = [], recordColumns = [];

      async function refreshRecords() {
        if (!historyOn) return;
        try {
          const resp = await fetch("api/records");
          if (!resp.ok) return;
          const data = await resp.json();
          records = Array.isArray(data.records) ? data.records : [];
          recordColumns = Array.isArray(data.columns) ? data.columns : [];
        } catch (e) { return; }
      }

      // Fired once the reply is complete, not from inside the stream: the
      // streaming path is delicate and a record is worth less than the answer
      // already on screen. A failure here is silent by design — the reply
      // stands on its own.
      // The EXIF of the most recent turn that carried any. Held on the message
      // itself, so it is still here after the reply — and absent entirely when
      // the photo-details toggle is off, which is the honest answer then.
      function lastPhotoMeta() {
        for (let i = messages.length - 1; i >= 0; i--) {
          const msg = messages[i];
          if (msg && msg.role === "user" && msg.image_meta && msg.image_meta.length) {
            return msg.image_meta;
          }
        }
        return null;
      }

      async function keepRecord(routine, answer, view) {
        if (!historyOn || !routine || !routine.record || !routine.record.length) return;
        if (!answer || !answer.trim()) return;
        try {
          const resp = await fetch("api/records", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              answer: answer, fields: routine.record,
              routine_id: routine.id, routine_name: routine.name,
              conversation_id: currentConvoId, model: modelEl.value || null,
              // What the camera recorded, straight from the file. A field
              // declared "= earliest photo taken" is filled from this and the
              // model is never asked for it — it has no labels on the pictures
              // to match a time to, which is why it got them wrong.
              photos: lastPhotoMeta() }) });
          if (!resp.ok) return;
          const data = await resp.json();
          if (data.record && view) showKept(view, data.record, routine.record);
        } catch (e) { /* the answer is what matters; this is the extra */ }
      }

      // A line under the reply, so a record being kept is visible at the moment
      // it happens rather than discovered later in a drawer.
      function showKept(view, record, order) {
        const line = document.createElement("div");
        line.className = "kept";
        // Ordered by what the routine declared, not by the object's keys:
        // Flask's jsonify sorts them, so the wire order is alphabetical and
        // "average speed" led a trip record that starts with the distance.
        const names = (order && order.length ? order : Object.keys(record.fields));
        const pairs = names.filter(n => record.fields[n])
                           .map(n => [n, record.fields[n]]);
        line.textContent = "🗒 Kept: " + pairs.map(p => p[0] + " " + p[1]).join(" · ");
        line.title = "Saved to Records. Tap to open them.";
        line.addEventListener("click", () => openDrawer("records"));
        view.root.appendChild(line);
      }

      // Which routine's records are on screen. "" is all of them, which is also
      // what makes the table widest: the columns are the union across every
      // routine, so three routines with four fields each is a twelve-column
      // table in a drawer. Narrowing to one is the fix for that, and it is the
      // question you usually have anyway ("how far have I driven this month").
      let recordFilter = "";

      // Which of a routine's columns are worked out rather than read, and from
      // what. Taken from the routine's own declarations, which is where the
      // formula is written — so a blank hourly rate can say that it is blank
      // because no elapsed time was recorded, rather than looking like a bug.
      const derivedCache = {};
      function derivedIn(routineName) {
        if (derivedCache[routineName]) return derivedCache[routineName];
        const routine = routines.filter(r => r.name === routineName)[0];
        const out = {};
        for (const line of (routine && routine.record) || []) {
          const at = String(line).indexOf("=");
          if (at > 0) {
            out[String(line).slice(0, at).trim()] = String(line).slice(at + 1).trim();
          }
        }
        derivedCache[routineName] = out;
        return out;
      }

      function renderRecords() {
        // Routines can be edited while the table is open, and a stale formula
        // would explain a cell by a rule that no longer applies.
        for (const key of Object.keys(derivedCache)) delete derivedCache[key];
        recordListEl.innerHTML = "";
        if (!records.length) {
          const note = document.createElement("p");
          note.className = "convo-empty";
          note.textContent = recordFilter
            ? "No records from that routine yet."
            : "Nothing kept yet. Give a routine some field names and every run " +
              "of it writes a row here.";
          recordListEl.appendChild(note);
          if (recordFilter) recordListEl.insertBefore(recordFilterBar(), note);
          return;
        }
        recordListEl.appendChild(recordFilterBar());

        const shown = recordFilter
          ? records.filter(r => r.routine_name === recordFilter) : records;
        // Only the columns the shown records actually use, so filtering to one
        // routine genuinely narrows the table rather than leaving empty columns.
        const columns = [];
        for (const record of shown) {
          for (const name of Object.keys(record.fields)) {
            if (columns.indexOf(name) < 0) columns.push(name);
          }
        }

        const bar = document.createElement("div");
        bar.className = "exportbar";
        const query = recordFilter ? "?routine=" + encodeURIComponent(recordFilter) : "";
        for (const [label, href, download] of [
          ["⤓ CSV", "api/records.csv" + query, "records.csv"],
          ["⤓ JSON", "api/records" + query, null],
        ]) {
          const link = document.createElement("a");
          link.className = "chip"; link.href = href; link.textContent = label;
          if (download) link.setAttribute("download", download);
          else { link.target = "_blank"; link.rel = "noopener noreferrer"; }
          bar.appendChild(link);
        }
        const count = document.createElement("span");
        count.className = "voicebar-label";
        count.textContent = shown.length + (shown.length === 1 ? " record" : " records");
        bar.appendChild(count);
        recordListEl.appendChild(bar);

        const wrap = document.createElement("div");
        wrap.className = "tablewrap";
        const table = document.createElement("table");
        const head = document.createElement("tr");
        // The routine column is redundant once you have filtered to one.
        const labels = ["When"].concat(recordFilter ? [] : ["Routine"])
                               .concat(columns).concat([""]);
        for (const label of labels) {
          const th = document.createElement("th");
          th.textContent = label;
          head.appendChild(th);
        }
        const thead = document.createElement("thead");
        thead.appendChild(head);
        table.appendChild(thead);

        const body = document.createElement("tbody");
        for (const record of shown) {
          const row = document.createElement("tr");
          // data-label is what turns each cell into its own labelled line when
          // the stylesheet drops the table layout on a narrow screen.
          const cell = (label, text) => {
            const td = document.createElement("td");
            td.setAttribute("data-label", label);
            td.textContent = text;      // never innerHTML: this is stored text
            row.appendChild(td);
            return td;
          };
          cell("When", new Date(record.created_at * 1000)
            .toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" }));
          if (!recordFilter) cell("Routine", record.routine_name);
          for (const name of columns) {
            const td = cell(name, record.fields[name] || "");
            // Values are standardised on the way in, and the wording they
            // replaced is kept. Show it: a tidy-up you cannot see the before
            // of is one you have to take on faith, and the whole reason the
            // original is stored is so that you do not have to.
            // Editable, because the fields were pulled out of prose by a model
            // and a log you cannot correct is one you stop trusting.
            td.contentEditable = "true";
            td.className = "editable";
            // After className, not before: assigning it wholesale drops any
            // class added first, which is how the marker below silently never
            // appeared the first time this was written.
            const was = (record.raw || {})[name];
            const formula = derivedIn(record.routine_name)[name];
            if (was) {
              td.title = "As it was recorded: " + was;
              td.classList.add("tidied");
            } else if (formula) {
              td.classList.add("derived");
              td.title = (record.fields[name] ? "Worked out as " : "Nothing to work it "
                          + "out from — needs ") + formula;
            }
            td.addEventListener("blur", () => {
              const value = td.textContent.trim();
              if (value === (record.fields[name] || "")) return;
              record.fields[name] = value;
              editRecord(record.id, name, value);
            });
          }
          const act = cell("", "");
          const rm = document.createElement("button");
          rm.className = "convo-act"; rm.type = "button"; rm.textContent = "✕";
          rm.title = "Delete this record";
          rm.addEventListener("click", () => dropRecord(record));
          act.appendChild(rm);
          body.appendChild(row);
        }
        table.appendChild(body);
        wrap.appendChild(table);
        recordListEl.appendChild(wrap);
      }

      function recordFilterBar() {
        const bar = document.createElement("div");
        bar.className = "chips recordfilter";
        const names = [];
        for (const record of records) {
          if (names.indexOf(record.routine_name) < 0) names.push(record.routine_name);
        }
        // One routine is not a choice, so do not draw one.
        if (names.length < 2) return bar;
        for (const [label, value] of [["All", ""]].concat(names.map(n => [n, n]))) {
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "chip" + (value === recordFilter ? " active" : "");
          chip.textContent = label;
          chip.addEventListener("click", () => { recordFilter = value; renderRecords(); });
          bar.appendChild(chip);
        }
        return bar;
      }

      async function editRecord(id, name, value) {
        const fields = {};
        fields[name] = value;
        try {
          await fetch("api/records/" + id, {
            method: "PATCH", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fields: fields }) });
        } catch (e) { /* the cell already shows what you typed */ }
      }

      async function dropRecord(record) {
        if (!window.confirm("Delete this record?")) return;
        try {
          const resp = await fetch("api/records/" + record.id, { method: "DELETE" });
          if (!resp.ok) return;
        } catch (e) { return; }
        await refreshRecords();
        renderRecords();
      }

      // ---- Routines: the drawer ----

      // The drawer holds the two things you pick from. Records is the third
      // tab but not a third pane: choosing it leaves the drawer and opens the
      // log over the whole window, which is where a table with a dozen columns
      // and its own toolbar actually belongs.
      async function showPane(which) {
        if (which === "records") { await openRecords(); return; }
        convoPaneEl.hidden = which !== "chats";
        routinePaneEl.hidden = which !== "routines";
        tabChatsEl.classList.toggle("active", which === "chats");
        tabRoutinesEl.classList.toggle("active", which === "routines");
        if (which === "routines") { closeRoutineEditor(); renderRoutineList(); }
        else refreshConversations();
      }

      async function openRecords() {
        // The drawer has done its job. On a rail it stays, so a conversation
        // is still one click away; on a phone it would only be in the way.
        dismissDrawer();
        showPane("chats");
        document.body.classList.add("records");
        recordPaneEl.hidden = false;
        backBtn.hidden = false;
        setChatTitle("Records");
        await refreshRecords();
        renderRecords();
      }

      function closeRecords() {
        document.body.classList.remove("records");
        recordPaneEl.hidden = true;
        backBtn.hidden = true;
        setChatTitle(currentTitle);
        inputEl.focus();
      }

      function recordsOpen() { return document.body.classList.contains("records"); }

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
        rRecordEl.value = routine && routine.record ? routine.record.join("\n") : "";
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
                          web: tri(rWebEl.value), photo_meta: tri(rMetaEl.value),
                          record: rRecordEl.value.split("\n")
                            .map(f => f.trim()).filter(Boolean) };
        // Cleared before each attempt, not after a failed one. Fixing what a
        // warning complained about and saving again left the old text sitting
        // there, so a routine that was now fine still read as broken.
        routineWarnEl.textContent = "";
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
          // Saved either way, and said here rather than discovered later. A
          // formula naming a field that does not exist is an error nowhere: it
          // computes to nothing every run, and an empty column looks exactly
          // like a run with no data. One typo, found in a month of records.
          if (saved.problems && saved.problems.length) {
            routineWarnEl.textContent = "Saved. " + saved.problems.join(" ");
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
      // ---- Which microphone ----
      // The browser's default is picked by the operating system and is
      // regularly not the one you are talking into. Labels are only exposed
      // once microphone permission has been granted, so before the first
      // recording this can only offer "Default" — and it refills itself
      // afterwards, and whenever a device is plugged in or pulled out.
      const micDeviceEl = document.getElementById("micDevice");
      const micLevelEl = document.getElementById("micLevel");
      const micLevelBarEl = document.getElementById("micLevelBar");
      const micLevelPeakEl = document.getElementById("micLevelPeak");
      let micLabel = "";        // what we actually ended up recording from

      async function loadMicDevices() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
          micDeviceEl.hidden = true;
          return;
        }
        let devices = [];
        try { devices = await navigator.mediaDevices.enumerateDevices(); }
        catch (e) { micDeviceEl.hidden = true; return; }
        const mics = devices.filter(d => d.kind === "audioinput");
        // One nameless entry is what you get before permission: offering a
        // choice of one blank is worse than offering none.
        if (mics.length < 2 && !mics.some(d => d.label)) { micDeviceEl.hidden = true; return; }
        const want = micDeviceEl.value || remembered("mic");
        micDeviceEl.innerHTML = "";
        for (const [value, label] of [["", "Default microphone"]].concat(
                 mics.map((d, i) => [d.deviceId, d.label || ("Microphone " + (i + 1))]))) {
          const opt = document.createElement("option");
          opt.value = value;
          opt.textContent = label;   // a device name is text from the OS
          micDeviceEl.appendChild(opt);
        }
        // Only restore a device that is still plugged in, or the select shows
        // a name while recording from something else entirely.
        if (want && mics.some(d => d.deviceId === want)) micDeviceEl.value = want;
        micDeviceEl.hidden = false;
      }

      micDeviceEl.addEventListener("change", async () => {
        remember("mic", micDeviceEl.value);
        // Take effect now rather than at the next tap: you change this
        // *because* the last recording came from the wrong place.
        if (recording) { await stopMic(); await startMic(); }
      });

      let recording = false, mediaStream = null, audioCtx = null, srcNode = null, procNode = null, buffers = [];
      let micRate = 16000, bufferedSamples = 0, micStarting = false, hpState = null;
      // ~10 minutes at 16 kHz; a 16-bit mono WAV of that is ~19 MB, inside the
      // server's 25 MB body cap with room for the header.
      const MAX_SAMPLES = 16000 * 600;

      // Most of a room is below the speech.
      //
      // A fan, a fridge, traffic through a window, mains hum, a hand resting on
      // the desk: nearly all of a "quiet" room's energy sits under 100 Hz, and
      // none of it is speech. It does nothing for the recogniser, and it was
      // doing real damage on the way in, because the level the silence detector
      // measures is broadband — so the rumble set the threshold that speech
      // then had to clear. Measured, a fan and 50 Hz hum put the floor at
      // 0.017 with 89% of that energy below 130 Hz; the threshold that follows
      // from it is 0.050, and speech in the same room reached 0.023. Nothing
      // was ever detected, and nothing was ever sent.
      //
      // Two cascaded Butterworth sections, 4th order at 100 Hz. Measured: -24 dB
      // at 50 Hz, -18 dB at 60 Hz, -42 dB at 30 Hz, and flat to within 0.2 dB
      // from 150 Hz up, so a low male fundamental keeps its harmonics — which
      // is where the formants the recogniser works from actually are.
      const HP_HZ = 100;

      function makeHighPass(rate) {
        const sections = [];
        // The two Q values that make a 4th-order Butterworth out of a pair of
        // biquads. Anything else here is a different filter with a bump in it.
        for (const q of [0.54119610, 1.30656296]) {
          const w = 2 * Math.PI * HP_HZ / rate, cs = Math.cos(w);
          const alpha = Math.sin(w) / (2 * q);
          const a0 = 1 + alpha;
          sections.push({
            b0: ((1 + cs) / 2) / a0, b1: (-(1 + cs)) / a0, b2: ((1 + cs) / 2) / a0,
            a1: (-2 * cs) / a0, a2: (1 - alpha) / a0,
            x1: 0, x2: 0, y1: 0, y2: 0,
          });
        }
        return sections;
      }

      // Carries its state between buffers on purpose: a filter restarted every
      // 4096 samples rings at every boundary, which is a click at 4 Hz through
      // the whole recording.
      function highPass(sections, samples) {
        const out = new Float32Array(samples.length);
        for (let i = 0; i < samples.length; i++) {
          let v = samples[i];
          for (const s of sections) {
            const y = s.b0 * v + s.b1 * s.x1 + s.b2 * s.x2 - s.a1 * s.y1 - s.a2 * s.y2;
            s.x2 = s.x1; s.x1 = v; s.y2 = s.y1; s.y1 = y; v = y;
          }
          out[i] = v;
        }
        return out;
      }

      // Silence detection, so a pause ends an utterance and the mic can stay on
      // for a whole conversation instead of being tapped once per sentence.
      const VAD_SILENCE_MS = 900;      // pause length that closes an utterance
      const VAD_MIN_SPEECH_MS = 300;   // shorter than this is a noise blip
      const VAD_IDLE_STOP_MS = 60000;  // close a mic that was left on by accident
      let vadFloor = 0, vadSpeechMs = 0, vadSilenceMs = 0, vadIdleMs = 0, vadHasSpeech = false;
      // Where in the buffer the speech actually was, so what gets uploaded is
      // the sentence rather than the sentence plus every second of room that
      // happened to precede it.
      let speechFrom = -1, speechTo = 0;

      // The loudest sample of the whole recording, so clipping that happened
      // once is still visible afterwards rather than only for one frame.
      let micPeak = 0;

      function showLevel(chunk) {
        let peak = 0, sum = 0;
        for (let i = 0; i < chunk.length; i++) {
          const v = Math.abs(chunk[i]);
          if (v > peak) peak = v;
          sum += chunk[i] * chunk[i];
        }
        if (peak > micPeak) micPeak = peak;
        const rms = Math.sqrt(sum / chunk.length);
        // A log scale, because speech at a sensible level sits around 0.05
        // linear and a linear bar leaves it a sliver against a full-scale end.
        // -50 dB to 0 across the width.
        const shown = Math.max(0, Math.min(1, (20 * Math.log10(Math.max(rms, 1e-6)) + 50) / 50));
        micLevelBarEl.style.width = (shown * 100).toFixed(1) + "%";
        micLevelBarEl.classList.toggle("hot", peak > 0.98);
        const peakShown = Math.max(0, Math.min(1, (20 * Math.log10(Math.max(micPeak, 1e-6)) + 50) / 50));
        micLevelPeakEl.style.left = (peakShown * 100).toFixed(1) + "%";
      }

      function resetLevel() {
        micPeak = 0;
        micLevelBarEl.style.width = "0";
        micLevelBarEl.classList.remove("hot");
        micLevelPeakEl.style.left = "0";
      }

      // "No speech detected" with no reason sends people to the model picker
      // when the actual problem is the microphone. The level the recording
      // reached says which it was. Said in both the places that report it: the
      // recogniser coming back empty, and the detector never having heard
      // anything worth sending in the first place.
      function nothingHeardHint() {
        const why = micPeak < 0.02
          ? " The input barely registered — check the microphone picker."
          : (micPeak > 0.98 ? " The input was clipping — move back from the mic." : "");
        return "No speech detected." + why;
      }

      function vadReset() {
        vadFloor = 0; vadSpeechMs = 0; vadSilenceMs = 0; vadIdleMs = 0; vadHasSpeech = false;
        speechFrom = -1; speechTo = 0;
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

        // 2x the floor, not the 3x this had. Three demanded that speech be
        // 9.5 dB above the room, which is more than it usually manages next to
        // a fan; two is 6 dB. Swept against a fan, traffic, a television and a
        // quiet room, each with and without a door slamming, a chair scraping
        // and someone typing: 2x detects the rooms 3x missed entirely, and
        // still ignores everything 3x ignored. Below 2x a door slam starts
        // being sent, which is where the sweep stopped.
        //
        // A second, lower threshold to stay in speech once it has started —
        // the usual answer to a sentence arriving in fragments — was tried and
        // measured worse: it lets a door slam hold itself above the lower bar
        // long enough to count, and with a realistic voice there was no
        // fragmentation left for it to fix.
        if (rms > Math.max(vadFloor * 2, 0.006)) {
          // Remember where it was. This buffer has already been pushed, so it
          // is the last chunk.length samples of what is held.
          if (speechFrom < 0) speechFrom = bufferedSamples - chunk.length;
          speechTo = bufferedSamples;
          vadSpeechMs += ms; vadSilenceMs = 0; vadIdleMs = 0;
          if (vadSpeechMs >= VAD_MIN_SPEECH_MS) vadHasSpeech = true;
        } else {
          vadSilenceMs += ms; vadIdleMs += ms;
          if (vadHasSpeech && vadSilenceMs >= VAD_SILENCE_MS) flushUtterance();
          else if (!vadHasSpeech && vadIdleMs >= VAD_IDLE_STOP_MS) setTimeout(stopMic, 0);
        }
      }

      // What to keep either side of the speech. Enough lead-in that a soft
      // first consonant is not clipped off, enough tail that the last word
      // finishes — and no more, because everything past that is room.
      const SPEECH_LEAD_MS = 250, SPEECH_TAIL_MS = 300;

      // Send the sentence, not the sentence plus everything before it.
      //
      // A pause-delimited utterance used to be "everything captured since the
      // last one", which in a room the detector never triggered in could be a
      // minute of fan noise with three seconds of speech at the end. The
      // recogniser does not ignore the rest — it looks for words in it, and
      // finds some.
      function trimToSpeech(samples) {
        if (speechFrom < 0) return samples;
        const lead = Math.round(micRate * SPEECH_LEAD_MS / 1000);
        const tail = Math.round(micRate * SPEECH_TAIL_MS / 1000);
        const from = Math.max(0, speechFrom - lead);
        const to = Math.min(samples.length, speechTo + tail);
        return to > from ? samples.subarray(from, to) : samples;
      }

      // Cut what has been captured so far into its own utterance and send it off
      // to be transcribed, without tearing down the mic.
      function flushUtterance() {
        const captured = mergeBuffers(buffers);
        const speech = trimToSpeech(captured);
        buffers = []; bufferedSamples = 0;
        vadSpeechMs = 0; vadSilenceMs = 0; vadHasSpeech = false;
        speechFrom = -1; speechTo = 0;
        const wav = encodeWav(speech, micRate, 16000);
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
          const want = micDeviceEl.hidden ? "" : micDeviceEl.value;
          const constraints = {
            channelCount: 1,
            echoCancellation: !headsetEl.checked,
            noiseSuppression: true,
            autoGainControl: true,
          };
          // exact, so a chosen device is honoured or the attempt fails —
          // "ideal" silently falls back, which is how you end up recording
          // from the webcam while the picker says headset.
          if (want) constraints.deviceId = { exact: want };
          try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
          } catch (err) {
            if (!want) throw err;
            // Unplugged since it was chosen. Fall back rather than refusing to
            // record, but say so — otherwise the next bad transcription is a
            // mystery.
            delete constraints.deviceId;
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
            hintEl.textContent = "That microphone is not available — recording from the default one.";
            micDeviceEl.value = "";
            remember("mic", "");
          }
          // Labels are only exposed once permission has been granted, so this
          // is the first moment the picker can name anything.
          const track = mediaStream.getAudioTracks()[0];
          micLabel = (track && track.label) || "";
          loadMicDevices();
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
        hpState = makeHighPass(micRate);
        procNode.onaudioprocess = (e) => {
          const raw = e.inputBuffer.getChannelData(0);
          // The meter reads the microphone, not the filter: "is it hearing me"
          // and "is it clipping" are both questions about the input, and
          // clipping has already happened by the time we see it.
          showLevel(raw);
          // Everything downstream — the detector, the upload — gets the
          // filtered signal, which is the one the recogniser cares about.
          // highPass allocates, so this is also the copy that has to be taken
          // anyway: the browser reuses the buffer behind getChannelData.
          const chunk = highPass(hpState, raw);
          buffers.push(chunk);
          bufferedSamples += chunk.length;
          if (continuousEl.checked) vadStep(chunk);
          // In both modes, or a mic left running in a room the detector never
          // triggers in records until the upload 413s.
          if (bufferedSamples >= MAX_SAMPLES) {
            hintEl.textContent = "Reached the maximum recording length — transcribing.";
            setTimeout(stopMic, 0);
          }
        };
        srcNode.connect(procNode); procNode.connect(audioCtx.destination);
        recording = true; micBtn.classList.add("rec");
        resetLevel(); micLevelEl.hidden = false;
        const from = micLabel ? " from " + micLabel : "";
        hintEl.textContent = (continuousEl.checked
          ? "Listening" + from + "… pause to send an utterance. Tap the mic to stop."
          : "Listening" + from + "… tap the mic to stop.");
      }

      async function stopMic() {
        if (!recording) return;
        recording = false; micBtn.classList.remove("rec");
        micLevelEl.hidden = true;
        try { procNode.disconnect(); srcNode.disconnect(); } catch (e) {}
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        const captured = mergeBuffers(buffers);
        // Nothing was ever heard: don't post a minute of room noise and let the
        // recogniser find words in it. Only in continuous mode — a deliberate
        // tap on the mic is a deliberate request, and push-to-talk runs no
        // detector to have an opinion.
        const nothingHeard = continuousEl.checked && speechFrom < 0;
        const speech = trimToSpeech(captured);
        buffers = []; bufferedSamples = 0; vadReset(); hpState = null;
        try { await audioCtx.close(); } catch (e) {}
        if (nothingHeard) {
          hintEl.textContent = captured.length ? nothingHeardHint() : "";
          return;
        }
        const wav = encodeWav(speech, micRate, 16000);
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
          if (!t) {
            hintEl.textContent = recording ? "Listening…" : nothingHeardHint();
            return;
          }
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

      // Reduce the sample rate. Plain decimation folds everything above the
      // new Nyquist (8 kHz) back down into the speech band — a 12 kHz cymbal
      // lands at 4 kHz — which is why background noise wrecked recognition.
      //
      // The box average this used to do is a 3-tap filter at 48 kHz, and
      // measured it rejects only 13 dB — most of the fold-back survived and
      // landed on top of the speech. This is a windowed sinc, measured at
      // -65 dB across everything above the 8 kHz output Nyquist with the
      // speech band flat to within 0.01 dB. It costs about 1.5M multiply-adds
      // per second of audio, once per utterance, on a recording that is about
      // to be uploaded anyway.
      //
      // Usually a no-op either way: the capture context is asked for 16 kHz
      // and gets it wherever that is supported, so this runs only on the
      // browsers that refuse.
      const RESAMPLE_TAPS = 48;   // per side; 97 total

      function sinc(x) { return x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x); }

      // Built once per rate pair rather than per utterance: the coefficients
      // depend only on the ratio, and a phone re-deriving 33 of them for every
      // pause in a conversation is work for nothing.
      let resampleKernel = null;
      function kernelFor(ratio) {
        if (resampleKernel && resampleKernel.ratio === ratio) return resampleKernel;
        // Cutoff just under the output Nyquist, in input samples.
        // 0.90 of the output Nyquist: the last 10% buys a much deeper
        // stopband for a band (7.2-8 kHz) that carries almost no speech.
        const cutoff = 0.5 / ratio * 0.90;
        const taps = [];
        let total = 0;
        for (let k = -RESAMPLE_TAPS; k <= RESAMPLE_TAPS; k++) {
          // Blackman window: a gentler roll-off than Hamming for a much
          // deeper stopband, which is the whole point here.
          const w = 0.42 - 0.5 * Math.cos(2 * Math.PI * (k + RESAMPLE_TAPS) / (2 * RESAMPLE_TAPS))
                    + 0.08 * Math.cos(4 * Math.PI * (k + RESAMPLE_TAPS) / (2 * RESAMPLE_TAPS));
          const v = 2 * cutoff * sinc(2 * cutoff * k) * w;
          taps.push(v); total += v;
        }
        for (let i = 0; i < taps.length; i++) taps[i] /= total;   // unity at DC
        resampleKernel = { ratio: ratio, taps: taps };
        return resampleKernel;
      }

      function resample(samples, inRate, outRate) {
        if (inRate === outRate) return samples;
        const ratio = inRate / outRate;
        const outLen = Math.floor(samples.length / ratio);
        const out = new Float32Array(outLen);
        // Upsampling would need interpolation rather than this; the mic only
        // ever runs at or above 16 kHz, so guard and fall back rather than
        // quietly producing something wrong.
        if (ratio < 1) {
          for (let i = 0; i < outLen; i++) out[i] = samples[Math.floor(i * ratio)] || 0;
          return out;
        }
        const taps = kernelFor(ratio).taps;
        for (let i = 0; i < outLen; i++) {
          const centre = Math.round(i * ratio);
          let sum = 0;
          for (let k = -RESAMPLE_TAPS; k <= RESAMPLE_TAPS; k++) {
            const j = centre + k;
            // Zero outside the buffer: an utterance is seconds long, so the
            // ends this affects are a third of a millisecond each.
            if (j >= 0 && j < samples.length) sum += samples[j] * taps[k + RESAMPLE_TAPS];
          }
          out[i] = sum;
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

      // ---- Room to read while you type ----
      // Measured on a 390x844 phone: the footer is 293px idle, and 291px with
      // the on-screen keyboard up — which leaves 42px of conversation. You
      // cannot see the message you are replying to while replying to it.
      //
      // visualViewport is what actually shrinks when the keyboard opens;
      // window.innerHeight does not move on Android Chrome. Where it is absent
      // (older browsers, and every desktop, where no keyboard ever covers
      // anything) nothing here runs and the footer stays as it was.
      const KEYBOARD_ROOM = 520;   // below this, the keyboard is up

      function fitFooter() {
        const vv = window.visualViewport;
        if (!vv) return;
        const tight = vv.height < KEYBOARD_ROOM;
        // The composer and the thumbnails stay. Everything above them is a
        // setting you chose before you started typing, and it will still be
        // there when the keyboard goes away.
        document.body.classList.toggle("typing", tight);
      }

      if (window.visualViewport) {
        window.visualViewport.addEventListener("resize", fitFooter);
        // Scroll too: Android fires resize, iOS Safari sometimes only offsets.
        window.visualViewport.addEventListener("scroll", fitFooter);
        fitFooter();
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
      tabRecordsEl.addEventListener("click", () => showPane("records"));
      routineNewBtn.addEventListener("click", () => openRoutineEditor(null));
      rSaveBtn.addEventListener("click", saveRoutine);
      document.getElementById("rCancel").addEventListener("click", closeRoutineEditor);
      rDeleteBtn.addEventListener("click", deleteRoutine);
      rPhotosEl.addEventListener("change", routineWarnUpdate);
      rWebEl.addEventListener("change", routineWarnUpdate);
      document.getElementById("drawerClose").addEventListener("click", closeDrawer);
      backdropEl.addEventListener("click", () => { dismissDrawer(); hidePhotoDetails(); });
      document.getElementById("metaClose").addEventListener("click", hidePhotoDetails);
      document.getElementById("drawerNew").addEventListener("click", function () {
        newChat(); dismissDrawer();
      });

      sendBtn.addEventListener("click", trySend);
      stopBtn.addEventListener("click", stop);
      newBtn.addEventListener("click", newChat);
      micBtn.addEventListener("click", toggleMic);

      // Dictate here, then take the words to an app that has no dictation of
      // its own. On a phone the share sheet is the whole journey — tap, pick
      // the messaging app, done — and the clipboard is the desktop's version
      // of the same thing.
      copyOutBtn.addEventListener("click", async () => {
        const text = inputEl.value.trim();
        if (!text) return;
        // Cancelling the share sheet says nothing, because the user already
        // knows they cancelled. Every other outcome is reported: a copy that
        // silently did nothing is only discovered in the other app, with the
        // words gone.
        await shareOrCopy(text, inputEl, (how) => {
          if (how === "copied") hintEl.textContent = "Copied — paste it wherever you like.";
          else if (how === "shared") hintEl.textContent = "Shared.";
          else if (how === "manual") hintEl.textContent = "Selected — press Ctrl+C to copy.";
        });
      });

      // Its own button rather than clearing after a copy. Copying is not a
      // decision to throw the text away — you might copy it *and* send it —
      // and a composer that empties itself when you did not ask it to is the
      // kind of surprise that costs a whole dictated paragraph.
      clearOutBtn.addEventListener("click", () => {
        inputEl.value = "";
        autosize();
        hintEl.textContent = "";
        inputEl.focus();
      });

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

      // A sidebar you can reach without the mouse. Ctrl/⌘+K is what every
      // other app with a search box has trained people to press.
      document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
          e.preventDefault();
          openDrawer("chats");
          searchEl.focus(); searchEl.select();
          return;
        }
        // Escape closes whatever is over the top, innermost first. On a rail
        // there is nothing over the top, so it does nothing — which is right.
        if (e.key !== "Escape") return;
        if (!metaEl.hidden) { hidePhotoDetails(); return; }
        if (!drawerEl.hidden && !railed()) { closeDrawer(); return; }
        if (recordsOpen()) closeRecords();
      });

      backBtn.addEventListener("click", closeRecords);

      if (ttsReady()) {
        speakBarEl.hidden = false;
        rememberToggle(speakEl, "chatSpeak", false);
        ttsVoiceEl.addEventListener("change",
                                    () => remember("ttsVoice", ttsVoiceEl.value));
        loadTtsVoices();
        // Chrome populates the list asynchronously and fires this when it does;
        // without it the picker is empty on the first load of every session.
        if (synth.addEventListener) synth.addEventListener("voiceschanged", loadTtsVoices);
        // A tab left speaking in the background keeps speaking, which on a
        // phone means a voice coming out of an app you have switched away from.
        window.addEventListener("pagehide", stopSpeaking);
        document.addEventListener("visibilitychange", () => {
          if (document.visibilityState !== "visible") stopSpeaking();
        });
      }
      searchEl.addEventListener("input", onSearchInput);
      searchEl.addEventListener("keydown", (e) => { if (e.key === "Escape") clearSearch(); });
      searchClearEl.addEventListener("click", () => { clearSearch(); searchEl.focus(); });

      // ---- Theme ----
      // Auto is the default and the stylesheet does it on its own; this is for
      // the times the system is wrong — a phone that follows the clock while
      // you are reading in daylight, and the reverse at night. The <head>
      // script has already applied any saved choice, so this only has to flip.
      themeBtn.addEventListener("click", () => {
        const system = window.matchMedia("(prefers-color-scheme: dark)").matches;
        const now = document.documentElement.getAttribute("data-theme") ||
                    (system ? "dark" : "light");
        const next = now === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        try { localStorage.setItem("theme", next); } catch (e) {}
      });

      // On a wide screen the conversation list is furniture, and starting with
      // it already there is one fewer tap on every visit.
      if (railed()) { drawerEl.hidden = false; showPane("chats"); }
      paintEmpty();

      loadModels();
      checkVoice().then(loadMicDevices);
      // Plugging a headset in mid-conversation is exactly when you want to
      // change this, and a stale list would still be showing what was there
      // before.
      if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
        navigator.mediaDevices.addEventListener("devicechange", loadMicDevices);
      }
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
