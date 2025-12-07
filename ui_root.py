#!/usr/bin/env python3
"""HTML UI for the CodeSmith Ollama Helper.

Kept in a separate module so app.py can stay focused on routing and
backend logic.
"""

from __future__ import annotations

ROOT_HTML = """<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\">
    <title>CodeSmith Ollama Helper</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <style>
      body {
        background:#020617;
        color:#e5e7eb;
        font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        margin:0;
        padding:1.5rem;
      }
      h1 {
        font-size:1.3rem;
        margin-bottom:0.75rem;
      }
      h2 {
        font-size:1.05rem;
        margin-top:1.5rem;
        margin-bottom:0.5rem;
      }
      code {
        background:#020617;
        border-radius:0.25rem;
        padding:0.05rem 0.2rem;
      }
      ul {
        padding-left:1.2rem;
      }
      textarea, input, button {
        font-family:inherit;
        font-size:0.95rem;
      }
      textarea {
        width:100%;
        min-height:6rem;
        background:#020617;
        border:1px solid #1f2937;
        color:#e5e7eb;
        border-radius:0.5rem;
        padding:0.5rem;
        margin-bottom:0.5rem;
        resize:vertical;
      }
      input {
        width:100%;
        background:#020617;
        border:1px solid #1f2937;
        color:#e5e7eb;
        border-radius:0.5rem;
        padding:0.45rem 0.6rem;
        margin-bottom:0.75rem;
      }
      button {
        background:#2563eb;
        border:none;
        color:white;
        padding:0.45rem 0.9rem;
        border-radius:999px;
        cursor:pointer;
        font-weight:500;
      }
      button:disabled {
        opacity:0.6;
        cursor:default;
      }
      pre {
        background:#020617;
        border:1px solid #1f2937;
        border-radius:0.5rem;
        padding:0.75rem;
        white-space:pre-wrap;
        word-wrap:break-word;
        max-height:40vh;
        overflow:auto;
      }
      small {
        color:#9ca3af;
      }
    </style>
  </head>
  <body>
    <h1>CodeSmith Ollama Helper</h1>

    <h2>Quick chat test</h2>
    <p><small>Uses <code>POST /api/chat</code> on this helper.</small></p>
    <label><small>Prompt</small></label>
    <textarea id=\"prompt\" placeholder=\"Ask the model something...\"></textarea>
    <label><small>Model (optional, blank = default)</small></label>
    <input id=\"model\" placeholder=\"qwen2.5-coder:7b\">
    <button id=\"send\">Send</button>
    <p><small><span id=\"status\"></span></small></p>
    <pre id=\"output\"></pre>

    <h2>Upload CodeSmith snapshot</h2>
    <p><small>Upload a snapshot <code>.json</code> file; the helper will save it under its uploads folder.</small></p>
    <input id=\"snapshotFile\" type=\"file\" accept=\".json\">
    <button id=\"uploadSnapshot\">Upload snapshot</button>
    <pre id=\"snapshotResult\"></pre>

    <h2>API endpoints</h2>
    <ul>
      <li><code>GET /api/health</code></li>
      <li><code>GET /api/models</code></li>
      <li><code>POST /api/chat</code> (raw chat proxy to Ollama)</li>
      <li><code>POST /api/code-assist</code> (CodeSmith-friendly endpoint)</li>
      <li><code>POST /api/patch/normalize</code> (normalize a model patch reply)</li>
      <li><code>POST /api/upload-snapshot</code> (upload a snapshot JSON into the uploads folder)</li>
    </ul>

    <script>
      const promptEl = document.getElementById("prompt");
      const modelEl  = document.getElementById("model");
      const sendBtn  = document.getElementById("send");
      const statusEl = document.getElementById("status");
      const outEl    = document.getElementById("output");

      const snapshotInput  = document.getElementById("snapshotFile");
      const uploadBtn      = document.getElementById("uploadSnapshot");
      const snapshotOutEl  = document.getElementById("snapshotResult");

      async function send() {
        const prompt = promptEl.value.trim();
        const model  = modelEl.value.trim();
        if (!prompt) return;
        sendBtn.disabled = true;
        statusEl.textContent = "Calling /api/chat...";
        outEl.textContent = "";
        try {
          const body = { prompt: prompt };
          if (model) body.model = model;
          const resp = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
          });
          const data = await resp.json();
          if (data.error) {
            outEl.textContent = "Error: " + data.error;
          } else {
            outEl.textContent = data.reply || JSON.stringify(data, null, 2);
          }
        } catch (err) {
          outEl.textContent = "Request failed: " + err;
        } finally {
          sendBtn.disabled = false;
          statusEl.textContent = "";
        }
      }

      async function uploadSnapshot() {
        const file = snapshotInput.files[0];
        if (!file) return;
        uploadBtn.disabled = true;
        statusEl.textContent = "Uploading snapshot...";
        snapshotOutEl.textContent = "";
        try {
          const formData = new FormData();
          formData.append("file", file);
          const resp = await fetch("/api/upload-snapshot", {
            method: "POST",
            body: formData
          });
          const data = await resp.json();
          if (data.error) {
            snapshotOutEl.textContent = "Error: " + data.error;
          } else {
            snapshotOutEl.textContent = JSON.stringify(data, null, 2);
          }
        } catch (err) {
          snapshotOutEl.textContent = "Upload failed: " + err;
        } finally {
          uploadBtn.disabled = false;
          statusEl.textContent = "";
        }
      }

      sendBtn.addEventListener("click", send);
      promptEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
          e.preventDefault();
          send();
        }
      });

      if (uploadBtn) {
        uploadBtn.addEventListener("click", uploadSnapshot);
      }
    </script>
  </body>
</html>
"""
