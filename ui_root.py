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
      textarea, input, button, select {
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
      input, select {
        width:100%;
        background:#020617;
        border:1px solid #1f2937;
        color:#e5e7eb;
        border-radius:0.5rem;
        padding:0.45rem 0.6rem;
        margin-bottom:0.75rem;
      }
      select {
        cursor:pointer;
      }
      button {
        background:#2563eb;
        border:none;
        color:white;
        padding:0.45rem 0.9rem;
        border-radius:999px;
        cursor:pointer;
        font-weight:500;
        margin-right:0.5rem;
      }
      button:disabled {
        opacity:0.6;
        cursor:default;
      }
      button.secondary {
        background:#4b5563;
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
      .snapshot-info {
        background:#1e293b;
        border:1px solid #334155;
        border-radius:0.5rem;
        padding:0.75rem;
        margin-bottom:0.75rem;
        font-size:0.9rem;
      }
      .snapshot-info.loaded {
        border-color:#10b981;
      }
      .snapshot-info .label {
        color:#9ca3af;
        font-size:0.85rem;
      }
      .snapshot-info .value {
        color:#e5e7eb;
        font-weight:500;
      }
      .flex {
        display:flex;
        gap:0.5rem;
        align-items:center;
      }
    </style>
  </head>
  <body>
    <h1>CodeSmith Ollama Helper</h1>

    <h2>Snapshot Context</h2>
    <p><small>Load a snapshot to provide code context to the model</small></p>
    <div class=\"flex\">
      <select id=\"snapshotSelect\" style=\"flex:1\">
        <option value=\"\">-- Select a snapshot --</option>
      </select>
      <button id=\"loadSnapshot\" class=\"secondary\">Load</button>
      <button id=\"refreshSnapshots\" class=\"secondary\">Refresh</button>
    </div>
    <div id=\"snapshotInfo\" class=\"snapshot-info\" style=\"display:none\">
      <div class=\"label\">Loaded Snapshot:</div>
      <div class=\"value\" id=\"loadedSnapshotName\">None</div>
      <div class=\"label\" style=\"margin-top:0.5rem\">Files: <span class=\"value\" id=\"loadedFileCount\">0</span> | Size: <span class=\"value\" id=\"loadedTotalSize\">0</span></div>
    </div>

    <h2>Quick chat test</h2>
    <p><small>Uses <code>POST /api/chat</code> on this helper. Includes loaded snapshot as context.</small></p>
    <label><small>Prompt</small></label>
    <textarea id=\"prompt\" placeholder=\"Ask the model something...\"></textarea>
    <label><small>Model (optional, blank = default)</small></label>
    <input id=\"model\" placeholder=\"qwen2.5-coder:7b\">
    <div class=\"flex\">
      <button id=\"send\">Send</button>
      <button id=\"clearSnapshot\" class=\"secondary\" style=\"display:none\">Clear Snapshot</button>
    </div>
    <p><small><span id=\"status\"></span></small></p>
    <pre id=\"output\"></pre>

    <h2>Upload CodeSmith snapshot</h2>
    <p><small>Upload a snapshot <code>.json</code> file. After upload, it will appear in the dropdown above.</small></p>
    <div class=\"flex\">
      <input id=\"snapshotFile\" type=\"file\" accept=\".json\" style=\"flex:1\">
      <button id=\"uploadSnapshot\">Upload</button>
    </div>
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

      const snapshotSelectEl = document.getElementById("snapshotSelect");
      const loadSnapshotBtn = document.getElementById("loadSnapshot");
      const refreshSnapshotsBtn = document.getElementById("refreshSnapshots");
      const clearSnapshotBtn = document.getElementById("clearSnapshot");
      const snapshotInfoEl = document.getElementById("snapshotInfo");
      const loadedSnapshotNameEl = document.getElementById("loadedSnapshotName");
      const loadedFileCountEl = document.getElementById("loadedFileCount");
      const loadedTotalSizeEl = document.getElementById("loadedTotalSize");

      const snapshotInput  = document.getElementById("snapshotFile");
      const uploadBtn      = document.getElementById("uploadSnapshot");
      const snapshotOutEl  = document.getElementById("snapshotResult");

      let currentSnapshot = null;
      let currentSnapshotFiles = [];

      async function loadSnapshots() {
        try {
          const resp = await fetch("/api/snapshots");
          const data = await resp.json();
          snapshotSelectEl.innerHTML = '<option value="">-- Select a snapshot --</option>';
          if (data.snapshots && data.snapshots.length > 0) {
            data.snapshots.forEach(snap => {
              const opt = document.createElement("option");
              opt.value = snap.filename;
              const fileCount = snap.meta?.file_count || 0;
              const size = (snap.size / 1024).toFixed(1);
              opt.textContent = `${snap.filename} (${fileCount} files, ${size}KB)`;
              snapshotSelectEl.appendChild(opt);
            });
          }
        } catch (err) {
          console.error("Failed to load snapshots:", err);
        }
      }

      async function loadSnapshot() {
        const filename = snapshotSelectEl.value;
        if (!filename) return;

        statusEl.textContent = "Loading snapshot...";
        try {
          const resp = await fetch("/api/snapshot/files", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              filename: filename,
              max_total_chars: 50000,
              max_file_chars: 10000,
              include_source: true
            })
          });
          const data = await resp.json();
          if (data.error) {
            alert("Error loading snapshot: " + data.error);
            return;
          }

          currentSnapshot = data.snapshot;
          currentSnapshotFiles = data.files || [];

          loadedSnapshotNameEl.textContent = filename;
          loadedFileCountEl.textContent = currentSnapshotFiles.length;
          loadedTotalSizeEl.textContent = ((currentSnapshot.total_size || 0) / 1024).toFixed(1) + "KB";

          snapshotInfoEl.style.display = "block";
          snapshotInfoEl.classList.add("loaded");
          clearSnapshotBtn.style.display = "inline-block";

          statusEl.textContent = `✓ Loaded ${currentSnapshotFiles.length} files`;
          setTimeout(() => statusEl.textContent = "", 3000);
        } catch (err) {
          alert("Failed to load snapshot: " + err);
        }
      }

      function clearSnapshot() {
        currentSnapshot = null;
        currentSnapshotFiles = [];
        snapshotInfoEl.style.display = "none";
        snapshotInfoEl.classList.remove("loaded");
        clearSnapshotBtn.style.display = "none";
        snapshotSelectEl.value = "";
        statusEl.textContent = "Snapshot cleared";
        setTimeout(() => statusEl.textContent = "", 2000);
      }

      async function send() {
        const prompt = promptEl.value.trim();
        const model  = modelEl.value.trim();
        if (!prompt) return;
        sendBtn.disabled = true;
        statusEl.textContent = "Calling /api/chat...";
        outEl.textContent = "";

        try {
          // Build context from loaded snapshot
          let fullPrompt = prompt;
          if (currentSnapshotFiles.length > 0) {
            const contextParts = ["Here is the codebase context:\n"];
            currentSnapshotFiles.forEach(file => {
              if (file.snippet) {
                contextParts.push(`\n--- ${file.path} ---`);
                contextParts.push(file.snippet);
                if (file.snippet_truncated) {
                  contextParts.push("\n[... truncated ...]");
                }
              }
            });
            contextParts.push(`\n\nUser question: ${prompt}`);
            fullPrompt = contextParts.join("\n");
          }

          const body = { prompt: fullPrompt };
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
            snapshotOutEl.style.color = "#ef4444";
          } else {
            snapshotOutEl.textContent = "✓ Upload successful!\n\n" + JSON.stringify(data, null, 2);
            snapshotOutEl.style.color = "#10b981";
            // Clear file input
            snapshotInput.value = "";
            // Auto-refresh snapshots list and show success message
            statusEl.textContent = "✓ Snapshot uploaded! Refreshing list...";
            await loadSnapshots();
            setTimeout(() => {
              statusEl.textContent = "✓ Snapshot ready to load";
              setTimeout(() => statusEl.textContent = "", 3000);
            }, 500);
          }
        } catch (err) {
          snapshotOutEl.textContent = "Upload failed: " + err;
          snapshotOutEl.style.color = "#ef4444";
        } finally {
          uploadBtn.disabled = false;
        }
      }

      sendBtn.addEventListener("click", send);
      promptEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
          e.preventDefault();
          send();
        }
      });

      loadSnapshotBtn.addEventListener("click", loadSnapshot);
      refreshSnapshotsBtn.addEventListener("click", loadSnapshots);
      clearSnapshotBtn.addEventListener("click", clearSnapshot);

      if (uploadBtn) {
        uploadBtn.addEventListener("click", uploadSnapshot);
      }

      // Load snapshots on page load
      loadSnapshots();
    </script>
  </body>
</html>
"""
