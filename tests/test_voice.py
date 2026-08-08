"""Tests for voice model resolution and WAV validation.

These deliberately avoid needing the ``vosk`` package: every path exercised
here fails or resolves before the lazy ``import vosk``.
"""

from __future__ import annotations

import io
import json
import wave

import pytest

import voice


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Point the module at a throwaway models directory."""
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr(voice, "_MODELS_DIR", d)
    return d


def make_model_dir(parent, name):
    """Create something ``_looks_like_model`` accepts."""
    p = parent / name
    (p / "conf").mkdir(parents=True)
    return p


class TestResolveDirRejectsTraversal:
    @pytest.mark.parametrize(
        "bad",
        [
            "../evil",
            "../../etc",
            "a/b",
            "a\\b",
            "..",
            ".",
            "/etc/passwd",
            "sub/../../escape",
        ],
    )
    def test_traversal_ids_are_rejected(self, bad, models_dir):
        with pytest.raises(ValueError, match="Unknown voice model"):
            voice._resolve_dir(bad, download=False)

    def test_traversal_is_rejected_even_when_the_target_exists(self, models_dir, tmp_path):
        # Without validation this resolved to a real directory outside models/.
        make_model_dir(tmp_path, "evil")
        with pytest.raises(ValueError, match="Unknown voice model"):
            voice._resolve_dir("../evil", download=False)


class TestResolveDir:
    def test_plain_directory_name_inside_models_dir(self, models_dir):
        make_model_dir(models_dir, "my-own-model")
        assert voice._resolve_dir("my-own-model", download=False) == models_dir / "my-own-model"

    def test_directory_without_model_layout_is_rejected(self, models_dir):
        (models_dir / "not-a-model").mkdir()
        with pytest.raises(ValueError, match="Unknown voice model"):
            voice._resolve_dir("not-a-model", download=False)

    def test_catalog_id_present_on_disk(self, models_dir):
        make_model_dir(models_dir, voice.CATALOG["es"]["dir"])
        assert voice._resolve_dir("es", download=False) == models_dir / voice.CATALOG["es"]["dir"]

    def test_catalog_id_absent_and_download_disallowed(self, models_dir):
        with pytest.raises(ValueError, match="not downloaded yet"):
            voice._resolve_dir("es", download=False)

    def test_custom_path_takes_precedence(self, models_dir, tmp_path, monkeypatch):
        custom = make_model_dir(tmp_path, "custom-model")
        monkeypatch.setenv("VOSK_MODEL_PATH", str(custom))
        assert voice._resolve_dir("custom", download=False) == custom
        assert voice._resolve_dir(None, download=False) == custom

    def test_custom_sentinel_falls_back_when_unconfigured(self, models_dir, monkeypatch):
        monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
        make_model_dir(models_dir, voice.CATALOG["en-us"]["dir"])
        assert voice._resolve_dir("custom", download=False).name == voice.CATALOG["en-us"]["dir"]


class TestDownloadModel:
    def test_unknown_id_is_rejected_before_any_network_call(self, models_dir):
        with pytest.raises(ValueError, match="Unknown voice model"):
            voice.download_model("../../etc")

    def test_non_catalog_id_is_rejected(self, models_dir):
        with pytest.raises(ValueError, match="Unknown voice model"):
            voice.download_model("klingon")


class TestListModels:
    def test_reports_catalog_and_download_state(self, models_dir):
        make_model_dir(models_dir, voice.CATALOG["fr"]["dir"])
        data = voice.list_models()

        assert {m["id"] for m in data["available"]} == {"fr"}
        by_id = {m["id"]: m for m in data["catalog"]}
        assert by_id["fr"]["downloaded"] is True
        assert by_id["de"]["downloaded"] is False
        assert by_id["fr"]["label"] == "French"

    def test_custom_model_is_listed_first(self, models_dir, tmp_path, monkeypatch):
        monkeypatch.setenv("VOSK_MODEL_PATH", str(make_model_dir(tmp_path, "mine")))
        data = voice.list_models()
        assert data["available"][0]["id"] == "custom"
        assert data["default"] == "custom"

    def test_empty_when_nothing_downloaded(self, models_dir, monkeypatch):
        monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
        data = voice.list_models()
        assert data["available"] == []
        assert all(m["downloaded"] is False for m in data["catalog"])


def make_wav(channels=1, sampwidth=2, framerate=16000, frames=b"\x00\x00" * 100):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(frames)
    return buf.getvalue()


class TestTranscribeValidation:
    def test_non_wav_bytes_are_rejected(self):
        with pytest.raises(ValueError, match="Could not read audio as WAV"):
            voice.transcribe(b"this is not a wav file")

    def test_empty_audio_is_rejected(self):
        with pytest.raises(ValueError, match="Could not read audio as WAV"):
            voice.transcribe(b"")

    def test_stereo_is_rejected(self):
        with pytest.raises(ValueError, match="16-bit mono"):
            voice.transcribe(make_wav(channels=2))

    def test_8bit_is_rejected(self):
        with pytest.raises(ValueError, match="16-bit mono"):
            voice.transcribe(make_wav(sampwidth=1, frames=b"\x00" * 100))


@pytest.fixture
def fake_vosk(monkeypatch):
    """vosk is optional and not installed here, and transcribe() imports it
    before resolving the model — so without this the test never reaches the
    line it is about."""
    import sys
    import types
    module = types.ModuleType("vosk")
    module.Model = lambda path: object()
    module.KaldiRecognizer = lambda model, rate: types.SimpleNamespace(
        AcceptWaveform=lambda data: False,
        Result=lambda: '{"text": ""}',
        FinalResult=lambda: '{"text": ""}',
    )
    monkeypatch.setitem(sys.modules, "vosk", module)
    return module


def _silent_wav(frames=160):
    """A real 16-bit mono PCM WAV — wave.open rejects anything less."""
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


class TestTranscribeNeverDownloads:
    """The model id comes from the client, on a CORS-simple request.

    Resolving it with downloads enabled meant a request naming an unfetched
    model made the box pull 1.8 GB before reading an audio frame, holding a
    server thread for the whole download — and concurrent requests each pulled
    their own copy. /api/voice/download is the deliberate way to fetch one.
    """

    def test_an_unfetched_model_is_refused_rather_than_downloaded(self, models_dir, monkeypatch):
        downloads = []
        monkeypatch.setattr(voice, "_download", lambda mid: downloads.append(mid))
        monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)

        with pytest.raises(ValueError, match="not downloaded"):
            voice._get_model("es", download=False)
        assert downloads == [], "a client request triggered a 1.8 GB download"

    def test_the_explicit_download_path_still_downloads(self, models_dir, monkeypatch):
        downloads = []
        monkeypatch.setattr(voice, "_download",
                            lambda mid: downloads.append(mid) or models_dir / "x")
        monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)

        voice._resolve_dir("es", download=True)
        assert downloads == ["es"]

    def test_transcribe_asks_for_no_download(self, models_dir, monkeypatch, fake_vosk):
        """The whole point: the client-facing path never fetches.

        This has to reach _get_model, which means real audio — a stub WAV is
        rejected by wave.open before transcribe() gets that far, which is how
        the first version of this test passed with the fix fully reverted.
        """
        seen = {}
        monkeypatch.setattr(voice, "_get_model",
                            lambda mid, download=True: seen.update(download=download))
        monkeypatch.setattr(voice, "_download",
                            lambda mid: pytest.fail("transcribe must never download"))
        voice.transcribe(_silent_wav(), "es")
        assert seen.get("download") is False, (
            "transcribe resolved the model with downloads enabled — a client-"
            "supplied id could make the box pull 1.8 GB"
        )

    def test_a_client_supplied_id_cannot_trigger_a_download_end_to_end(
            self, models_dir, monkeypatch, fake_vosk):
        """No stubbing of _get_model: the real resolution path must refuse."""
        monkeypatch.setattr(voice, "_download",
                            lambda mid: pytest.fail("a request triggered a 1.8 GB download"))
        monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
        with pytest.raises(ValueError, match="not downloaded"):
            voice.transcribe(_silent_wav(), "es")


# --------------------------------------------------------------------------
# Downloading a model
# --------------------------------------------------------------------------

class TestDownloadProgress:
    """The large English model is 1.8 GB — ten minutes on a domestic line, and
    ten minutes of a silent request is indistinguishable from a broken one.
    """

    def _serve(self, monkeypatch, tmp_path, payload=b"x" * (900 * 1024)):
        """A model archive, served locally, that _download can really fetch."""
        import http.server, socketserver, threading, zipfile, io

        folder = "vosk-model-en-us-0.22"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name in ("am/final.mdl", "conf/mfcc.conf", "conf/model.conf",
                         "graph/HCLr.fst", "graph/Gr.fst",
                         "graph/phones/word_boundary.int", "ivector/final.ie"):
                zf.writestr(f"{folder}/{name}", payload if "final.mdl" in name else b"x")
        body = buf.getvalue()

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = socketserver.TCPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        monkeypatch.setattr(voice, "_MODEL_HOST",
                            f"http://127.0.0.1:{srv.server_address[1]}/")
        monkeypatch.setattr(voice, "_MODELS_DIR", tmp_path / "models")
        return srv, len(body)

    def test_it_reports_as_it_goes(self, monkeypatch, tmp_path):
        srv, total = self._serve(monkeypatch, tmp_path)
        seen = []
        path = voice._download("en-us-large", lambda done, size: seen.append((done, size)))
        srv.shutdown()
        assert path.is_dir() and (path / "am" / "final.mdl").exists()
        assert len(seen) >= 3, f"only {len(seen)} progress callbacks for {total} bytes"
        assert seen[0][0] == 0 and seen[-1][0] == total
        assert all(size == total for _, size in seen), "the total should be known upfront"
        assert [d for d, _ in seen] == sorted(d for d, _ in seen), "and never go backwards"

    def test_without_a_callback_it_still_works(self, monkeypatch, tmp_path):
        """The plain path is what _resolve_dir uses on first transcription."""
        srv, _ = self._serve(monkeypatch, tmp_path)
        assert voice._download("en-us-large").is_dir()
        srv.shutdown()

    def test_a_model_already_there_says_so(self, monkeypatch, tmp_path):
        """Otherwise a second click looks like a download that finished
        suspiciously fast."""
        srv, _ = self._serve(monkeypatch, tmp_path)
        voice.download_model("en-us-large")
        again = voice.download_model("en-us-large")
        srv.shutdown()
        assert again["already"] is True

    def test_the_route_streams_it(self, monkeypatch, tmp_path):
        import importlib
        import app as app_module

        srv, total = self._serve(monkeypatch, tmp_path)
        for key in ("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        app = importlib.reload(app_module)
        monkeypatch.setattr(app.voice, "voice_available", lambda: True)
        monkeypatch.setattr(app.voice, "_MODEL_HOST", voice._MODEL_HOST)
        monkeypatch.setattr(app.voice, "_MODELS_DIR", tmp_path / "models")
        resp = app.app.test_client().post("/api/voice/download", json={"id": "en-us-large"})
        srv.shutdown()
        assert resp.status_code == 200
        assert resp.mimetype == "application/x-ndjson"
        lines = [json.loads(x) for x in resp.get_data(as_text=True).splitlines() if x.strip()]
        percents = [x for x in lines if "percent" in x]
        assert len(percents) >= 2, lines
        assert percents[-1]["percent"] == 100
        assert lines[-1]["dir"] == "vosk-model-en-us-0.22"

    def test_a_failure_arrives_as_a_line_not_a_500(self, monkeypatch, tmp_path):
        """The stream has already started, so there is no status code left."""
        import importlib
        import app as app_module

        for key in ("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        app = importlib.reload(app_module)
        monkeypatch.setattr(app.voice, "voice_available", lambda: True)
        monkeypatch.setattr(app.voice, "_MODEL_HOST", "http://127.0.0.1:1/")
        monkeypatch.setattr(app.voice, "_MODELS_DIR", tmp_path / "models")
        resp = app.app.test_client().post("/api/voice/download", json={"id": "en-us-large"})
        assert resp.status_code == 200
        lines = [json.loads(x) for x in resp.get_data(as_text=True).splitlines() if x.strip()]
        assert "error" in lines[-1]

    def test_the_percent_lines_are_not_a_download_of_their_own(self, monkeypatch, tmp_path):
        """1.8 GB in 256KB chunks is 7000 callbacks; one line each is silly.

        A small chunk size rather than a huge payload: 225 reads of a 900KB
        file makes the same point as 7000 reads of 1.8 GB, in a test that
        finishes.
        """
        srv, _ = self._serve(monkeypatch, tmp_path)
        monkeypatch.setattr(voice, "_CHUNK", 4096)
        import importlib
        import app as app_module

        for key in ("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        app = importlib.reload(app_module)
        monkeypatch.setattr(app.voice, "voice_available", lambda: True)
        monkeypatch.setattr(app.voice, "_MODEL_HOST", voice._MODEL_HOST)
        monkeypatch.setattr(app.voice, "_MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(app.voice, "_CHUNK", 4096)
        resp = app.app.test_client().post("/api/voice/download", json={"id": "en-us-large"})
        srv.shutdown()
        lines = resp.get_data(as_text=True).splitlines()
        assert len(lines) <= 105, f"{len(lines)} lines for one download"
