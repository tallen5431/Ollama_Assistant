"""Tests for voice model resolution and WAV validation.

These deliberately avoid needing the ``vosk`` package: every path exercised
here fails or resolves before the lazy ``import vosk``.
"""

from __future__ import annotations

import io
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
