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
