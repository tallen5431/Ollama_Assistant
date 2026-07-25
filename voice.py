#!/usr/bin/env python3
"""Offline speech-to-text with Vosk.

Kept optional and lazy: if the ``vosk`` package isn't installed, voice input is
simply disabled and the rest of the app runs unchanged. The (small English)
model is loaded on first use and, if absent, downloaded once into ``models/``.

Environment:
  VOSK_MODEL_PATH   Path to an already-unpacked Vosk model directory. If set and
                    valid, it's used as-is (no download).
  VOSK_MODELS_DIR   Where to store/download models. Default ``./models``.
  VOSK_MODEL_URL    Override the model to download.
"""

from __future__ import annotations

import io
import json
import os
import wave
import zipfile
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

from config import logger

_BASE_DIR = Path(__file__).parent.resolve()
_MODELS_DIR = Path(os.getenv("VOSK_MODELS_DIR", _BASE_DIR / "models")).resolve()
_DEFAULT_MODEL_URL = os.getenv(
    "VOSK_MODEL_URL",
    "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
)
_DEFAULT_MODEL_DIR = "vosk-model-small-en-us-0.15"

_model = None  # cached vosk.Model once loaded


def vosk_installed() -> bool:
    """True when the ``vosk`` package can be imported."""
    try:
        import vosk  # noqa: F401
        return True
    except Exception:
        return False


def voice_available() -> bool:
    """Whether voice input can be offered (the library is present).

    The model itself may still need a one-time download on first transcription.
    """
    return vosk_installed()


def _configured_model_path() -> Optional[Path]:
    """Return an existing model dir from env or the default location, else None."""
    env = os.getenv("VOSK_MODEL_PATH")
    if env and Path(env).is_dir():
        return Path(env)
    default = _MODELS_DIR / _DEFAULT_MODEL_DIR
    return default if default.is_dir() else None


def _ensure_model_path() -> Path:
    """Return a model directory, downloading the small default one if needed."""
    existing = _configured_model_path()
    if existing:
        return existing

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _MODELS_DIR / "_model.zip"
    logger.info("Downloading Vosk model from %s (one-time) ...", _DEFAULT_MODEL_URL)
    try:
        with urlopen(_DEFAULT_MODEL_URL, timeout=180) as resp, open(zip_path, "wb") as fh:
            fh.write(resp.read())
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(_MODELS_DIR)
    except URLError as exc:
        raise ValueError(
            "Could not download the Vosk speech model "
            f"({exc.reason}). Download it manually from {_DEFAULT_MODEL_URL}, "
            "unzip it, and set VOSK_MODEL_PATH to the unpacked folder."
        ) from exc
    finally:
        try:
            zip_path.unlink()
        except OSError:
            pass

    target = _MODELS_DIR / _DEFAULT_MODEL_DIR
    if not target.is_dir():
        raise ValueError("Vosk model download did not produce the expected folder")
    return target


def _get_model():
    """Load (and cache) the Vosk model."""
    global _model
    if _model is None:
        from vosk import Model

        _model = Model(str(_ensure_model_path()))
    return _model


def transcribe(wav_bytes: bytes) -> str:
    """Transcribe 16-bit mono PCM WAV bytes to text.

    Raises ValueError on malformed audio so the caller can return a 400.
    """
    try:
        wf = wave.open(io.BytesIO(wav_bytes), "rb")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Could not read audio as WAV: {exc}") from exc

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        raise ValueError("Audio must be 16-bit mono PCM WAV")

    from vosk import KaldiRecognizer

    rec = KaldiRecognizer(_get_model(), wf.getframerate())
    pieces = []
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            pieces.append(json.loads(rec.Result()).get("text", ""))
    pieces.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(p for p in pieces if p).strip()
