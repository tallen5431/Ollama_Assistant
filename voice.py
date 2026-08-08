#!/usr/bin/env python3
"""Offline speech-to-text with Vosk, with a small multi-language model catalog.

Kept optional and lazy: if the ``vosk`` package isn't installed, voice input is
simply disabled and the rest of the app runs unchanged. Models are loaded (and,
for catalog entries, downloaded) on first use and cached per directory.

Environment:
  VOSK_MODEL_PATH   Path to an already-unpacked Vosk model directory. Used as the
                    default when set (no download).
  VOSK_MODELS_DIR   Where to store/download models. Default ``./models``.
  VOSK_MODEL        Catalog id to use as the default (e.g. "es", "fr").
                    Default "en-us".
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import re
import shutil
import tempfile
import threading
import wave
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

from config import logger

_BASE_DIR = Path(__file__).parent.resolve()
_MODELS_DIR = Path(os.getenv("VOSK_MODELS_DIR", _BASE_DIR / "models")).resolve()
_MODEL_HOST = "https://alphacephei.com/vosk/models/"
_CHUNK = 256 * 1024

# A compact catalog of small Vosk models (~40-50 MB each) plus one large English
# model. ``id`` is the short handle used in the UI/API; ``dir`` is the folder
# name inside the downloaded zip.
CATALOG: Dict[str, Dict[str, str]] = {
    "en-us":       {"label": "English (US)",        "size": "40 MB",  "dir": "vosk-model-small-en-us-0.15"},
    "en-us-large": {"label": "English (US, large)", "size": "1.8 GB", "dir": "vosk-model-en-us-0.22"},
    "en-in":       {"label": "English (Indian)",    "size": "36 MB",  "dir": "vosk-model-small-en-in-0.4"},
    "es":          {"label": "Spanish",             "size": "39 MB",  "dir": "vosk-model-small-es-0.42"},
    "fr":          {"label": "French",              "size": "41 MB",  "dir": "vosk-model-small-fr-0.22"},
    "de":          {"label": "German",              "size": "45 MB",  "dir": "vosk-model-small-de-0.15"},
    "it":          {"label": "Italian",             "size": "48 MB",  "dir": "vosk-model-small-it-0.22"},
    "pt":          {"label": "Portuguese",          "size": "31 MB",  "dir": "vosk-model-small-pt-0.3"},
    "nl":          {"label": "Dutch",               "size": "39 MB",  "dir": "vosk-model-small-nl-0.22"},
    "ru":          {"label": "Russian",             "size": "45 MB",  "dir": "vosk-model-small-ru-0.22"},
    "cn":          {"label": "Chinese",             "size": "42 MB",  "dir": "vosk-model-small-cn-0.22"},
    "ja":          {"label": "Japanese",            "size": "48 MB",  "dir": "vosk-model-small-ja-0.22"},
    "ko":          {"label": "Korean",              "size": "82 MB",  "dir": "vosk-model-small-ko-0.22"},
    "hi":          {"label": "Hindi",               "size": "42 MB",  "dir": "vosk-model-small-hi-0.22"},
}

_DEFAULT_ID = os.getenv("VOSK_MODEL", "en-us")
_DIR_TO_ID = {entry["dir"]: mid for mid, entry in CATALOG.items()}

# Model ids arrive from the client (``?model=``) and may be joined onto the
# models dir, so they must stay a single plain folder name — no separators, no
# "..", nothing that could climb out of _MODELS_DIR.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

_models: Dict[str, object] = {}  # cache: resolved dir path -> vosk.Model
_models_lock = threading.Lock()  # loading is slow; don't do it twice at once


def vosk_installed() -> bool:
    """True when the ``vosk`` package can be imported."""
    try:
        import vosk  # noqa: F401
        return True
    except Exception:
        return False


def voice_available() -> bool:
    """Whether voice input can be offered (the library is present)."""
    return vosk_installed()


# ---------------------------------------------------------------------------
# Model discovery / catalog
# ---------------------------------------------------------------------------


def _looks_like_model(path: Path) -> bool:
    """Heuristic: a Vosk model dir contains an ``am`` or ``conf`` subfolder."""
    return path.is_dir() and ((path / "conf").is_dir() or (path / "am").is_dir())


def _downloaded_dirs() -> List[Path]:
    """Every unpacked model directory currently under the models dir."""
    if not _MODELS_DIR.is_dir():
        return []
    return sorted(p for p in _MODELS_DIR.iterdir() if _looks_like_model(p))


def _label_for_dir(dirname: str) -> str:
    mid = _DIR_TO_ID.get(dirname)
    return CATALOG[mid]["label"] if mid else dirname


def default_model_id() -> str:
    """The id/label shown as pre-selected in the UI."""
    if os.getenv("VOSK_MODEL_PATH"):
        return "custom"
    return _DEFAULT_ID if _DEFAULT_ID in CATALOG else "en-us"


def list_models() -> Dict[str, object]:
    """Describe which models are available and which can be downloaded.

    Returns ``{"available": [...], "catalog": [...], "default": <id>}`` where
    each entry is ``{"id", "label", "size"?, "downloaded"}``.
    """
    available: List[Dict[str, object]] = []
    seen_ids = set()

    # A custom model dir set via VOSK_MODEL_PATH always comes first.
    custom = os.getenv("VOSK_MODEL_PATH")
    if custom and Path(custom).is_dir():
        available.append({"id": "custom", "label": f"Custom ({Path(custom).name})", "downloaded": True})
        seen_ids.add("custom")

    for path in _downloaded_dirs():
        mid = _DIR_TO_ID.get(path.name, path.name)
        # Don't advertise an id _resolve_dir would reject (e.g. a folder named
        # "vosk-model-small-en-us-0.15 (1)") — it would 400 on first use.
        if mid in seen_ids or not (mid in CATALOG or _SAFE_ID.match(mid)):
            continue
        seen_ids.add(mid)
        available.append({"id": mid, "label": _label_for_dir(path.name), "downloaded": True})

    catalog: List[Dict[str, object]] = []
    for mid, entry in CATALOG.items():
        catalog.append(
            {
                "id": mid,
                "label": entry["label"],
                "size": entry["size"],
                "downloaded": _looks_like_model(_MODELS_DIR / entry["dir"]),
            }
        )

    return {"available": available, "catalog": catalog, "default": default_model_id()}


# ---------------------------------------------------------------------------
# Model resolution / download
# ---------------------------------------------------------------------------


def _resolve_dir(model_id: Optional[str], *, download: bool = True) -> Path:
    """Resolve a model id to an on-disk directory, downloading if allowed."""
    # Explicit custom path (or the "custom" sentinel from the UI).
    custom = os.getenv("VOSK_MODEL_PATH")
    if model_id in (None, "", "custom") and custom and Path(custom).is_dir():
        return Path(custom)

    mid = model_id or default_model_id()
    if mid == "custom":  # requested custom but none configured -> fall back
        mid = _DEFAULT_ID if _DEFAULT_ID in CATALOG else "en-us"

    # A raw directory name that already exists under the models dir. Reject
    # anything that isn't a plain name before touching the filesystem.
    if mid not in CATALOG:
        if not _SAFE_ID.match(mid) or mid in (".", ".."):
            raise ValueError(f"Unknown voice model: {mid!r}")
        direct = _MODELS_DIR / mid
        if _looks_like_model(direct):
            return direct
        raise ValueError(f"Unknown voice model: {mid!r}")

    target = _MODELS_DIR / CATALOG[mid]["dir"]
    if _looks_like_model(target):
        return target
    if not download:
        raise ValueError(f"Voice model '{mid}' is not downloaded yet")
    return _download(mid)


def _download(model_id: str, on_progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Download and unpack a catalog model, returning its directory.

    The zip is streamed to a private temp file rather than read into memory —
    the large English model is 1.8 GB — and the temp name is unique so two
    concurrent downloads of the same model can't clobber each other's file.

    ``on_progress(done_bytes, total_bytes)`` is called as it goes, so the page
    can say what is happening. The large model is a ten-minute download on a
    domestic line, and ten minutes of a silent request looks exactly like a
    broken one — which is the state this was in.
    """
    entry = CATALOG[model_id]
    url = _MODEL_HOST + entry["dir"] + ".zip"
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"_{model_id}-", suffix=".zip", dir=_MODELS_DIR)
    zip_path = Path(tmp_name)
    staging: Optional[Path] = None
    logger.info("Downloading Vosk model '%s' from %s (one-time) ...", model_id, url)
    try:
        # fdopen first: if urlopen raises, the fd is still wrapped and closed.
        with os.fdopen(fd, "wb") as fh, urlopen(url, timeout=600) as resp:
            if on_progress is None:
                shutil.copyfileobj(resp, fh)
            else:
                try:
                    total = int(resp.headers.get("Content-Length") or 0)
                except (TypeError, ValueError):
                    total = 0
                done = 0
                on_progress(0, total)
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    on_progress(done, total)
        # Unpack to a staging dir and move into place only once complete: a
        # download interrupted at 1.7 of 1.8 GB would otherwise leave a folder
        # that every later check reports as "already downloaded".
        staging = Path(tempfile.mkdtemp(prefix=f"_{model_id}-unpack-", dir=_MODELS_DIR))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(staging)
        unpacked = staging / entry["dir"]
        if not _looks_like_model(unpacked):
            raise ValueError("Voice model archive did not contain the expected folder")
        final = _MODELS_DIR / entry["dir"]
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        os.replace(unpacked, final)
    except URLError as exc:
        raise ValueError(
            f"Could not download the '{model_id}' voice model ({exc.reason}). "
            f"Download it manually from {url}, unzip it into {_MODELS_DIR}, "
            "or set VOSK_MODEL_PATH to an unpacked model folder."
        ) from exc
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"The downloaded '{model_id}' voice model was not a valid zip "
            "(the download may have been truncated). Try again."
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ValueError(
            f"The '{model_id}' voice model download failed ({exc}). "
            f"Download it manually from {url}, unzip it into {_MODELS_DIR}, "
            "or set VOSK_MODEL_PATH to an unpacked model folder."
        ) from exc
    finally:
        try:
            zip_path.unlink()
        except OSError:
            pass
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    target = _MODELS_DIR / entry["dir"]
    if not target.is_dir():
        raise ValueError("Voice model download did not produce the expected folder")
    return target


def download_model(
    model_id: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, object]:
    """Public helper: ensure a catalog model is present (used by the API)."""
    if model_id not in CATALOG:
        raise ValueError(f"Unknown voice model: {model_id!r}")
    entry = _MODELS_DIR / CATALOG[model_id]["dir"]
    if _looks_like_model(entry):
        # Already here: report it rather than reporting nothing, or a second
        # click looks like a download that finished suspiciously fast.
        return {"id": model_id, "label": CATALOG[model_id]["label"],
                "dir": entry.name, "already": True}
    path = _download(model_id, on_progress)
    return {"id": model_id, "label": CATALOG[model_id]["label"], "dir": path.name}


def _get_model(model_id: Optional[str], *, download: bool = True):
    """Load (and cache) the Vosk model for ``model_id``.

    Keyed on the resolved path so a relative VOSK_MODEL_PATH and its catalog
    equivalent don't load the same 1.8 GB model twice, and guarded by a lock
    because continuous mode can fire overlapping transcriptions on a cold cache.
    """
    key = str(_resolve_dir(model_id, download=download).resolve())
    cached = _models.get(key)
    if cached is not None:
        return cached
    with _models_lock:
        if key not in _models:
            from vosk import Model

            _models[key] = Model(key)
        return _models[key]


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe(wav_bytes: bytes, model_id: Optional[str] = None) -> str:
    """Transcribe 16-bit mono PCM WAV bytes to text with the chosen model.

    Never downloads. The model id arrives from the client, so resolving with
    downloads enabled meant a request naming an unfetched model made the box
    pull 1.8 GB before reading a single audio frame — and several at once each
    pulled their own copy while holding a server thread. /api/voice/download is
    the explicit, deliberate way to fetch one.

    Raises ValueError on malformed audio / unknown model so the caller can
    return a 400.
    """
    try:
        wf = wave.open(io.BytesIO(wav_bytes), "rb")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Could not read audio as WAV: {exc}") from exc

    with contextlib.closing(wf):
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("Audio must be 16-bit mono PCM WAV")

        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(_get_model(model_id, download=False), wf.getframerate())
        pieces = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                pieces.append(json.loads(rec.Result()).get("text", ""))
        pieces.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(p for p in pieces if p).strip()
