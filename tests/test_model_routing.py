"""Which model gets which job when an image is attached.

Two distinct roles that a single "can it see images" flag conflates: reading
text *out* of an image to build a search query, and answering a question
*about* one. An OCR model is the best choice for the first and a poor one for
the second, so the split is asserted here rather than left to chance.
"""

from __future__ import annotations

import pytest

import ollama_client as oc

# The models actually installed on the target machine.
INSTALLED = [
    {"name": "glm-ocr:latest", "size": 2_200_000_000, "details": {"families": ["glm"]}},
    {"name": "qwen3-vl:30b", "size": 19_000_000_000, "details": {"families": ["qwen3vl"]}},
    {"name": "qwen3.5:4b", "size": 3_400_000_000, "details": {"families": ["qwen3"]}},
    {"name": "minicpm-v:latest", "size": 5_500_000_000, "details": {"families": ["qwen2", "clip"]}},
    {"name": "qwen2.5vl:3b", "size": 3_200_000_000, "details": {"families": ["qwen25vl"]}},
    {"name": "deepseek-r1:8b", "size": 5_200_000_000, "details": {"families": ["qwen2"]}},
    {"name": "llama3.1:8b", "size": 4_900_000_000, "details": {"families": ["llama"]}},
    {"name": "qwen2.5-coder:14b", "size": 9_000_000_000, "details": {"families": ["qwen2"]}},
    {"name": "qwen3-coder:30b", "size": 18_000_000_000, "details": {"families": ["qwen3"]}},
]


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(oc, "list_models", lambda: INSTALLED)


class TestCapabilityDetection:
    @pytest.mark.parametrize(
        "name,vision",
        [
            ("glm-ocr:latest", True),
            ("qwen3-vl:30b", True),
            ("minicpm-v:latest", True),
            ("qwen2.5vl:3b", True),
            ("qwen3.5:4b", False),
            ("llama3.1:8b", False),
            ("deepseek-r1:8b", False),
            ("qwen2.5-coder:14b", False),
            ("qwen3-coder:30b", False),
        ],
    )
    def test_vision_detection(self, name, vision):
        model = next(m for m in INSTALLED if m["name"] == name)
        assert oc.has_vision(model) is vision

    def test_only_the_ocr_model_is_flagged_as_ocr(self):
        flagged = [m["name"] for m in INSTALLED if oc.is_ocr(m)]
        assert flagged == ["glm-ocr:latest"]

    def test_detection_survives_a_missing_details_block(self):
        """Some builds report no families; the name has to carry it."""
        assert oc.has_vision({"name": "glm-ocr:latest"}) is True
        assert oc.has_vision({"name": "qwen3-vl:30b"}) is True
        assert oc.has_vision({"name": "llama3.1:8b"}) is False


class TestRoleSeparation:
    def test_answering_excludes_ocr_and_prefers_the_smallest(self, installed):
        """A 19 GB load because someone pasted a screenshot is a poor surprise."""
        answering = oc.vision_models(include_ocr=False)
        assert "glm-ocr:latest" not in answering
        assert answering[0] == "qwen2.5vl:3b"

    def test_ocr_models_are_listed_separately(self, installed):
        assert oc.ocr_models() == ["glm-ocr:latest"]

    def test_the_full_vision_set_still_includes_ocr(self, installed):
        assert "glm-ocr:latest" in oc.vision_models()

    def test_no_ocr_installed_leaves_the_reader_to_a_general_model(self, monkeypatch):
        monkeypatch.setattr(oc, "list_models",
                            lambda: [m for m in INSTALLED if not oc.is_ocr(m)])
        assert oc.ocr_models() == []
        assert oc.vision_models(include_ocr=False)[0] == "qwen2.5vl:3b"

    def test_unreachable_ollama_yields_empty_lists(self, monkeypatch):
        def boom():
            raise ValueError("no ollama")
        monkeypatch.setattr(oc, "list_models", boom)
        assert oc.vision_models() == []
        assert oc.ocr_models() == []


class TestImageReader:
    """app._image_reader picks who reads an attached image, and how."""

    def test_ocr_wins_when_installed(self, installed, monkeypatch):
        import app
        monkeypatch.delenv("WEB_VISION_MODEL", raising=False)
        monkeypatch.setattr(app, "ocr_models", oc.ocr_models)
        monkeypatch.setattr(app, "vision_models", oc.vision_models)
        assert app._image_reader("llama3.1:8b") == ("glm-ocr:latest", True)

    def test_falls_back_to_the_answering_model_when_it_can_see(self, monkeypatch):
        import app
        monkeypatch.delenv("WEB_VISION_MODEL", raising=False)
        monkeypatch.setattr(app, "ocr_models", lambda: [])
        monkeypatch.setattr(app, "vision_models", lambda include_ocr=True: ["minicpm-v:latest"])
        assert app._image_reader("minicpm-v:latest") == ("minicpm-v:latest", False)

    def test_an_explicit_pin_wins(self, monkeypatch):
        import app
        monkeypatch.setenv("WEB_VISION_MODEL", "qwen3-vl:30b")
        assert app._image_reader("llama3.1:8b") == ("qwen3-vl:30b", False)

    def test_a_pinned_ocr_model_is_recognised_as_one(self, monkeypatch):
        import app
        monkeypatch.setenv("WEB_VISION_MODEL", "glm-ocr:latest")
        assert app._image_reader("llama3.1:8b") == ("glm-ocr:latest", True)

    def test_nothing_installed_means_no_reader(self, monkeypatch):
        import app
        monkeypatch.delenv("WEB_VISION_MODEL", raising=False)
        monkeypatch.setattr(app, "ocr_models", lambda: [])
        monkeypatch.setattr(app, "vision_models", lambda include_ocr=True: [])
        assert app._image_reader("llama3.1:8b") == ("", False)


class TestTranscriptionPrompt:
    def test_an_ocr_model_is_asked_to_transcribe_not_describe(self, monkeypatch):
        import web
        seen = {}

        def capture(model, messages, options=None):
            seen["prompt"] = messages[0]["content"]
            return "TypeError: can't concat str to bytes"

        monkeypatch.setattr("ollama_client.chat", capture)
        web.describe_images(["aW1n"], "glm-ocr:latest", ocr=True)
        assert "Transcribe" in seen["prompt"]

        web.describe_images(["aW1n"], "minicpm-v:latest", ocr=False)
        assert "Describe" in seen["prompt"]

    def test_extracted_text_reaches_the_search_planner(self, monkeypatch):
        import web
        seen = {}

        def capture(model, messages, options=None):
            seen["prompt"] = messages[-1]["content"]
            return "Q: TypeError concat str to bytes python"

        monkeypatch.setattr("ollama_client.chat", capture)
        queries = web.plan_searches(
            [{"role": "user", "content": "what's this?"}],
            "llama3.1:8b",
            image_note="TypeError: can't concat str to bytes",
        )
        assert "can't concat str to bytes" in seen["prompt"]
        assert queries == ["TypeError concat str to bytes python"]
