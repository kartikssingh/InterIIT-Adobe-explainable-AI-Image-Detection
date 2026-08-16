"""Stage 3: post-processing, prompt construction, the registry and the rule-based writer."""

from __future__ import annotations

import pytest

from aidetect.describe import postprocess, prompts
from aidetect.describe.rule_based import RuleBasedBackend, compose_description
from aidetect.types import ArtifactMatch, ArtifactReport, DetectionResult, Region


def make_report() -> ArtifactReport:
    return ArtifactReport(
        matches=[
            ArtifactMatch("Artificial smoothness", 0.9, "Texture & Material", 1),
            ArtifactMatch("Inconsistent shadow directions", 0.8, "Lighting & Shadows", 2),
            ArtifactMatch("Over-smoothing of natural textures", 0.7, "Texture & Material", 3),
        ],
        categories=["Texture & Material", "Lighting & Shadows"],
    )


def make_detection() -> DetectionResult:
    return DetectionResult(
        image_path="x.png",
        prediction="FAKE",
        confidence=94.0,
        probabilities={"FAKE": 0.94, "REAL": 0.06},
    )


# ---- post-processing ------------------------------------------------------ #


def test_strip_filler_removes_assistant_preamble():
    assert postprocess.strip_filler("Sure, the surface looks plastic.") == "The surface looks plastic."
    assert postprocess.strip_filler("Answer: edges are soft.") == "Edges are soft."


def test_strip_filler_removes_code_fences_and_quotes():
    assert postprocess.strip_filler('```\n"Edges bleed."\n```') == "Edges bleed."


def test_limit_words_prefers_a_sentence_boundary():
    text = "One two three four five. Six seven eight nine ten eleven twelve."
    trimmed, truncated = postprocess.limit_words(text, 8)
    assert truncated
    assert trimmed == "One two three four five."


def test_limit_words_hard_cuts_when_no_sentence_fits():
    trimmed, truncated = postprocess.limit_words("a b c d e f g h", 4)
    assert truncated
    assert len(trimmed.split()) == 4


def test_limit_words_is_a_noop_within_budget():
    trimmed, truncated = postprocess.limit_words("short text", 50)
    assert not truncated and trimmed == "short text"


def test_dedupe_sentences():
    text = "Edges bleed. Edges bleed! Shadows conflict."
    assert postprocess.limit_sentences(text, 3) == "Edges bleed. Shadows conflict."


def test_polish_enforces_budget_and_punctuation():
    raw = "Sure! " + " ".join(["word"] * 80)
    text, truncated = postprocess.polish(raw, max_words=50, max_sentences=3)
    assert truncated
    assert len(text.split()) <= 50
    assert text.endswith(".")


def test_polish_on_empty_input():
    assert postprocess.polish("   ") == ("", False)


# ---- prompts -------------------------------------------------------------- #


def test_full_prompt_mentions_evidence_and_budget():
    prompt = prompts.build_prompt(
        make_report(),
        make_detection(),
        [Region(bbox=(0, 0, 10, 10), score=0.9, peak_pct=(10.0, 10.0))],
        max_words=50,
    )
    assert "Artificial smoothness" in prompt
    assert "Texture & Material" in prompt
    assert "upper-left" in prompt
    assert "50 words" in prompt


def test_short_prompt_is_shorter():
    report, detection = make_report(), make_detection()
    assert len(prompts.build_short_prompt(report, detection)) < len(
        prompts.build_prompt(report, detection)
    )


def test_backend_specific_prompt_selection():
    report = make_report()
    moondream = prompts.build_prompt_for_backend("moondream", report)
    qwen = prompts.build_prompt_for_backend("qwen2vl", report)
    assert moondream != qwen
    assert "forensic image analyst" in qwen


def test_prompt_survives_missing_stage2():
    prompt = prompts.build_prompt(None, None, [])
    assert "no specific artifact detected" in prompt


# ---- rule-based writer ---------------------------------------------------- #


def test_rule_based_description_uses_the_evidence():
    text = compose_description(
        make_report(), make_detection(), [Region(bbox=(0, 0, 5, 5), score=0.9, peak_pct=(80.0, 80.0))]
    )
    assert "artificial smoothness" in text.lower()
    assert "lower-right" in text
    assert text.endswith(".")
    assert len(text.split(". ")) >= 2


def test_rule_based_varies_with_the_dominant_family():
    texture = compose_description(make_report())
    lighting_report = ArtifactReport(
        matches=[ArtifactMatch("Inconsistent shadow directions", 0.9, "Lighting & Shadows", 1)],
        categories=["Lighting & Shadows"],
    )
    assert compose_description(lighting_report) != texture


def test_rule_based_handles_no_artifacts():
    text = compose_description(None, make_detection(), [])
    assert text and text.endswith(".")


def test_rule_based_backend_is_always_available():
    backend = RuleBasedBackend()
    assert backend.is_available()
    assert not backend.is_visual
    text = backend.generate(None, "ignored", report=make_report(), detection=make_detection())
    assert "smoothness" in text.lower()


def test_per_artifact_sentences():
    sentences = RuleBasedBackend().per_artifact(make_report(), make_detection(), [], limit=2)
    assert len(sentences) == 2
    assert all(s.endswith(".") for s in sentences)


# ---- backend registry ----------------------------------------------------- #


def test_registry_lists_every_backend_after_generator_import():
    """Importing the generator pulls in rule_based; the VLMs must still register."""
    from aidetect.describe.generator import DescriptionGenerator  # noqa: F401
    from aidetect.describe.base import available_backends, get_backend_class

    backends = available_backends()
    assert set(backends) == {"rule_based", "moondream", "qwen2vl"}
    assert backends["rule_based"] is True
    assert get_backend_class("qwen2vl").name == "qwen2vl"


def test_unknown_backend_raises_with_a_helpful_message():
    from aidetect.describe.base import build_backend
    from aidetect.exceptions import BackendUnavailableError

    with pytest.raises(BackendUnavailableError) as excinfo:
        build_backend("gpt5-vision")
    assert "rule_based" in str(excinfo.value)


def test_generator_falls_back_to_rules_when_a_backend_fails(monkeypatch):
    from aidetect.config import AppConfig
    from aidetect.describe.generator import DescriptionGenerator

    config = AppConfig()
    config.describe.backend = "moondream"
    generator = DescriptionGenerator(config)

    class ExplodingBackend:
        name = "moondream"
        is_visual = True

        def generate(self, *args, **kwargs):
            raise RuntimeError("weights not cached")

        def unload(self):
            pass

    generator._backend = ExplodingBackend()
    explanation = generator.generate(image=None, report=make_report(), detection=make_detection())

    assert explanation.fallback_used
    assert explanation.backend == "rule_based"
    assert explanation.text


def test_generator_rejects_degenerate_backend_output():
    from aidetect.config import AppConfig
    from aidetect.describe.generator import DescriptionGenerator

    generator = DescriptionGenerator(AppConfig())

    class TerseBackend:
        name = "moondream"
        is_visual = True

        def generate(self, *args, **kwargs):
            return "Yes."

        def unload(self):
            pass

    generator.backend_name = "moondream"
    generator._backend = TerseBackend()
    explanation = generator.generate(image=None, report=make_report())

    assert explanation.fallback_used
    assert len(explanation.text.split()) > 5
