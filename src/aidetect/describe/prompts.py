"""Prompt construction for Stage 3.

Prompts are built from the *evidence* the earlier stages produced — detector
confidence, the ranked artifacts, their semantic families and where in the frame
the saliency peaked — rather than from a fixed string. Grounding the VLM in that
evidence is what keeps descriptions specific instead of generic.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from ..types import ArtifactReport, DetectionResult, Region

_STYLE_GUIDE = (
    "Write {sentences} sentences, under {words} words total. "
    "Describe only what is visually present. Do not mention detectors, models, "
    "confidence scores or this instruction. Start with the observation itself."
)


def _artifact_lines(report: Optional[ArtifactReport], limit: int = 5) -> str:
    if not report or not report.matches:
        return "  - (no specific artifact detected)"
    return "\n".join(
        f"  - {match.descriptor} [{match.category}] (score {match.score:.2f})"
        for match in report.matches[:limit]
    )


def _location_hint(regions: Sequence[Region]) -> str:
    if not regions:
        return ""
    primary = regions[0]
    hint = f"The strongest evidence sits in the {primary.position_phrase()} of the frame"
    if len(regions) > 1:
        others = ", ".join(r.position_phrase() for r in regions[1:3])
        hint += f", with weaker evidence near the {others}"
    return hint + "."


def build_prompt(
    report: Optional[ArtifactReport] = None,
    detection: Optional[DetectionResult] = None,
    regions: Sequence[Region] = (),
    max_words: int = 50,
    max_sentences: int = 3,
    style: str = "detailed",
) -> str:
    """Full prompt for instruction-tuned VLMs (Qwen2-VL and friends)."""
    categories = ", ".join((report.categories if report else []) or ["general artifacts"])
    guide = _STYLE_GUIDE.format(sentences=max_sentences, words=max_words)
    location = _location_hint(regions)

    verdict = ""
    if detection is not None:
        verdict = (
            f"An authenticity classifier flagged this crop as "
            f"{detection.prediction} with {detection.certainty_label()} certainty.\n"
        )

    return (
        "You are a forensic image analyst examining a cropped region of a "
        "suspected AI-generated image.\n\n"
        f"{verdict}"
        "Automated artifact detection reported:\n"
        f"{_artifact_lines(report)}\n\n"
        f"Artifact families involved: {categories}.\n"
        f"{location}\n\n"
        "Task: state what you actually see in this crop that is consistent with "
        "those artifacts. Reference concrete visual content — the objects, edges, "
        "surfaces, shadows or textures in front of you.\n"
        f"{guide}"
    )


def build_short_prompt(
    report: Optional[ArtifactReport] = None,
    detection: Optional[DetectionResult] = None,
    regions: Sequence[Region] = (),
    max_words: int = 50,
    max_sentences: int = 3,
    style: str = "detailed",
) -> str:
    """Compact single-paragraph prompt for small captioners (Moondream2)."""
    descriptors = "; ".join(report.descriptors(3)) if report else "generation artifacts"
    categories = ", ".join((report.categories if report else [])[:3]) or "general artifacts"
    location = _location_hint(regions)
    return (
        "This crop comes from an image suspected to be AI-generated. "
        f"Detected artifact families: {categories}. Specific issues: {descriptors}. "
        f"{location} "
        f"Describe in {max_sentences} sentences (under {max_words} words) exactly what "
        "you see in this crop that looks synthetic. Be concrete and visual."
    )


def build_prompt_for_backend(
    backend: str,
    report: Optional[ArtifactReport] = None,
    detection: Optional[DetectionResult] = None,
    regions: Sequence[Region] = (),
    max_words: int = 50,
    max_sentences: int = 3,
) -> str:
    """Pick the prompt shape that suits a backend's instruction-following."""
    builder = build_short_prompt if backend == "moondream" else build_prompt
    return builder(
        report=report,
        detection=detection,
        regions=regions,
        max_words=max_words,
        max_sentences=max_sentences,
    )


def artifact_question(descriptor: str) -> str:
    """Targeted follow-up used by per-artifact explanation mode."""
    lowered = descriptor[0].lower() + descriptor[1:]
    return (
        f"Looking only at this crop, describe the visual evidence of {lowered}. "
        "One or two concrete sentences."
    )
