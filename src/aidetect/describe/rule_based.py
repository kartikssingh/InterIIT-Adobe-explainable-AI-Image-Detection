"""Deterministic, dependency-free description backend.

This is the default backend: it needs no model download, runs in microseconds and
always produces a grammatical, on-topic explanation. It composes text from the
evidence the earlier stages produced — the ranked artifacts, their semantic
family, where the saliency peaked and how decisive the classifier was — so the
output varies meaningfully with the input instead of being one fixed sentence.

Because it is fully deterministic it is also the fallback whenever a VLM backend
fails, which guarantees the pipeline always returns *something* useful.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..artifacts import taxonomy
from ..types import ArtifactReport, DetectionResult, Region
from .base import DescriptionBackend, register_backend

#: Opening clause keyed by the dominant artifact family.
_FAMILY_OPENERS = {
    "Structure & Geometry": "Object outlines in this region do not resolve cleanly",
    "Texture & Material": "Surface detail in this region behaves unlike a photographed material",
    "Anatomy & Biology": "The biological structure in this region is inconsistent",
    "Lighting & Shadows": "The lighting in this region cannot come from one physical setup",
    "Perspective & Depth": "Depth cues in this region contradict each other",
    "Signal & Rendering": "Pixel-level detail in this region carries rendering marks",
    "Color & Tone": "Colour behaves too uniformly in this region for a camera capture",
    "Style & Composition": "The framing of this region is staged rather than observed",
}

_CERTAINTY_CLAUSE = {
    "very high": "the evidence is unambiguous",
    "high": "the evidence is strong",
    "moderate": "the evidence is clear but localised",
    "borderline": "the evidence is subtle",
}


def _position_clause(regions: Sequence[Region]) -> str:
    if not regions:
        return ""
    primary = regions[0]
    where = primary.position_phrase()
    if len(regions) > 1:
        return f"concentrated in the {where} with a second affected area elsewhere in the frame"
    return f"concentrated in the {where} of the frame"


def _artifact_clause(descriptors: Sequence[str]) -> str:
    lowered = [d[0].lower() + d[1:] for d in descriptors if d]
    if not lowered:
        return "generic generation artifacts"
    if len(lowered) == 1:
        return lowered[0]
    if len(lowered) == 2:
        return f"{lowered[0]} and {lowered[1]}"
    return f"{', '.join(lowered[:-1])} and {lowered[-1]}"


def compose_description(
    report: Optional[ArtifactReport] = None,
    detection: Optional[DetectionResult] = None,
    regions: Sequence[Region] = (),
    max_artifacts: int = 3,
) -> str:
    """Build a 2-3 sentence explanation from structured evidence."""
    matches = list(report.matches[:max_artifacts]) if report else []

    if not matches:
        base = (
            "This region shows visual inconsistencies typical of generated imagery: "
            "detail resolves unevenly and surfaces lack the micro-variation of a "
            "camera capture."
        )
        position = _position_clause(regions)
        return f"{base} The affected area is {position}." if position else base

    primary = matches[0]
    family = primary.category
    opener = _FAMILY_OPENERS.get(family, "This region shows synthetic rendering traces")

    sentence_1 = f"{opener}, showing {_artifact_clause([primary.descriptor])}"
    position = _position_clause(regions)
    if position:
        sentence_1 += f", {position}"
    sentence_1 += "."

    secondary = [m.descriptor for m in matches[1:]]
    if secondary:
        sentence_2 = (
            f"The same area also shows {_artifact_clause(secondary)}, "
            f"so {taxonomy.evidence_phrase(family)}."
        )
    else:
        sentence_2 = f"As a result {taxonomy.evidence_phrase(family)}."

    families = list(dict.fromkeys(m.category for m in matches))
    if len(families) > 1:
        closing_subject = f"Failures across {len(families)} unrelated artifact families"
    else:
        closing_subject = f"Repeated {family.split(' & ')[0].lower()} failures"

    certainty = _CERTAINTY_CLAUSE.get(
        detection.certainty_label() if detection else "high", "the evidence is strong"
    )
    sentence_3 = (
        f"{closing_subject} in a single region point to synthetic generation rather "
        f"than photographic capture, and {certainty}."
    )

    return " ".join([sentence_1, sentence_2, sentence_3])


@register_backend("rule_based")
class RuleBasedBackend(DescriptionBackend):
    """Template composer — always available, never downloads anything."""

    name = "rule_based"
    is_visual = False

    @classmethod
    def is_available(cls) -> bool:
        return True

    def generate(self, image, prompt: str, **kwargs: Any) -> str:
        """Ignores ``prompt``/``image``; composes from structured evidence."""
        return compose_description(
            report=kwargs.get("report"),
            detection=kwargs.get("detection"),
            regions=kwargs.get("regions") or (),
            max_artifacts=kwargs.get("max_artifacts", 3),
        )

    def per_artifact(
        self,
        report: Optional[ArtifactReport],
        detection: Optional[DetectionResult] = None,
        regions: Sequence[Region] = (),
        limit: int = 3,
    ) -> List[str]:
        """One short explanation per artifact, for the Task-2 submission format."""
        if not report:
            return []
        sentences: List[str] = []
        for match in report.matches[:limit]:
            lowered = match.descriptor[0].lower() + match.descriptor[1:]
            where = f" in the {regions[0].position_phrase()} of the frame" if regions else ""
            sentences.append(
                f"The crop shows {lowered}{where}, which is inconsistent with how a "
                f"real camera records this kind of surface."
            )
        return sentences
