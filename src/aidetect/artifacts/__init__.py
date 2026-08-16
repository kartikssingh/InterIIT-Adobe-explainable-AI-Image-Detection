"""Stage 2: the artifact taxonomy and its zero-shot classifier."""

from __future__ import annotations

from typing import Any

from .taxonomy import (
    CATEGORIES,
    CATEGORY_NAMES,
    CATEGORY_TO_DESCRIPTORS,
    DESCRIPTOR_TO_CATEGORY,
    DESCRIPTORS,
    OFFICIAL_ARTIFACTS,
    artifact_id,
    category_of,
    taxonomy_table,
    validate_taxonomy,
)

__all__ = [
    "CATEGORIES",
    "CATEGORY_NAMES",
    "CATEGORY_TO_DESCRIPTORS",
    "DESCRIPTORS",
    "DESCRIPTOR_TO_CATEGORY",
    "OFFICIAL_ARTIFACTS",
    "ArtifactClassifier",
    "artifact_id",
    "category_of",
    "taxonomy_table",
    "validate_taxonomy",
]


def __getattr__(name: str) -> Any:  # torch/transformers stay lazy
    if name == "ArtifactClassifier":
        from .classifier import ArtifactClassifier

        return ArtifactClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
