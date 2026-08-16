"""Stage 3: natural-language explanation backends."""

from __future__ import annotations

from typing import Any

from .base import DescriptionBackend, available_backends, build_backend, register_backend

__all__ = [
    "DescriptionBackend",
    "DescriptionGenerator",
    "available_backends",
    "build_backend",
    "register_backend",
    "compose_description",
]


def __getattr__(name: str) -> Any:
    if name == "DescriptionGenerator":
        from .generator import DescriptionGenerator

        return DescriptionGenerator
    if name == "compose_description":
        from .rule_based import compose_description

        return compose_description
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
