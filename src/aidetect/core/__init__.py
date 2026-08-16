"""Stage 1 building blocks: model, preprocessing, detection, localisation."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["Detector", "DINOv2Classifier", "load_model", "localization"]

_LAZY = {
    "Detector": ("aidetect.core.detector", "Detector"),
    "DINOv2Classifier": ("aidetect.core.model", "DINOv2Classifier"),
    "load_model": ("aidetect.core.model", "load_model"),
}


def __getattr__(name: str) -> Any:  # keep torch out of package import time
    if name == "localization":
        # importlib (not `from . import ...`) so this never re-enters __getattr__.
        module = importlib.import_module("aidetect.core.localization")
        globals()[name] = module
        return module

    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value
