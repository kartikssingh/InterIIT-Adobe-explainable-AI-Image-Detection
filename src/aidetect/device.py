"""Device and precision resolution.

Torch is imported lazily inside the functions so that importing ``aidetect``
(for the CLI help text, the taxonomy, or the report writer) stays instant and
works on machines without a GPU stack.
"""

from __future__ import annotations

from typing import Any, Dict

from .logging_utils import get_logger

logger = get_logger(__name__)


def resolve_device(preference: str = "auto") -> str:
    """Return a concrete torch device string for a user preference.

    ``auto`` picks CUDA, then Apple MPS, then CPU. An explicit preference that is
    unavailable degrades to CPU with a warning rather than crashing.
    """
    import torch

    pref = (preference or "auto").strip().lower()

    if pref in ("auto", ""):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — falling back to CPU")
            return "cpu"
        if ":" in pref:
            index = int(pref.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                logger.warning("cuda:%d not present — using cuda:0", index)
                return "cuda:0"
        return pref

    if pref == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            logger.warning("MPS requested but unavailable — falling back to CPU")
            return "cpu"
        return "mps"

    if pref == "cpu":
        return "cpu"

    logger.warning("Unknown device %r — falling back to auto-detection", preference)
    return resolve_device("auto")


def resolve_dtype(dtype: str, device: str) -> Any:
    """Map a dtype name plus a device onto a concrete ``torch.dtype``.

    ``auto`` means: bfloat16 on Ampere+ GPUs, float16 on older GPUs, float32
    everywhere else (CPU float16 matmuls are slow and often unsupported).
    """
    import torch

    name = (dtype or "auto").strip().lower()
    explicit = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name in explicit:
        return explicit[name]

    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def autocast_enabled(device: str, dtype: Any) -> bool:
    """Autocast only helps on CUDA with a reduced-precision dtype."""
    import torch

    return device.startswith("cuda") and dtype in (torch.float16, torch.bfloat16)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs for reproducible inference."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch always present in practice
        pass


def device_report() -> Dict[str, Any]:
    """Collect a diagnostics dictionary about the current machine."""
    import platform
    import sys

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpus"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps is not None and mps.is_available())
    except ImportError:
        info["torch"] = None

    for module in ("timm", "transformers", "numpy", "PIL"):
        try:
            mod = __import__(module)
            info[module] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[module] = None

    return info


def memory_snapshot(device: str) -> Dict[str, float]:
    """Return allocated/reserved GPU memory in GiB (empty dict off-GPU)."""
    if not device.startswith("cuda"):
        return {}
    import torch

    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def free_memory(device: str) -> None:
    """Release cached GPU blocks (useful between batch chunks)."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
