"""On-disk cache for descriptor text embeddings.

Encoding 70 descriptors x 5 prompt templates costs a few seconds of GPU time on
every process start. The embeddings only depend on (model, templates,
descriptors), so they are cached in a small ``.npz`` keyed by a hash of exactly
those inputs — stale caches can never be silently reused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..io_utils import ensure_dir, text_digest
from ..logging_utils import get_logger

logger = get_logger(__name__)

CACHE_VERSION = "v2"


def cache_key(model_name: str, descriptors: Sequence[str], templates: Sequence[str]) -> str:
    """Stable short hash identifying one embedding matrix."""
    return text_digest(
        CACHE_VERSION,
        model_name,
        "|".join(descriptors),
        "|".join(templates),
        length=20,
    )


def cache_path(directory: str | Path, key: str) -> Path:
    return Path(directory).expanduser() / f"text_emb_{key}.npz"


def load(directory: Optional[str | Path], key: str, expected_rows: int) -> Optional[np.ndarray]:
    """Return the cached matrix, or ``None`` when absent/invalid."""
    if not directory:
        return None
    path = cache_path(directory, key)
    if not path.is_file():
        return None
    try:
        with np.load(path) as payload:
            matrix = payload["embeddings"]
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Ignoring unreadable embedding cache %s (%s)", path, exc)
        return None

    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        logger.warning(
            "Embedding cache %s has shape %s, expected %d rows — recomputing",
            path,
            matrix.shape,
            expected_rows,
        )
        return None

    logger.debug("Loaded text embeddings from %s", path)
    return matrix


def save(directory: Optional[str | Path], key: str, matrix: np.ndarray) -> Optional[Path]:
    """Persist an embedding matrix; failures are logged, never raised."""
    if not directory:
        return None
    try:
        ensure_dir(directory)
        path = cache_path(directory, key)
        np.savez_compressed(path, embeddings=matrix.astype(np.float32))
        logger.debug("Cached text embeddings at %s", path)
        return path
    except OSError as exc:  # pragma: no cover - depends on filesystem perms
        logger.warning("Could not write embedding cache: %s", exc)
        return None


def clear(directory: Optional[str | Path]) -> int:
    """Delete every cached embedding file. Returns the number removed."""
    if not directory:
        return 0
    folder = Path(directory).expanduser()
    if not folder.is_dir():
        return 0
    removed = 0
    for path in folder.glob("text_emb_*.npz"):
        try:
            path.unlink()
            removed += 1
        except OSError:  # pragma: no cover
            logger.warning("Could not delete %s", path)
    return removed
