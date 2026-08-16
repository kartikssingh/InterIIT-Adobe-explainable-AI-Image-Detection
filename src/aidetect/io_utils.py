"""Filesystem and image IO helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .exceptions import ImageLoadError
from .logging_utils import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".ppm",
    ".jfif",
)


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def load_image(path: str | Path, apply_exif: bool = True):
    """Load an image as RGB, honouring EXIF orientation.

    Raises :class:`ImageLoadError` with the offending path instead of leaking a
    raw PIL exception, so the batch runner can report a useful message.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    file_path = Path(path)
    if not file_path.exists():
        raise ImageLoadError(f"Image not found: {file_path}")
    if file_path.is_dir():
        raise ImageLoadError(f"Expected an image file but got a directory: {file_path}")

    try:
        with open(file_path, "rb") as handle:
            image = Image.open(handle)
            image.load()
    except UnidentifiedImageError as exc:
        raise ImageLoadError(f"Unsupported or corrupt image: {file_path}") from exc
    except OSError as exc:
        raise ImageLoadError(f"Could not read image {file_path}: {exc}") from exc

    if apply_exif:
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:  # pragma: no cover - malformed EXIF is not fatal
            logger.debug("EXIF transpose failed for %s", file_path)

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def iter_image_paths(
    inputs: Sequence[str | Path],
    recursive: bool = True,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
    sort: bool = True,
) -> List[Path]:
    """Expand files, directories and glob patterns into a list of image paths."""
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    found: List[Path] = []
    seen: set[str] = set()

    def _add(candidate: Path) -> None:
        resolved = str(candidate.resolve())
        if resolved not in seen and candidate.suffix.lower() in exts:
            seen.add(resolved)
            found.append(candidate)

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in path.glob(pattern):
                if child.is_file():
                    _add(child)
        elif path.is_file():
            _add(path)
        else:
            # Treat as a glob pattern relative to the current directory.
            matches = sorted(Path().glob(str(raw)))
            if not matches:
                logger.warning("No images matched %s", raw)
            for match in matches:
                if match.is_file():
                    _add(match)

    return sorted(found, key=lambda p: str(p)) if sort else found


def image_size(path: str | Path) -> tuple[int, int]:
    """Read image dimensions without decoding the full pixel data."""
    from PIL import Image

    with Image.open(path) as img:
        return img.size


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(text: str, max_length: int = 60) -> str:
    """Make an arbitrary string safe to use as a filename component."""
    slug = _SLUG_RE.sub("-", str(text)).strip("-._")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:max_length] or "item").lower()


def unique_path(path: str | Path) -> Path:
    """Return ``path`` or ``path-1``/``path-2``… if it already exists."""
    candidate = Path(path)
    if not candidate.exists():
        return candidate
    stem, suffix, parent = candidate.stem, candidate.suffix, candidate.parent
    for index in range(1, 10_000):
        alternative = parent / f"{stem}-{index}{suffix}"
        if not alternative.exists():
            return alternative
    raise OSError(f"Could not find a free filename near {path}")


def file_digest(path: str | Path, algorithm: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """Stream a file through a hash function (used for cache keys)."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_digest(*parts: str, algorithm: str = "sha256", length: int = 16) -> str:
    digest = hashlib.new(algorithm)
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:length]


def relative_to(path: str | Path, base: str | Path) -> str:
    """Best-effort relative path, falling back to the absolute one."""
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Write via a temporary file + rename so readers never see a partial file."""
    target = Path(path)
    ensure_dir(target.parent)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, target)
    return target


def save_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(data, indent=indent, default=str))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d in %s", line_number, path)


def save_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Optional[Sequence[str]] = None) -> Path:
    """Write rows to CSV; column order defaults to the first row's keys."""
    target = Path(path)
    ensure_dir(target.parent)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    fieldnames = list(columns) if columns else list(rows[0].keys())
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def data_uri(path: str | Path, mime: Optional[str] = None) -> str:
    """Encode a file as a ``data:`` URI (used for self-contained HTML reports)."""
    import base64
    import mimetypes

    file_path = Path(path)
    mime = mime or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


def chunked(items: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Yield successive ``size``-length chunks from ``items``."""
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
