"""Logging helpers.

The previous version of this project printed progress with bare ``print()``
calls scattered across every module, which made the output impossible to silence
or redirect. Everything now goes through the standard :mod:`logging` module:

* ``configure_logging`` installs a colourised console handler (colour is dropped
  automatically when stderr is not a TTY or ``NO_COLOR`` is set) and, optionally,
  a plain-text file handler.
* ``get_logger`` returns a namespaced child logger.
* ``StageTimer`` is a tiny context manager used to record per-stage latency.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

ROOT_LOGGER_NAME = "aidetect"

_LEVEL_COLORS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[1;38;5;203m",
}
_RESET = "\033[0m"
_DIM = "\033[38;5;244m"


def supports_color(stream=None) -> bool:
    """Return ``True`` when ANSI colour codes are safe to emit."""
    stream = stream or sys.stderr
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("AIDETECT_FORCE_COLOR") is not None:
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


class ConsoleFormatter(logging.Formatter):
    """Compact single-line formatter: ``LEVEL  name  message``."""

    def __init__(self, color: bool = True, show_name: bool = True) -> None:
        super().__init__()
        self.color = color
        self.show_name = show_name

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        name = record.name
        if name.startswith(ROOT_LOGGER_NAME + "."):
            name = name[len(ROOT_LOGGER_NAME) + 1 :]
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        prefix = f"{level:<7}"
        suffix = f" [{name}]" if self.show_name else ""
        if self.color:
            color = _LEVEL_COLORS.get(level, "")
            prefix = f"{color}{prefix}{_RESET}"
            suffix = f" {_DIM}[{name}]{_RESET}" if self.show_name else ""
        return f"{prefix} {message}{suffix}"


def configure_logging(
    level: str | int = "INFO",
    log_file: Optional[str | Path] = None,
    color: Optional[bool] = None,
    show_name: bool = False,
) -> logging.Logger:
    """Configure and return the package root logger.

    Repeated calls replace the handlers instead of stacking them, which keeps
    output clean when the CLI is invoked more than once in the same process
    (notebooks, tests, the REST server).
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(
        ConsoleFormatter(
            color=supports_color() if color is None else color,
            show_name=show_name,
        )
    )
    logger.addHandler(console)

    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def verbosity_to_level(verbose: int = 0, quiet: bool = False) -> str:
    """Map ``-v/-vv/-q`` counters onto logging level names."""
    if quiet:
        return "ERROR"
    if verbose >= 2:
        return "DEBUG"
    if verbose == 1:
        return "INFO"
    return "INFO"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the package root logger."""
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


class StageTimer:
    """Collect wall-clock timings for named pipeline stages."""

    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[stage] = round((time.perf_counter() - start) * 1000.0, 2)

    def as_dict(self) -> Dict[str, float]:
        total = round(sum(self.timings.values()), 2)
        return {**self.timings, "total_ms": total}


@contextmanager
def log_duration(logger: logging.Logger, message: str, level: int = logging.INFO) -> Iterator[None]:
    """Log ``message`` with the elapsed time once the block finishes."""
    start = time.perf_counter()
    logger.log(level, "%s ...", message)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "%s done in %.2fs", message, elapsed)
