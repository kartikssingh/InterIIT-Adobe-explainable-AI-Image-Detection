"""Exception hierarchy for the aidetect package.

Every failure mode raised on purpose by this package derives from
:class:`AidetectError`, so callers (CLI, REST API, notebooks) can catch a single
base class and still discriminate on the specific subclass when they care.
"""

from __future__ import annotations


class AidetectError(Exception):
    """Base class for every error raised by aidetect."""


class ConfigError(AidetectError):
    """Raised when a configuration file/override is malformed."""


class CheckpointError(AidetectError):
    """Raised when a detector checkpoint is missing or incompatible."""


class ImageLoadError(AidetectError):
    """Raised when an image cannot be read or decoded."""


class BackendUnavailableError(AidetectError):
    """Raised when an optional backend (VLM, server, metric) is not installed."""

    def __init__(self, backend: str, hint: str = "") -> None:
        message = f"Backend '{backend}' is unavailable."
        if hint:
            message = f"{message} {hint}"
        super().__init__(message)
        self.backend = backend
        self.hint = hint


class ExplainabilityError(AidetectError):
    """Raised when a saliency method cannot be applied to the given model."""


class PipelineError(AidetectError):
    """Raised when a pipeline stage fails irrecoverably."""
