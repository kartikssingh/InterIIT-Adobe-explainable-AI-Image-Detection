"""Backend contract for Stage 3 description generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..config import AppConfig, DescribeConfig
from ..device import resolve_device
from ..exceptions import BackendUnavailableError
from ..logging_utils import get_logger

logger = get_logger(__name__)


class DescriptionBackend(ABC):
    """Common interface every description backend implements."""

    name: str = "base"
    #: Whether the backend actually looks at pixels (rule-based does not).
    is_visual: bool = True

    def __init__(self, config: Optional[AppConfig] = None, device: Optional[str] = None) -> None:
        self.config = config or AppConfig()
        self.cfg: DescribeConfig = self.config.describe
        self.device = resolve_device(device or self.config.runtime.device)
        self._loaded = False

    # -- lifecycle --------------------------------------------------------- #

    @classmethod
    def is_available(cls) -> bool:
        """Whether the backend's dependencies are importable."""
        return True

    def load(self) -> "DescriptionBackend":
        """Load weights (idempotent). Subclasses override :meth:`_load`."""
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def _load(self) -> None:  # pragma: no cover - trivial default
        return None

    def unload(self) -> None:
        self._loaded = False

    def __enter__(self) -> "DescriptionBackend":
        return self.load()

    def __exit__(self, *exc_info: Any) -> None:
        self.unload()

    # -- generation -------------------------------------------------------- #

    @abstractmethod
    def generate(self, image, prompt: str, **kwargs: Any) -> str:
        """Return raw model text for ``prompt`` conditioned on ``image``."""

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "device": self.device,
            "visual": self.is_visual,
            "loaded": self._loaded,
            "available": self.is_available(),
        }


_REGISTRY: Dict[str, type] = {}
_MODULES_IMPORTED = False


def register_backend(name: str):
    """Class decorator adding a backend to the registry."""

    def wrapper(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrapper


def available_backends() -> Dict[str, bool]:
    """Map every registered backend name to whether its deps are installed."""
    _ensure_registered()
    return {name: cls.is_available() for name, cls in _REGISTRY.items()}


def get_backend_class(name: str) -> type:
    _ensure_registered()
    if name not in _REGISTRY:
        raise BackendUnavailableError(
            name, f"Known backends: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name]


def build_backend(
    name: str,
    config: Optional[AppConfig] = None,
    device: Optional[str] = None,
) -> DescriptionBackend:
    """Instantiate a backend by name."""
    cls = get_backend_class(name)
    if not cls.is_available():
        raise BackendUnavailableError(
            name, "Its Python dependencies are not installed."
        )
    return cls(config=config, device=device)  # type: ignore[return-value]


def _ensure_registered() -> None:
    """Import every backend module so its decorator runs.

    Guarded by an explicit flag rather than ``if _REGISTRY`` — importing
    ``generator`` pulls in ``rule_based`` on its own, and an emptiness check
    would then treat the registry as complete and hide the VLM backends.
    """
    global _MODULES_IMPORTED
    if _MODULES_IMPORTED:
        return
    from . import moondream, qwen2vl, rule_based  # noqa: F401

    _MODULES_IMPORTED = True
