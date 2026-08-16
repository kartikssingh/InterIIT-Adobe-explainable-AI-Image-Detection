"""Typed, layered configuration.

Configuration is resolved in this order (later wins)::

    dataclass defaults  ->  YAML/JSON file  ->  AIDETECT__* env vars  ->  CLI --set

Every section is a plain dataclass so IDEs and type checkers understand it, and
:meth:`AppConfig.to_dict` / :meth:`AppConfig.from_dict` round-trip losslessly so
a run's exact configuration can be embedded in its result JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .exceptions import ConfigError

ENV_PREFIX = "AIDETECT__"

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass
class DetectorConfig:
    """Stage 1 — REAL/FAKE classifier.

    The architecture fields mirror the training script exactly; changing them
    would make trained checkpoints unloadable.
    """

    backbone: str = "vit_base_patch14_dinov2.lvd142m"
    image_size: int = 518
    freeze_blocks: int = 8
    num_classes: int = 2
    head_hidden_dim: int = 256
    head_dropout: float = 0.3
    checkpoint: str = "checkpoints_adv/adv_best_model.pt"
    fallback_checkpoints: List[str] = field(
        default_factory=lambda: ["checkpoints/best_model.pt"]
    )
    #: Class order produced by ``torchvision.datasets.ImageFolder`` (alphabetical).
    class_names: List[str] = field(default_factory=lambda: ["FAKE", "REAL"])
    fake_index: int = 0
    #: Probability of FAKE above which an image is reported as FAKE.
    decision_threshold: float = 0.5
    #: Logit temperature; >1 softens probabilities, <1 sharpens them.
    temperature: float = 1.0
    #: Average logits over the image and its horizontal mirror.
    tta_hflip: bool = False
    batch_size: int = 8
    timm_cache_dir: Optional[str] = None
    mean: List[float] = field(default_factory=lambda: list(IMAGENET_MEAN))
    std: List[float] = field(default_factory=lambda: list(IMAGENET_STD))

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ConfigError("detector.image_size must be positive")
        if not 0.0 < self.decision_threshold < 1.0:
            raise ConfigError("detector.decision_threshold must be in (0, 1)")
        if self.temperature <= 0:
            raise ConfigError("detector.temperature must be positive")
        if len(self.class_names) != self.num_classes:
            raise ConfigError("detector.class_names must have num_classes entries")
        if not 0 <= self.fake_index < self.num_classes:
            raise ConfigError("detector.fake_index out of range")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ConfigError("detector.mean/std must have three channels")

    @property
    def real_index(self) -> int:
        return 1 - self.fake_index if self.num_classes == 2 else self.fake_index


@dataclass
class ExplainConfig:
    """Saliency / localisation settings."""

    #: gradcam | gradcam++ | rollout | occlusion
    method: str = "gradcam"
    colormap: str = "turbo"
    overlay_alpha: float = 0.45
    #: Gamma applied to the CAM before colouring (<1 boosts weak activations).
    gamma: float = 1.0
    #: Relative activation above which a pixel counts as "hot".
    hot_threshold: float = 0.5
    #: Fallback percentile used when the threshold selects nothing.
    fallback_percentile: float = 90.0
    #: Maximum number of disjoint hot regions to extract.
    max_regions: int = 3
    #: Minimum fraction of the image a region must cover to be kept.
    min_region_fraction: float = 0.002
    #: Padding (pixels, at original resolution) added around each region.
    region_padding: int = 20
    #: Crops are grown to at least this size so downstream VLMs accept them.
    min_crop_size: int = 96
    #: Grid used for connected-component analysis (keeps labelling cheap).
    component_grid: int = 64
    occlusion_patch: int = 64
    occlusion_stride: int = 32
    #: Which visual artefacts to write to disk.
    save_outputs: List[str] = field(
        default_factory=lambda: ["overlay", "crop", "annotated"]
    )

    ALL_OUTPUTS = ("raw", "overlay", "crop", "annotated", "panel")
    METHODS = ("gradcam", "gradcam++", "rollout", "occlusion")

    def validate(self) -> None:
        if self.method not in self.METHODS:
            raise ConfigError(
                f"explain.method must be one of {self.METHODS}, got {self.method!r}"
            )
        unknown = sorted(set(self.save_outputs) - set(self.ALL_OUTPUTS))
        if unknown:
            raise ConfigError(
                f"explain.save_outputs contains unknown entries: {unknown}. "
                f"Valid values: {list(self.ALL_OUTPUTS)}"
            )
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ConfigError("explain.overlay_alpha must be within [0, 1]")
        if self.max_regions < 1:
            raise ConfigError("explain.max_regions must be >= 1")


@dataclass
class ArtifactConfig:
    """Stage 2 — zero-shot artifact classification."""

    model_name: str = "google/siglip-base-patch16-224"
    top_k: int = 5
    #: Average each descriptor over several prompt templates.
    prompt_ensemble: bool = True
    #: raw | minmax | softmax | zscore — how descriptor scores are reported.
    score_mode: str = "minmax"
    #: Softmax temperature used when ``score_mode == "softmax"``.
    softmax_temperature: float = 0.02
    text_batch_size: int = 16
    #: Directory for the on-disk text-embedding cache (``None`` disables it).
    embedding_cache_dir: Optional[str] = ".cache/aidetect/text_embeddings"
    #: Minimum normalised score for a descriptor to be reported at all.
    min_score: float = 0.0

    SCORE_MODES = ("raw", "minmax", "softmax", "zscore")

    def validate(self) -> None:
        if self.top_k < 1:
            raise ConfigError("artifacts.top_k must be >= 1")
        if self.score_mode not in self.SCORE_MODES:
            raise ConfigError(
                f"artifacts.score_mode must be one of {self.SCORE_MODES}, "
                f"got {self.score_mode!r}"
            )
        if self.softmax_temperature <= 0:
            raise ConfigError("artifacts.softmax_temperature must be positive")


@dataclass
class DescribeConfig:
    """Stage 3 — natural-language explanation."""

    #: rule_based | moondream | qwen2vl
    backend: str = "rule_based"
    max_new_tokens: int = 200
    #: Hard cap applied to the final text (competition submissions cap at 50).
    max_words: int = 50
    max_sentences: int = 3
    #: Fall back to the rule-based writer when a VLM backend fails.
    fallback_to_rules: bool = True
    moondream_revision: str = "2025-01-09"
    qwen_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    moondream_model_id: str = "vikhyatk/moondream2"
    #: Deterministic decoding by default so runs are reproducible.
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42

    BACKENDS = ("rule_based", "moondream", "qwen2vl")

    def validate(self) -> None:
        if self.backend not in self.BACKENDS:
            raise ConfigError(
                f"describe.backend must be one of {self.BACKENDS}, got {self.backend!r}"
            )
        if self.max_words < 5:
            raise ConfigError("describe.max_words must be >= 5")


@dataclass
class RuntimeConfig:
    """Device, precision and output locations."""

    #: auto | cuda | cpu | mps | cuda:N
    device: str = "auto"
    #: auto | float32 | float16 | bfloat16
    dtype: str = "auto"
    seed: int = 42
    output_dir: str = "outputs"
    log_level: str = "INFO"
    log_file: Optional[str] = None
    #: Analyse Stage 2/3 even when Stage 1 says the image is REAL.
    explain_real_images: bool = False
    #: Number of worker threads used by the batch runner for image IO.
    num_workers: int = 0

    def validate(self) -> None:
        if self.dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ConfigError(f"runtime.dtype invalid: {self.dtype!r}")


@dataclass
class AppConfig:
    """Root configuration object."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    describe: DescribeConfig = field(default_factory=DescribeConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # -- validation ------------------------------------------------------- #

    def validate(self) -> "AppConfig":
        for section in (self.detector, self.explain, self.artifacts, self.describe, self.runtime):
            section.validate()
        return self

    # -- (de)serialisation ------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_yaml(self) -> str:
        try:
            import yaml  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            return self.to_json()
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        cfg = cls()
        _apply_mapping(cfg, data)
        return cfg.validate()

    # -- layered loading -------------------------------------------------- #

    @classmethod
    def load(
        cls,
        path: Optional[str | Path] = None,
        overrides: Optional[Sequence[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        use_env: bool = True,
    ) -> "AppConfig":
        """Build a config from file + environment + ``key=value`` overrides."""
        cfg = cls()

        path = path or (env or os.environ).get("AIDETECT_CONFIG") if use_env else path
        if path:
            _apply_mapping(cfg, load_config_file(path))

        if use_env:
            _apply_mapping(cfg, _env_overrides(env or os.environ))

        for override in overrides or ():
            key, sep, value = override.partition("=")
            if not sep:
                raise ConfigError(
                    f"Override {override!r} is not in 'section.key=value' form"
                )
            set_by_path(cfg, key.strip(), _coerce_scalar(value.strip()))

        return cfg.validate()

    # -- convenience ------------------------------------------------------ #

    def get(self, dotted_key: str) -> Any:
        node: Any = self
        for part in dotted_key.split("."):
            if not is_dataclass(node) or not hasattr(node, part):
                raise ConfigError(f"Unknown configuration key: {dotted_key!r}")
            node = getattr(node, part)
        return node

    def copy_with(self, **overrides: Any) -> "AppConfig":
        """Return a deep copy with dotted ``key=value`` overrides applied."""
        clone = AppConfig.from_dict(self.to_dict())
        for key, value in overrides.items():
            set_by_path(clone, key.replace("__", "."), value)
        return clone.validate()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_config_file(path: str | Path) -> Dict[str, Any]:
    """Read a YAML or JSON config file into a plain dict."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise ConfigError(f"Config file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigError(
                "PyYAML is required to read YAML config files (pip install pyyaml)"
            ) from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")

    if not isinstance(data, dict):
        raise ConfigError(f"Config file {file_path} must contain a mapping at the top level")
    return data


def set_by_path(config: Any, dotted_key: str, value: Any) -> None:
    """Assign ``value`` to a dotted path such as ``detector.image_size``."""
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        if not is_dataclass(node) or not hasattr(node, part):
            raise ConfigError(f"Unknown configuration section: {dotted_key!r}")
        node = getattr(node, part)
    leaf = parts[-1]
    if not is_dataclass(node) or not hasattr(node, leaf):
        raise ConfigError(f"Unknown configuration key: {dotted_key!r}")
    setattr(node, leaf, _coerce_to_field_type(node, leaf, value))


def _apply_mapping(config: Any, data: Mapping[str, Any]) -> None:
    """Recursively apply a (possibly nested) mapping onto a dataclass tree."""
    for key, value in data.items():
        if not hasattr(config, key):
            raise ConfigError(
                f"Unknown configuration key: {key!r} "
                f"(valid: {sorted(f.name for f in fields(config))})"
            )
        current = getattr(config, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply_mapping(current, value)
        else:
            setattr(config, key, _coerce_to_field_type(config, key, value))


def _env_overrides(env: Mapping[str, str]) -> Dict[str, Any]:
    """Turn ``AIDETECT__detector__image_size=224`` into a nested dict."""
    nested: Dict[str, Any] = {}
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = raw_key[len(ENV_PREFIX) :].lower().split("__")
        cursor = nested
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce_scalar(raw_value)
    return nested


def _coerce_to_field_type(node: Any, name: str, value: Any) -> Any:
    """Best-effort coercion of ``value`` to the declared field type."""
    declared = {f.name: f.type for f in fields(node)}.get(name)
    current = getattr(node, name, None)

    if declared is None:
        return value

    type_text = declared if isinstance(declared, str) else getattr(declared, "__name__", str(declared))

    if isinstance(value, str) and "List" in type_text:
        # Accept comma-separated strings for list-valued options.
        value = [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(value, (list, tuple)) and "List" in type_text:
        if "float" in type_text:
            return [float(v) for v in value]
        if "int" in type_text:
            return [int(v) for v in value]
        return [str(v) for v in value]

    if value is None:
        return None

    if "bool" in type_text and not isinstance(value, bool):
        return _coerce_scalar(str(value)) if isinstance(value, str) else bool(value)
    if "Optional[int]" in type_text or type_text == "int":
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} expects an integer, got {value!r}") from exc
    if "Optional[float]" in type_text or type_text == "float":
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} expects a number, got {value!r}") from exc
    if "str" in type_text and not isinstance(value, str):
        return str(value)

    if isinstance(current, bool) and not isinstance(value, bool):
        return bool(value)
    return value


def _coerce_scalar(text: str) -> Any:
    """Parse a CLI/env scalar into bool/int/float/None/str."""
    lowered = text.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


DEFAULT_CONFIG_LOCATIONS = (
    "aidetect.yaml",
    "configs/default.yaml",
    "config.yaml",
)


def discover_config_file(start: Optional[str | Path] = None) -> Optional[Path]:
    """Return the first default config file that exists, if any."""
    base = Path(start or ".").resolve()
    for candidate in DEFAULT_CONFIG_LOCATIONS:
        path = base / candidate
        if path.is_file():
            return path
    return None
