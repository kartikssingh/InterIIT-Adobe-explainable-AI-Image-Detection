"""Stage 1 model definition and checkpoint loading.

IMPORTANT — the network architecture below is byte-for-byte equivalent to the one
used by ``training/adversarialTraining.py`` and ``training/Dino.ipynb``:

    DINOv2 ViT-B/14 backbone (``global_pool="token"``, ``num_classes=0``)
      -> Linear(embed_dim, 256) -> GELU -> Dropout(0.3) -> Linear(256, 2)

Do not change the layer order, sizes or names: existing checkpoints
(``checkpoints/best_model.pt``, ``checkpoints_adv/adv_best_model.pt``) would stop
loading. Everything around the architecture (loading, validation, metadata,
device placement) is what this module improves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..config import DetectorConfig
from ..exceptions import CheckpointError
from ..logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Architecture (unchanged)
# --------------------------------------------------------------------------- #


class DINOv2Classifier(nn.Module):
    """DINOv2 backbone + MLP head. Identical to the training definition."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled backbone embedding — useful for retrieval / debugging."""
        return self.backbone(x)


def build_head(embed_dim: int, cfg: DetectorConfig) -> nn.Sequential:
    """Classification head — must match training exactly."""
    return nn.Sequential(
        nn.Linear(embed_dim, cfg.head_hidden_dim),
        nn.GELU(),
        nn.Dropout(cfg.head_dropout),
        nn.Linear(cfg.head_hidden_dim, cfg.num_classes),
    )


def build_backbone(cfg: DetectorConfig) -> nn.Module:
    """Create the timm backbone with the same freezing scheme used in training.

    ``pretrained=False`` because our own checkpoint supplies every weight — this
    keeps inference fully offline and avoids a multi-hundred-MB download.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - hard dependency
        raise CheckpointError(
            "timm is required to build the detector backbone (pip install timm)"
        ) from exc

    kwargs: Dict[str, Any] = {
        "pretrained": False,
        "num_classes": 0,
        "global_pool": "token",
    }
    if cfg.timm_cache_dir:
        kwargs["cache_dir"] = cfg.timm_cache_dir

    backbone = timm.create_model(cfg.backbone, **kwargs)

    # Freezing does not affect inference numerics, but keeping it identical to
    # training means `requires_grad` layouts (and therefore optimiser state
    # dicts) stay compatible if a checkpoint is ever resumed from here.
    for name, param in backbone.named_parameters():
        if any(key in name for key in ("patch_embed", "pos_embed", "cls_token")):
            param.requires_grad = False
    blocks = getattr(backbone, "blocks", [])
    for index, block in enumerate(blocks):
        if index < cfg.freeze_blocks:
            for param in block.parameters():
                param.requires_grad = False

    return backbone


def build_model(cfg: DetectorConfig, device: str | torch.device = "cpu") -> DINOv2Classifier:
    """Assemble an (untrained) classifier on the requested device."""
    backbone = build_backbone(cfg)
    embed_dim = int(getattr(backbone, "num_features", 768))
    model = DINOv2Classifier(backbone, build_head(embed_dim, cfg))
    return model.to(device)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


@dataclass
class CheckpointInfo:
    """Metadata recovered from a checkpoint file."""

    path: str
    epoch: Optional[int] = None
    clean_acc: Optional[float] = None
    adv_acc: Optional[float] = None
    baseline_clean_acc: Optional[float] = None
    baseline_adv_acc: Optional[float] = None
    kind: str = "unknown"  # adversarial | original | unknown
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "epoch": self.epoch,
            "clean_acc": self.clean_acc,
            "adv_acc": self.adv_acc,
            "baseline_clean_acc": self.baseline_clean_acc,
            "baseline_adv_acc": self.baseline_adv_acc,
            "size_mb": self.size_mb,
            "missing_keys": self.missing_keys[:10],
            "unexpected_keys": self.unexpected_keys[:10],
        }

    def summary(self) -> str:
        bits = [f"{self.kind} checkpoint"]
        if self.epoch is not None:
            bits.append(f"epoch {self.epoch}")
        if self.clean_acc is not None:
            bits.append(f"clean {self.clean_acc:.4f}")
        if self.adv_acc is not None:
            bits.append(f"PGD {self.adv_acc:.4f}")
        return ", ".join(bits)


def infer_checkpoint_kind(path: str | Path) -> str:
    """Guess whether a checkpoint is the adversarial or the original model."""
    text = str(path).lower()
    if "adv" in text:
        return "adversarial"
    if "best_model" in text or "checkpoints/" in text:
        return "original"
    return "unknown"


def _torch_load(path: Path, device: str | torch.device) -> Any:
    """Load a checkpoint, preferring the safe ``weights_only`` code path.

    ``weights_only=True`` (torch >= 2.0) refuses to execute arbitrary pickle
    payloads. Our training script stores a plain ``cfg`` dict alongside the
    tensors, which some torch versions reject under that flag — hence the
    explicit, logged fallback rather than silently unpickling everything.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # torch < 2.0 has no weights_only parameter.
        return torch.load(path, map_location=device)
    except Exception as exc:
        logger.debug("weights_only load failed (%s); retrying in legacy mode", exc)
        logger.warning(
            "Loading %s with weights_only=False — only do this for checkpoints you trust",
            path,
        )
        return torch.load(path, map_location=device, weights_only=False)


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Support every checkpoint layout this project has produced."""
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise CheckpointError("Unrecognised checkpoint format: no state dict found")


def strip_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop ``module.`` / ``_orig_mod.`` prefixes left by DDP or ``torch.compile``."""
    prefixes = ("module.", "_orig_mod.")
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def resolve_checkpoint_path(cfg: DetectorConfig, override: Optional[str] = None) -> Path:
    """Return the first checkpoint that exists, honouring the fallback list."""
    candidates = [override] if override else []
    candidates.append(cfg.checkpoint)
    candidates.extend(cfg.fallback_checkpoints)

    tried: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        tried.append(str(path))
        if path.is_file():
            if str(path) != str(Path(candidates[0]).expanduser()):
                logger.warning("Using fallback checkpoint %s", path)
            return path

    raise CheckpointError(
        "No detector checkpoint found. Tried:\n  "
        + "\n  ".join(tried)
        + "\nTrain a model first, or point --checkpoint at an existing .pt file."
    )


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    cfg: Optional[DetectorConfig] = None,
    strict: bool = True,
) -> tuple[DINOv2Classifier, CheckpointInfo]:
    """Build the architecture and restore trained weights.

    Returns the model in ``eval()`` mode together with the checkpoint metadata.
    """
    cfg = cfg or DetectorConfig()
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")

    model = build_model(cfg, device)
    checkpoint = _torch_load(path, device)
    state = strip_prefixes(extract_state_dict(checkpoint))

    try:
        result = model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        raise CheckpointError(
            f"Checkpoint {path} does not match the model architecture.\n"
            f"Expected backbone={cfg.backbone!r} image_size={cfg.image_size} "
            f"head_hidden_dim={cfg.head_hidden_dim}.\nTorch said: {exc}"
        ) from exc

    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing:
        logger.warning("Checkpoint is missing %d keys (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("Checkpoint has %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])

    meta = checkpoint if isinstance(checkpoint, dict) else {}
    info = CheckpointInfo(
        path=str(path),
        epoch=_maybe_int(meta.get("epoch")),
        clean_acc=_maybe_float(meta.get("clean_acc")),
        adv_acc=_maybe_float(meta.get("adv_acc")),
        baseline_clean_acc=_maybe_float(meta.get("baseline_clean_acc")),
        baseline_adv_acc=_maybe_float(meta.get("baseline_adv_acc")),
        kind=infer_checkpoint_kind(path),
        missing_keys=missing,
        unexpected_keys=unexpected,
        size_mb=round(path.stat().st_size / 1024**2, 1),
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    logger.info("Loaded %s (%.1f MB) — %s", path, info.size_mb, info.summary())
    return model, info


def _maybe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
