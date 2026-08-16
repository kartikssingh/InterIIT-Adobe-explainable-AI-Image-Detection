"""Moondream2 backend (~1 GB, CPU-friendly).

Moondream changed its inference API between releases: newer revisions expose
``model.query(image, question)`` while older ones use
``encode_image`` + ``answer_question``. Both are supported here, newest first,
so upgrading the pinned revision does not break the pipeline.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ..exceptions import BackendUnavailableError
from ..logging_utils import get_logger
from .base import DescriptionBackend, register_backend

logger = get_logger(__name__)


@register_backend("moondream")
class Moondream2Backend(DescriptionBackend):
    """Small vision-language model used for fast, CPU-viable descriptions."""

    name = "moondream"
    is_visual = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model: Any = None
        self.tokenizer: Any = None
        # Moondream is unstable on Apple MPS; CPU is slower but correct.
        if self.device == "mps":
            logger.warning("Moondream2 is unreliable on MPS — using CPU instead")
            self.device = "cpu"

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("transformers") is not None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.cfg.moondream_model_id
        revision = self.cfg.moondream_revision
        logger.info("Loading %s (revision %s) on %s", model_id, revision, self.device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=revision, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            ).to(self.device)
        except OSError as exc:
            raise BackendUnavailableError(
                self.name,
                f"Weights for {model_id} are not cached and could not be downloaded. "
                f"Pre-download with: huggingface-cli download {model_id}",
            ) from exc

        self.model.eval()

    def generate(self, image, prompt: str, **kwargs: Any) -> str:
        self.load()
        if image is None:
            raise BackendUnavailableError(self.name, "This backend requires an image.")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Newer API first.
        query = getattr(self.model, "query", None)
        if callable(query):
            try:
                answer = query(image, prompt)
                if isinstance(answer, dict):
                    return str(answer.get("answer", "")).strip()
                return str(answer).strip()
            except (TypeError, AttributeError) as exc:
                logger.debug("moondream .query() unavailable (%s); using legacy API", exc)

        encoded = self.model.encode_image(image)
        return str(self.model.answer_question(encoded, prompt, self.tokenizer)).strip()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        super().unload()
