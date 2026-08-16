"""Qwen2-VL-2B backend (~4 GB, best description quality).

``qwen_vl_utils`` is optional: when it is missing the processor is fed the PIL
image directly, which works for single-image prompts. Very small crops are
upscaled first because Qwen2-VL's vision tower rejects images under 28px.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ..exceptions import BackendUnavailableError
from ..logging_utils import get_logger
from .base import DescriptionBackend, register_backend

logger = get_logger(__name__)

#: Qwen2-VL's patch grid requires at least this many pixels per side.
MIN_SIDE = 32


@register_backend("qwen2vl")
class Qwen2VLBackend(DescriptionBackend):
    """Instruction-tuned VLM producing the most specific descriptions."""

    name = "qwen2vl"
    is_visual = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model: Any = None
        self.processor: Any = None

    @classmethod
    def is_available(cls) -> bool:
        if importlib.util.find_spec("transformers") is None:
            return False
        try:
            from transformers import Qwen2VLForConditionalGeneration  # noqa: F401
        except (ImportError, AttributeError):
            return False
        return True

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model_id = self.cfg.qwen_model_id
        logger.info("Loading %s on %s", model_id, self.device)

        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            ).to(self.device)
        except OSError as exc:
            raise BackendUnavailableError(
                self.name,
                f"Weights for {model_id} are not cached and could not be downloaded. "
                f"Pre-download with: huggingface-cli download {model_id}",
            ) from exc

        self.model.eval()

    @staticmethod
    def _ensure_min_size(image):
        """Upscale crops that are too small for the vision encoder."""
        from PIL import Image

        width, height = image.size
        if width >= MIN_SIDE and height >= MIN_SIDE:
            return image
        scale = max(MIN_SIDE / max(width, 1), MIN_SIDE / max(height, 1))
        new_size = (max(MIN_SIDE, int(width * scale)), max(MIN_SIDE, int(height * scale)))
        logger.debug("Upscaling %sx%s crop to %s for Qwen2-VL", width, height, new_size)
        return image.resize(new_size, Image.LANCZOS)

    def generate(self, image, prompt: str, **kwargs: Any) -> str:
        import torch

        self.load()
        if image is None:
            raise BackendUnavailableError(self.name, "This backend requires an image.")
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self._ensure_min_size(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        images = [image]
        videos = None
        if importlib.util.find_spec("qwen_vl_utils") is not None:
            from qwen_vl_utils import process_vision_info

            images, videos = process_vision_info(messages)

        inputs = self.processor(
            text=[text_input],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.cfg.max_new_tokens),
            "do_sample": self.cfg.do_sample,
        }
        if self.cfg.do_sample:
            generation_kwargs.update(
                temperature=self.cfg.temperature, top_p=self.cfg.top_p
            )

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        generated = output_ids[:, inputs.input_ids.shape[1] :]
        decoded = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return decoded[0].strip() if decoded else ""

    def unload(self) -> None:
        self.model = None
        self.processor = None
        super().unload()
