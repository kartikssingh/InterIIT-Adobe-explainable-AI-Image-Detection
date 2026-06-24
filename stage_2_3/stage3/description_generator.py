"""
Stage 3 — Description Generation
====================================
Takes cropped image + artifact labels from Stage 2
and generates a natural language explanation of WHY the image looks AI-generated.

Models supported (in order of preference for speed/quality):
  1. Moondream2  (vikhyatk/moondream2)  — fast, lightweight, good quality
  2. Qwen2-VL-2B (Qwen/Qwen2-VL-2B-Instruct) — slightly heavier, better prompting

Input:  PIL Image crop + list of artifact dicts from Stage 2
Output: Natural language description string
"""

import torch
from PIL import Image
from typing import Union
import logging
import sys
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────
def build_prompt(
    artifact_results: list[dict],
    category_summary: dict = None,
) -> str:
    """
    Build a focused prompt that steers the VLM toward describing
    the specific artifacts identified in Stage 2.
    """
    print("\n  → Building Qwen2-VL prompt...")
    top_matches = artifact_results if isinstance(artifact_results, list) else artifact_results.get("top_matches", [])
    categories = (
        category_summary.get("detected_categories", [])
        if category_summary
        else list(dict.fromkeys([r["category"] for r in top_matches]))
    )

    # Format artifact list
    artifact_lines = "\n".join(
        f"  - {r['descriptor']} ({r['category']})"
        for r in top_matches[:5]
    )

    category_str = ", ".join(categories) if categories else "general artifacts"

    prompt = f"""You are analyzing a cropped region from a suspected AI-generated image.

The following artifacts have been automatically detected in this image region:
{artifact_lines}

These fall into the artifact categories: {category_str}.

Your task: Write 2-3 sentences describing exactly what you visually observe in this image that confirms it is AI-generated. 
Be specific about what you see. Reference the actual visual content. Do not use generic statements.
Focus on the detected artifact types listed above.
Start directly with your observation."""

    print(f"  ✓ Prompt built ({len(prompt)} characters)")
    return prompt


def build_prompt_moondream(artifact_results: list[dict], category_summary: dict = None) -> str:
    """Shorter prompt optimized for Moondream2's instruction format."""
    print("\n  → Building Moondream2 prompt...")
    top_matches = artifact_results if isinstance(artifact_results, list) else artifact_results.get("top_matches", [])
    categories = list(dict.fromkeys([r["category"] for r in top_matches]))[:3]
    descs = [r["descriptor"] for r in top_matches[:3]]

    desc_str = "; ".join(descs)
    cat_str = ", ".join(categories)

    prompt = (
        f"This image crop is from an AI-generated image. "
        f"Detected artifact types: {cat_str}. "
        f"Specific issues found: {desc_str}. "
        f"Describe in 2-3 sentences what you visually see that makes this look AI-generated. Be specific."
    )
    print(f"  ✓ Moondream2 prompt built ({len(prompt)} characters)")
    return prompt


# ─────────────────────────────────────────────
# Moondream2 Backend
# ─────────────────────────────────────────────
class Moondream2Backend:
    MODEL_ID = "vikhyatk/moondream2"

    def __init__(self, device: str = None, revision: str = "2025-01-09"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "mps":
            print("  ⚠️  Moondream2 is unstable on MPS; using CPU instead.")
            self.device = "cpu"
        self.revision = revision
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            print("\n  ✓ Moondream2 already loaded (using cached instance)")
            return
        
        print("\n" + "="*60)
        print("  LOADING MOONDREAM2 VLM")
        print("="*60)
        print(f"  Model: {self.MODEL_ID}")
        print(f"  Device: {self.device}")
        print(f"  Revision: {self.revision}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Check if model exists in cache
            cache_dir = os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
            model_cache_path = Path(cache_dir) / f"models--{self.MODEL_ID.replace('/', '--')}"
            
            if model_cache_path.exists():
                print(f"  ✓ Model found in cache: {model_cache_path}")
                print("  → Using cached model (no download needed)")
            else:
                print(f"  ⚠️  Model not found in cache")
                print("  → Downloading Moondream2 from Hugging Face...")
                print("  → This may take a few minutes (~1GB)")
            
            print("  → Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID, revision=self.revision, trust_remote_code=True
            )
            print("  ✓ Tokenizer loaded")
            
            print("  → Loading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                revision=self.revision,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            print("  ✓ Model weights loaded")
            
            self.model.eval()
            print("  ✓ Model set to evaluation mode")
            
            # Show model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  ✓ Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
            print("  ✓ Moondream2 ready for inference")
            print("="*60)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Moondream2: {e}")

    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 200) -> str:
        print("\n  → Generating description with Moondream2...")
        print(f"  → Max tokens: {max_tokens}")
        
        self.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        try:
            print("  → Encoding image...")
            enc_image = self.model.encode_image(image)
            print("  ✓ Image encoded")
            
            print("  → Answering question...")
            answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
            
            if answer:
                print(f"  ✓ Description generated ({len(answer)} characters)")
            else:
                print("  ⚠️  Empty response from Moondream2")
            
            return answer.strip()
        except Exception as e:
            print(f"  ❌ Moondream2 generation failed: {e}")
            logger.error(f"Moondream2 generation failed: {e}")
            return ""


# ─────────────────────────────────────────────
# Qwen2-VL Backend
# ─────────────────────────────────────────────
class Qwen2VLBackend:
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            print("\n  ✓ Qwen2-VL already loaded (using cached instance)")
            return
        
        print("\n" + "="*60)
        print("  LOADING QWEN2-VL-2B VLM")
        print("="*60)
        print(f"  Model: {self.MODEL_ID}")
        print(f"  Device: {self.device}")
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            # Check if model exists in cache
            cache_dir = os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
            model_cache_path = Path(cache_dir) / f"models--{self.MODEL_ID.replace('/', '--')}"
            
            if model_cache_path.exists():
                print(f"  ✓ Model found in cache: {model_cache_path}")
                print("  → Using cached model (no download needed)")
            else:
                print(f"  ⚠️  Model not found in cache")
                print("  → Downloading Qwen2-VL-2B from Hugging Face...")
                print("  → This may take a few minutes (~4GB)")
            
            print("  → Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            print("  ✓ Processor loaded")
            
            print("  → Loading model weights...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            print("  ✓ Model weights loaded")
            
            self.model.eval()
            print("  ✓ Model set to evaluation mode")
            
            # Show model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  ✓ Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
            print("  ✓ Qwen2-VL-2B ready for inference")
            print("="*60)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen2-VL: {e}")

    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 200) -> str:
        print("\n  → Generating description with Qwen2-VL...")
        print(f"  → Max tokens: {max_tokens}")
        
        self.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        try:
            from qwen_vl_utils import process_vision_info

            print("  → Preparing messages...")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            print("  → Applying chat template...")
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            print("  → Processing vision info...")
            image_inputs, video_inputs = process_vision_info(messages)
            
            print("  → Processing inputs...")
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            print("  → Generating response...")
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            
            print("  → Decoding response...")
            generated = output_ids[:, inputs.input_ids.shape[1]:]
            text = self.processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            
            if text:
                print(f"  ✓ Description generated ({len(text)} characters)")
            else:
                print("  ⚠️  Empty response from Qwen2-VL")
            
            return text.strip()
            
        except ImportError:
            print("  ⚠️  qwen_vl_utils not found, using simple fallback")
            return self._generate_simple(image, prompt, max_tokens)
        except Exception as e:
            print(f"  ❌ Qwen2-VL generation failed: {e}")
            logger.error(f"Qwen2-VL generation failed: {e}")
            return ""

    def _generate_simple(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        """Simpler Qwen2-VL invocation without qwen_vl_utils."""
        print("\n  → Using simple Qwen2-VL generation (without qwen_vl_utils)...")
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        
        print("  → Applying chat template...")
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        print("  → Processing inputs...")
        inputs = self.processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
        ).to(self.device)
        
        print("  → Generating response...")
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        print("  → Decoding response...")
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        
        if text:
            print(f"  ✓ Description generated ({len(text)} characters)")
        else:
            print("  ⚠️  Empty response from Qwen2-VL")
        
        return text


def rule_based_description(artifact_results: list[dict]) -> str:
    """Generate a description using templates when no VLM is available."""
    print("\n  → Generating rule-based description...")
    
    top = artifact_results[:3]
    if not top:
        print("  ⚠️  No artifacts found, using generic description")
        return "This image region contains visual artifacts consistent with AI generation."

    artifact_names = [r["descriptor"] for r in top]
    primary = artifact_names[0]
    
    if len(artifact_names) == 1:
        description = (
            f"The image shows {primary.lower()}, where the visual structure appears "
            "inconsistent with a natural camera-captured scene."
        )
    else:
        others = ", ".join(artifact_names[1:])
        description = (
            f"The image shows {primary.lower()}, with additional signs of {others}. "
            "These visual inconsistencies suggest synthetic generation rather than a natural photograph."
        )
    
    print(f"  ✓ Rule-based description generated ({len(description)} characters)")
    return description


# ─────────────────────────────────────────────
# Main DescriptionGenerator class
# ─────────────────────────────────────────────
class DescriptionGenerator:
    """
    Generates natural language descriptions of AI image artifacts.

    Usage:
        gen = DescriptionGenerator(backend="moondream")
        text = gen.generate(crop_image, stage2_results)
    """

    BACKENDS = ("moondream", "qwen2vl", "rule_based")

    def __init__(
        self,
        backend: str = "moondream",
        device: str = None,
        max_tokens: int = 200,
    ):
        assert backend in self.BACKENDS, f"backend must be one of {self.BACKENDS}"
        self.backend_name = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tokens = max_tokens
        self._backend = None
        
        print("\n" + "="*60)
        print("  INITIALIZING DESCRIPTION GENERATOR")
        print("="*60)
        print(f"  Backend: {backend}")
        print(f"  Device: {self.device}")
        print(f"  Max tokens: {max_tokens}")

    def _init_backend(self):
        if self._backend is not None:
            return
        if self.backend_name == "moondream":
            print("  → Initializing Moondream2 backend...")
            self._backend = Moondream2Backend(device=self.device)
            print("  ✓ Moondream2 backend initialized")
        elif self.backend_name == "qwen2vl":
            print("  → Initializing Qwen2-VL backend...")
            self._backend = Qwen2VLBackend(device=self.device)
            print("  ✓ Qwen2-VL backend initialized")
        # rule_based needs no init

    def generate(
        self,
        image: Union[Image.Image, str],
        stage2_results: Union[list, dict],
        fallback_on_error: bool = True,
    ) -> str:
        """
        Generate a natural language description.

        Args:
            image:          PIL Image crop or path
            stage2_results: Output from Stage 2 (list of dicts or classify_with_category_summary dict)
            fallback_on_error: If VLM fails, fall back to rule-based description

        Returns:
            Description string
        """
        print("\n" + "="*60)
        print("  GENERATING DESCRIPTION")
        print("="*60)
        print(f"  Backend: {self.backend_name}")
        print(f"  Fallback on error: {fallback_on_error}")
        
        if isinstance(image, str):
            print(f"  Loading image from: {image}")
            image = Image.open(image).convert("RGB")
        
        print(f"  Image size: {image.size}")
        print(f"  Image mode: {image.mode}")

        # Normalise input format
        if isinstance(stage2_results, dict):
            top_matches = stage2_results.get("top_matches", [])
            category_summary = stage2_results
            print(f"  Using Stage 2 results (dict format)")
            print(f"  Top matches: {len(top_matches)}")
            print(f"  Categories: {len(category_summary.get('detected_categories', []))}")
        else:
            top_matches = stage2_results
            category_summary = None
            print(f"  Using Stage 2 results (list format)")
            print(f"  Top matches: {len(top_matches)}")

        if self.backend_name == "rule_based":
            print("  → Using rule-based backend")
            return rule_based_description(top_matches)

        self._init_backend()

        # Build prompt
        print(f"\n  → Building prompt for {self.backend_name}...")
        if self.backend_name == "moondream":
            prompt = build_prompt_moondream(top_matches, category_summary)
        else:
            prompt = build_prompt(top_matches, category_summary)
        
        print(f"  → Prompt preview: {prompt[:100]}...")

        try:
            description = self._backend.generate(image, prompt, self.max_tokens)
            if not description:
                raise ValueError("Empty response from VLM")
            
            print("\n  ✓ Description generated successfully!")
            print(f"  → Length: {len(description)} characters")
            print(f"  → Preview: {description[:150]}...")
            
            return description
            
        except Exception as e:
            print(f"\n  ❌ VLM generation failed: {e}")
            if fallback_on_error:
                print("  → Falling back to rule-based description...")
                return rule_based_description(top_matches)
            raise

    def generate_full_output(
        self,
        image: Union[Image.Image, str],
        stage2_results: Union[list, dict],
    ) -> dict:
        """
        Generate complete Task 2 output in the expected format.

        Returns:
            {
                "description": str,
                "detected_categories": list[str],
                "top_artifacts": list[dict],
                "category_scores": dict
            }
        """
        print("\n" + "="*60)
        print("  GENERATING FULL OUTPUT")
        print("="*60)
        
        if isinstance(image, str):
            print(f"  Loading image from: {image}")
            image = Image.open(image).convert("RGB")
        
        print(f"  Image size: {image.size}")

        # Normalise
        if isinstance(stage2_results, dict):
            top_matches = stage2_results.get("top_matches", [])
            detected_categories = stage2_results.get("detected_categories", [])
            category_scores = stage2_results.get("category_scores", {})
            print(f"  Using Stage 2 results (dict format)")
            print(f"  Categories: {len(detected_categories)}")
        else:
            top_matches = stage2_results
            detected_categories = list(dict.fromkeys([r["category"] for r in top_matches]))
            category_scores = {}
            print(f"  Using Stage 2 results (list format)")

        print(f"\n  → Generating description...")
        description = self.generate(image, stage2_results)

        result = {
            "description": description,
            "detected_categories": detected_categories,
            "top_artifacts": [
                {"descriptor": r["descriptor"], "category": r["category"], "score": r["score"]}
                for r in top_matches[:5]
            ],
            "category_scores": category_scores,
        }
        
        print("\n  ✓ Full output generated:")
        print(f"  → Description: {description[:100]}...")
        print(f"  → Categories: {', '.join(detected_categories)}")
        print(f"  → Top artifacts: {len(result['top_artifacts'])}")
        print("="*60)
        
        return result


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python description_generator.py <image_path> [moondream|qwen2vl|rule_based]")
        sys.exit(1)

    image_path = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "moondream"

    print("\n" + "="*60)
    print("  DESCRIPTION GENERATOR - STANDALONE TEST")
    print("="*60)
    print(f"  Image: {image_path}")
    print(f"  Backend: {backend}")

    # Dummy Stage 2 results for testing
    dummy_stage2 = [
        {"descriptor": "Artificial smoothness", "score": 0.87, "category": "Artificial smoothness"},
        {"descriptor": "Inconsistent object boundaries", "score": 0.81, "category": "Inconsistent object boundaries"},
        {"descriptor": "Inconsistent shadow directions", "score": 0.74, "category": "Inconsistent shadow directions"},
    ]

    gen = DescriptionGenerator(backend=backend)
    img = Image.open(image_path).convert("RGB")
    output = gen.generate_full_output(img, dummy_stage2)
    
    print("\n" + "="*60)
    print("  FINAL RESULT")
    print("="*60)
    print(json.dumps(output, indent=2))
    print("="*60)