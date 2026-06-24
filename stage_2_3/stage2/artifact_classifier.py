"""
Stage 2 — Artifact Classification
===================================
Zero-shot classification using SigLIP against 70 descriptors.
Maps to Adobe's 5 official artifact categories.

Input:  PIL Image crop (from Stage 1 localization)
Output: List of (descriptor, score, adobe_category) tuples
"""

import torch
import numpy as np
from PIL import Image
from typing import Union
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# Official Adobe artifact list: 70 artifacts
OFFICIAL_ARTIFACTS = [
    "Inconsistent object boundaries",
    "Discontinuous surfaces",
    "Non-manifold geometries in rigid structures",
    "Floating or disconnected components",
    "Asymmetric features in naturally symmetric objects",
    "Misaligned bilateral elements in animal faces",
    "Irregular proportions in mechanical components",
    "Texture bleeding between adjacent regions",
    "Texture repetition patterns",
    "Over-smoothing of natural textures",
    "Artificial noise patterns in uniform surfaces",
    "Unrealistic specular highlights",
    "Inconsistent material properties",
    "Metallic surface artifacts",
    "Dental anomalies in mammals",
    "Anatomically incorrect paw structures",
    "Improper fur direction flows",
    "Unrealistic eye reflections",
    "Misshapen ears or appendages",
    "Impossible mechanical connections",
    "Inconsistent scale of mechanical parts",
    "Physically impossible structural elements",
    "Inconsistent shadow directions",
    "Multiple light source conflicts",
    "Missing ambient occlusion",
    "Incorrect reflection mapping",
    "Incorrect perspective rendering",
    "Scale inconsistencies within single objects",
    "Spatial relationship errors",
    "Depth perception anomalies",
    "Over-sharpening artifacts",
    "Aliasing along high-contrast edges",
    "Blurred boundaries in fine details",
    "Jagged edges in curved structures",
    "Random noise patterns in detailed areas",
    "Loss of fine detail in complex structures",
    "Artificial enhancement artifacts",
    "Incorrect wheel geometry",
    "Implausible aerodynamic structures",
    "Misaligned body panels",
    "Impossible mechanical joints",
    "Distorted window reflections",
    "Anatomically impossible joint configurations",
    "Unnatural pose artifacts",
    "Biological asymmetry errors",
    "Regular grid-like artifacts in textures",
    "Repeated element patterns",
    "Systematic color distribution anomalies",
    "Frequency domain signatures",
    "Color coherence breaks",
    "Unnatural color transitions",
    "Resolution inconsistencies within regions",
    "Unnatural Lighting Gradients",
    "Incorrect Skin Tones",
    "Fake depth of field",
    "Abruptly cut off objects",
    "Glow or light bleed around object boundaries",
    "Ghosting effects: Semi-transparent duplicates of elements",
    "Cinematization Effects",
    "Excessive sharpness in certain image regions",
    "Artificial smoothness",
    "Movie-poster like composition of ordinary scenes",
    "Dramatic lighting that defies natural physics",
    "Artificial depth of field in object presentation",
    "Unnaturally glossy surfaces",
    "Synthetic material appearance",
    "Multiple inconsistent shadow sources",
    "Exaggerated characteristic features",
    "Impossible foreshortening in animal bodies",
    "Scale inconsistencies within the same object class",
]

DESCRIPTORS: list[str] = OFFICIAL_ARTIFACTS
DESCRIPTOR_TO_CATEGORY: dict[str, str] = {
    artifact: artifact for artifact in OFFICIAL_ARTIFACTS
}


# ─────────────────────────────────────────────
# Cache Setup with Logging
# ─────────────────────────────────────────────
def setup_cache_directory():
    """
    Set up a local cache directory that doesn't try to use external drives.
    Returns the path to the cache directory.
    """
    print("\n" + "="*60)
    print("  SETTING UP MODEL CACHE")
    print("="*60)
    
    # Check if HF_HOME is already set
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        print(f"  ✓ Using HF_HOME from environment: {hf_home}")
        cache_dir = Path(hf_home)
    else:
        # Use local home directory cache
        cache_dir = Path.home() / '.cache' / 'huggingface'
        print(f"  ✓ Using default cache: {cache_dir}")
    
    # Try to create the cache directory
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Cache directory ready: {cache_dir}")
        return str(cache_dir)
    except (PermissionError, OSError) as e:
        print(f"  ⚠️  Permission error: {e}")
        # If we can't create the directory, use a fallback in the current directory
        fallback_dir = Path.cwd() / '.hf_cache'
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Using fallback cache: {fallback_dir}")
        return str(fallback_dir)


# Set up cache before any imports that might use it
os.environ.setdefault('HF_HOME', setup_cache_directory())


# ─────────────────────────────────────────────
# Main Classifier Class
# ─────────────────────────────────────────────
class ArtifactClassifier:
    """
    Zero-shot artifact classifier using SigLIP.

    Usage:
        classifier = ArtifactClassifier()
        results = classifier.classify(crop_image, top_k=5)
        # results -> [{"descriptor": ..., "score": ..., "category": ...}, ...]
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str = None,
        top_k: int = 5,
        use_cache: bool = True,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.use_cache = use_cache
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._text_embeddings = None  # cached
        
        print("\n" + "="*60)
        print("  INITIALIZING ARTIFACT CLASSIFIER")
        print("="*60)
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Cache enabled: {self.use_cache}")
        print(f"  Top-K: {self.top_k}")
        
        # Ensure cache is set up
        if self.use_cache:
            cache_dir = setup_cache_directory()
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            print(f"  ✓ Transformers cache set to: {cache_dir}")

    def load_model(self):
        """Lazy-load SigLIP model and processor."""
        if self.model is not None:
            print("\n  ✓ Model already loaded (using cached instance)")
            return
        
        print("\n" + "="*60)
        print("  LOADING SIGLIP MODEL")
        print("="*60)
        
        try:
            from transformers import AutoProcessor, AutoModel
            print(f"  Model: {self.model_name}")
            print(f"  Device: {self.device}")
            
            # Check if model exists in cache
            cache_path = Path(os.environ.get('TRANSFORMERS_CACHE', ''))
            model_cache_path = cache_path / f"models--{self.model_name.replace('/', '--')}"
            
            if model_cache_path.exists():
                print(f"  ✓ Model found in cache: {model_cache_path}")
                print("  → Using cached model (no download needed)")
            else:
                print(f"  ⚠️  Model not found in cache")
                print("  → Downloading SigLIP model from Hugging Face...")
                print("  → This may take a few minutes depending on your internet speed")
            
            # Try to load with cache
            try:
                print("  → Loading processor...")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                print("  ✓ Processor loaded")
                
                print("  → Loading model weights...")
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                print("  ✓ Model weights loaded")
                
            except (PermissionError, OSError) as e:
                print(f"  ⚠️  Permission error with cache: {e}")
                print("  → Retrying without cache...")
                self.processor = AutoProcessor.from_pretrained(self.model_name, use_cache=False)
                self.model = AutoModel.from_pretrained(self.model_name, use_cache=False).to(self.device)
                print("  ✓ Model loaded without cache")
            
            self.model.eval()
            print("  ✓ Model set to evaluation mode")
            print("  ✓ SigLIP model ready for inference")
            
            # Show model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  ✓ Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
            
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of text strings into normalized embeddings."""
        print(f"\n  → Encoding {len(texts)} text descriptors...")
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        print(f"  ✓ Text embeddings shape: {text_features.shape}")
        return text_features  # (N, D)

    def _encode_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        print("\n  → Encoding image...")
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(f"  ✓ Image embedding shape: {image_features.shape}")
        return image_features

    def _get_text_embeddings(self) -> torch.Tensor:
        """Cache descriptor embeddings (computed once per session)."""
        if self._text_embeddings is None:
            print("\n" + "="*60)
            print("  PRE-COMPUTING TEXT EMBEDDINGS")
            print("="*60)
            print(f"  Total descriptors: {len(DESCRIPTORS)}")
            print("  → Computing embeddings for all 70 artifact descriptors...")
            print("  → This happens once and is cached for future use")
            
            # Process in batches to avoid OOM
            batch_size = 16
            all_embs = []
            for i in range(0, len(DESCRIPTORS), batch_size):
                batch = DESCRIPTORS[i : i + batch_size]
                print(f"  → Processing batch {i//batch_size + 1}/{(len(DESCRIPTORS)-1)//batch_size + 1} ({len(batch)} descriptors)")
                embs = self._encode_texts(batch)
                all_embs.append(embs)
            self._text_embeddings = torch.cat(all_embs, dim=0)  # (70, D)
            print(f"  ✓ All text embeddings computed and cached")
            print(f"  ✓ Embeddings shape: {self._text_embeddings.shape}")
            print("="*60)
        else:
            print("\n  ✓ Using cached text embeddings (computed earlier in this session)")
        return self._text_embeddings

    def classify(
        self,
        image: Union[Image.Image, str],
        top_k: int = None,
        return_all: bool = False,
    ) -> list[dict]:
        """
        Classify artifacts in an image crop.

        Args:
            image: PIL Image or path to image file
            top_k: Number of top results to return (default: self.top_k)
            return_all: If True, return all 70 scores sorted

        Returns:
            List of dicts: [{"descriptor", "score", "category", "rank"}]
        """
        print("\n" + "="*60)
        print("  CLASSIFYING ARTIFACTS")
        print("="*60)
        
        self.load_model()
        k = top_k or self.top_k
        print(f"  Top-K: {k}")
        print(f"  Return all: {return_all}")

        if isinstance(image, str):
            print(f"  Loading image from: {image}")
            image = Image.open(image).convert("RGB")
        
        print(f"  Image size: {image.size}")
        print(f"  Image mode: {image.mode}")

        # Encode image
        image_emb = self._encode_image(image)  # (1, D)

        # Get cached text embeddings
        text_embs = self._get_text_embeddings()  # (70, D)

        # SigLIP uses sigmoid scores, not softmax
        # logit_scale is part of the model
        print("\n  → Computing similarity scores...")
        with torch.no_grad():
            logit_scale = self.model.logit_scale.exp()
            logits = (image_emb @ text_embs.T) * logit_scale  # (1, 70)
            scores = torch.sigmoid(logits).squeeze(0)  # (70,)

        scores_np = scores.cpu().numpy()
        ranked_indices = np.argsort(scores_np)[::-1]
        
        print(f"  ✓ Similarity scores computed")
        print(f"  → Top score: {scores_np[ranked_indices[0]]:.4f}")
        print(f"  → Bottom score: {scores_np[ranked_indices[-1]]:.4f}")

        results = []
        limit = len(DESCRIPTORS) if return_all else k
        print(f"\n  → Extracting top {limit} artifacts...")
        for rank, idx in enumerate(ranked_indices[:limit]):
            desc = DESCRIPTORS[idx]
            score = float(scores_np[idx])
            results.append(
                {
                    "rank": rank + 1,
                    "descriptor": desc,
                    "score": score,
                    "category": DESCRIPTOR_TO_CATEGORY[desc],
                }
            )
            if rank < 5:  # Show top 5 in logs
                print(f"    #{rank+1}: {desc[:50]}{'...' if len(desc) > 50 else ''} ({score:.4f})")

        print(f"\n  ✓ Classification complete")
        print(f"  ✓ Found {len(results)} artifacts")
        print("="*60)
        return results

    def classify_with_category_summary(
        self, image: Union[Image.Image, str], top_k: int = 5
    ) -> dict:
        """
        Returns both top descriptors AND a per-category summary.

        Returns:
            {
                "top_matches": [...],         # top_k descriptor matches
                "detected_categories": [...], # unique Adobe categories found
                "category_scores": {...}      # max score per category
            }
        """
        print("\n" + "="*60)
        print("  CLASSIFYING WITH CATEGORY SUMMARY")
        print("="*60)
        print(f"  Top-K for categories: {top_k}")
        
        all_results = self.classify(image, top_k=len(DESCRIPTORS), return_all=True)

        # Category-level aggregation (max score per category)
        print("\n  → Aggregating by category...")
        category_scores: dict[str, float] = {}
        for r in all_results:
            cat = r["category"]
            if cat not in category_scores or r["score"] > category_scores[cat]:
                category_scores[cat] = r["score"]
        print(f"  ✓ Found {len(category_scores)} unique categories")

        top_matches = all_results[:top_k]
        detected_categories = list(
            dict.fromkeys([r["category"] for r in top_matches])
        )  # preserve order, deduplicate
        
        print(f"\n  → Detected categories:")
        for cat in detected_categories:
            score = category_scores.get(cat, 0)
            print(f"    • {cat}: {score:.4f}")

        print("\n  → Top artifacts:")
        for i, match in enumerate(top_matches[:5]):
            print(f"    #{i+1}: {match['descriptor'][:60]}{'...' if len(match['descriptor']) > 60 else ''}")
            print(f"       Score: {match['score']:.4f}")
            print(f"       Category: {match['category']}")

        print("\n" + "="*60)
        print("  CLASSIFICATION COMPLETE")
        print("="*60)

        return {
            "top_matches": top_matches,
            "detected_categories": detected_categories,
            "category_scores": dict(
                sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            ),
        }


# ─────────────────────────────────────────────
# CLI / quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json

    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not image_path:
        print("Usage: python artifact_classifier.py <image_path>")
        sys.exit(1)

    print("\n" + "="*60)
    print("  ARTIFACT CLASSIFIER - STANDALONE TEST")
    print("="*60)
    print(f"  Image: {image_path}")
    
    logging.basicConfig(level=logging.INFO)
    clf = ArtifactClassifier()
    img = Image.open(image_path).convert("RGB")
    results = clf.classify_with_category_summary(img, top_k=5)
    
    print("\n" + "="*60)
    print("  FINAL RESULTS (JSON)")
    print("="*60)
    print(json.dumps(results, indent=2))