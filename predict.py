"""
predict.py  —  Inference + GradCAM for AI-Generated Image Detection
====================================================================
Loads the adversarially trained model (checkpoints_adv/adv_best_model.pt),
predicts REAL or FAKE, and if FAKE runs GradCAM to localize artifact regions.

Outputs saved to current directory:
    gradcam_overlay_<filename>.png   — heatmap overlaid on original image
    gradcam_crop_<filename>.png      — tightest crop around the hottest region
    gradcam_raw_<filename>.png       — raw heatmap without overlay

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --model checkpoints_adv/adv_best_model.pt
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image, ImageDraw
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match exactly what was used during adversarial training
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    "backbone":      "vit_base_patch14_dinov2.lvd142m",
    "image_size":    518,
    "freeze_blocks": 8,
    "model_path":    "checkpoints_adv/adv_best_model.pt",  # Updated to adversarial model
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL DEFINITION  (identical to adversarialTraining.py)
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def load_model(model_path: str, device: torch.device) -> DINOv2Classifier:
    """Load backbone + head and restore trained weights (adversarial model)."""
    backbone = timm.create_model(
        CFG["backbone"],
        pretrained=False,          # no download needed — we load our own weights
        num_classes=0,
        global_pool="token",
        cache_dir="/home/kartik/.cache/torch",
    )

    # Freeze same blocks as training (must match or weight shapes won't align)
    for name, param in backbone.named_parameters():
        if any(k in name for k in ["patch_embed", "pos_embed", "cls_token"]):
            param.requires_grad = False
    for i, block in enumerate(backbone.blocks):
        if i < CFG["freeze_blocks"]:
            for param in block.parameters():
                param.requires_grad = False

    embed_dim = backbone.num_features   # 768
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )

    model = DINOv2Classifier(backbone, head).to(device)

    ckpt = torch.load(model_path, map_location=device)
    
    # Handle both training script formats
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        epoch = ckpt.get('epoch', '?')
        clean_acc = ckpt.get('clean_acc', '?')
        adv_acc = ckpt.get('adv_acc', '?')
    else:
        model.load_state_dict(ckpt)
        epoch = '?'
        clean_acc = '?'
        adv_acc = '?'
    
    model.eval()

    print(f"  Loaded model from  : {model_path}")
    print(f"  Checkpoint epoch   : {epoch}")
    print(f"  Clean accuracy     : {clean_acc if clean_acc == '?' else f'{clean_acc:.4f}'}")
    print(f"  Adversarial acc    : {adv_acc if adv_acc == '?' else f'{adv_acc:.4f}'}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> tuple:
    """
    Returns:
        tensor  : (1, 3, H, W) normalised tensor ready for the model
        pil_img : original PIL image (for overlay drawing)
    """
    transform = transforms.Compose([
        transforms.Resize((CFG["image_size"], CFG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    with open(image_path, "rb") as f:
        pil_img = Image.open(f).convert("RGB")

    # Keep a resized copy at model resolution for overlays
    pil_resized = pil_img.resize(
        (CFG["image_size"], CFG["image_size"]), Image.LANCZOS
    )
    tensor = transform(pil_img).unsqueeze(0)   # (1, 3, H, W)
    return tensor, pil_resized, pil_img


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRADCAM FOR VISION TRANSFORMERS
# ─────────────────────────────────────────────────────────────────────────────

class ViTGradCAM:
    """
    GradCAM adapted for Vision Transformers (DINOv2).

    In a ViT, there are no conv feature maps. Instead we use the attention
    output of the LAST transformer block as our "feature map".

    The spatial tokens (everything except the [CLS] token) form a
    (H_patches × W_patches) grid — this is our spatial map.

    Algorithm:
        1. Forward pass, register hook on last attention block output
        2. Backward pass on the FAKE class logit
        3. Global average pool the gradients (channel importance weights)
        4. Weight the feature map channels by their importance
        5. ReLU → normalise → upsample
    """

    def __init__(self, model: DINOv2Classifier):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._hooks     = []

    def _register_hooks(self):
        """Hook into the last transformer block's output."""
        last_block = self.model.backbone.blocks[-1]

        def forward_hook(module, input, output):
            # output shape: (B, num_tokens, embed_dim)
            # num_tokens = 1 (CLS) + num_patches (spatial)
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(last_block.register_forward_hook(forward_hook))
        self._hooks.append(last_block.register_full_backward_hook(backward_hook))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        tensor: torch.Tensor,
        device: torch.device,
        target_class: int = 0,   # 0 = FAKE (FAKE folder comes first alphabetically)
    ) -> np.ndarray:
        """
        Returns a (H, W) numpy array in [0, 1] — the GradCAM heatmap
        at the same spatial resolution as the input tensor.
        """
        self._register_hooks()
        tensor = tensor.to(device).requires_grad_(False)

        # ── Forward ─────────────────────────────────────────────────────────
        self.model.zero_grad()
        with torch.enable_grad():
            tensor_grad = tensor.clone().requires_grad_(True)
            logits = self.model(tensor_grad)              # (1, 2)

            # Score of the target class (FAKE)
            score = logits[0, target_class]
            score.backward()

        self._remove_hooks()

        # ── Extract spatial tokens ───────────────────────────────────────────
        # activations: (1, num_tokens, 768) — first token is CLS, rest are spatial
        spatial_acts  = self.activations[0, 1:, :]    # (num_patches, 768)
        spatial_grads = self.gradients[0, 1:, :]      # (num_patches, 768)

        # ── Compute importance weights (global avg pool over patches) ────────
        weights = spatial_grads.mean(dim=0)            # (768,)

        # ── Weighted combination of activation channels ──────────────────────
        cam = (weights.unsqueeze(0) * spatial_acts).sum(dim=-1)   # (num_patches,)
        cam = F.relu(cam)                              # only positive contributions

        # ── Reshape to 2D spatial grid ───────────────────────────────────────
        patch_size   = self.model.backbone.patch_embed.patch_size
        ps = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        H_patches = CFG["image_size"] // ps
        W_patches = CFG["image_size"] // ps

        cam = cam.reshape(H_patches, W_patches)        # (16, 16) for 224/14

        # ── Upsample to image resolution ─────────────────────────────────────
        cam = cam.unsqueeze(0).unsqueeze(0)            # (1, 1, 16, 16)
        cam = F.interpolate(
            cam,
            size=(CFG["image_size"], CFG["image_size"]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()              # (224, 224)

        # ── Normalise to [0, 1] ──────────────────────────────────────────────
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def heatmap_to_rgb(cam: np.ndarray) -> np.ndarray:
    """
    Convert a [0,1] grayscale heatmap to a BGR-style jet colormap RGB array.
    Uses only numpy — no opencv or matplotlib dependency.
    """
    # Jet colormap approximation using numpy
    # R channel
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    # G channel
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    # B channel
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)

    rgb = np.stack([r, g, b], axis=-1)   # (H, W, 3) in [0, 1]
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(
    pil_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Blend the jet heatmap onto the original image.
    alpha = opacity of the heatmap layer (0=invisible, 1=full heatmap).
    """
    heatmap_rgb = heatmap_to_rgb(cam)
    heatmap_pil = Image.fromarray(heatmap_rgb).resize(
        pil_img.size, Image.LANCZOS
    )
    # Blend: result = (1-alpha)*original + alpha*heatmap
    overlay = Image.blend(pil_img, heatmap_pil, alpha=alpha)
    return overlay


def get_hottest_crop(
    pil_img: Image.Image,
    cam: np.ndarray,
    threshold: float = 0.5,
    padding: int = 20,
) -> tuple:
    H, W = cam.shape
    hot_mask = cam > threshold

    if hot_mask.sum() == 0:
        # fallback 1: top 10% hottest pixels
        flat_threshold = np.percentile(cam, 90)
        hot_mask = cam > flat_threshold

    if hot_mask.sum() == 0:
        # fallback 2: just use the single hottest point and make a box around it
        flat_idx = np.argmax(cam)
        peak_row = flat_idx // W
        peak_col = flat_idx  % W
        box_size = 50
        rows = np.array([max(0, peak_row - box_size), min(H-1, peak_row + box_size)])
        cols = np.array([max(0, peak_col - box_size), min(W-1, peak_col + box_size)])
    else:
        rows = np.where(hot_mask.any(axis=1))[0]
        cols = np.where(hot_mask.any(axis=0))[0]

    y1, y2 = int(rows.min()), int(rows.max())
    x1, x2 = int(cols.min()), int(cols.max())

    # Scale from cam resolution to original image resolution
    orig_W, orig_H = pil_img.size
    scale_x = orig_W / W
    scale_y = orig_H / H

    x1 = max(0, int(x1 * scale_x) - padding)
    y1 = max(0, int(y1 * scale_y) - padding)
    x2 = min(orig_W, int(x2 * scale_x) + padding)
    y2 = min(orig_H, int(y2 * scale_y) + padding)

    crop = pil_img.crop((x1, y1, x2, y2))
    return crop, (x1, y1, x2, y2)


def draw_bbox_on_image(
    pil_img: Image.Image,
    bbox: tuple,
    color: tuple = (255, 50, 50),
    thickness: int = 3,
) -> Image.Image:
    """Draw a rectangle on the image marking the artifact region."""
    import struct

    # Pure PIL drawing without external deps
    img_array = np.array(pil_img.copy())
    x1, y1, x2, y2 = bbox

    for t in range(thickness):
        # Top edge
        img_array[max(0, y1-t), x1:x2] = color
        # Bottom edge
        img_array[min(img_array.shape[0]-1, y2+t), x1:x2] = color
        # Left edge
        img_array[y1:y2, max(0, x1-t)] = color
        # Right edge
        img_array[y1:y2, min(img_array.shape[1]-1, x2+t)] = color

    return Image.fromarray(img_array)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDICTION + ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    image_path: str,
    model: DINOv2Classifier,
    device: torch.device,
) -> dict:
    """
    Full pipeline:
        1. Preprocess image
        2. Forward pass → class + confidence
        3. If FAKE → GradCAM → overlay + crop + bbox
        4. Save all outputs
        5. Return result dict
    """
    image_path = Path(image_path)
    stem       = image_path.stem        # filename without extension

    print(f"\n{'='*55}")
    print(f"  Image : {image_path}")
    print(f"{'='*55}")

    # ── Preprocess ───────────────────────────────────────────────────────────
    tensor, pil_resized, pil_original = preprocess(str(image_path))

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor.to(device))          # (1, 2)
        probs  = torch.softmax(logits, dim=1)[0]   # (2,)

    # class_to_idx depends on alphabetical sort of folder names
    # ImageFolder sorts alphabetically: FAKE=0, REAL=1
    fake_prob = probs[0].item()
    real_prob = probs[1].item()
    predicted_class = "FAKE" if fake_prob > real_prob else "REAL"
    confidence      = max(fake_prob, real_prob) * 100

    print(f"\n  ┌─ PREDICTION ────────────────────────────────┐")
    print(f"  │  Result     : {'🔴 FAKE (AI-generated)' if predicted_class == 'FAKE' else '🟢 REAL (authentic)'}")
    print(f"  │  Confidence : {confidence:.2f}%")
    print(f"  │  FAKE prob  : {fake_prob*100:.2f}%")
    print(f"  │  REAL prob  : {real_prob*100:.2f}%")
    print(f"  └─────────────────────────────────────────────┘")

    result = {
        "image_path":    str(image_path),
        "prediction":    predicted_class,
        "confidence":    confidence,
        "fake_prob":     fake_prob,
        "real_prob":     real_prob,
        "saved_files":   [],
    }

    # ── GradCAM — always run, most useful when FAKE ──────────────────────────
    print(f"\n  Running GradCAM {'(image is FAKE — artifact localisation)' if predicted_class == 'FAKE' else '(image is REAL — confirming clean regions)'}")

    gradcam  = ViTGradCAM(model)
    cam      = gradcam.generate(tensor, device, target_class=0)   # 0=FAKE

    # ── Save: raw heatmap ────────────────────────────────────────────────────
    raw_heatmap_rgb  = heatmap_to_rgb(cam)
    raw_heatmap_pil  = Image.fromarray(raw_heatmap_rgb).resize(
        pil_original.size, Image.LANCZOS
    )
    raw_path = f"gradcam_raw_{stem}.png"
    raw_heatmap_pil.save(raw_path)
    result["saved_files"].append(raw_path)
    print(f"  Saved raw heatmap  → {raw_path}")

    # ── Save: overlay on original image ─────────────────────────────────────
    # Resize cam to original image dimensions for overlay
    cam_for_overlay = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            pil_original.size, Image.LANCZOS
        )
    ) / 255.0

    overlay          = overlay_heatmap(pil_original, cam_for_overlay, alpha=0.45)
    overlay_path     = f"gradcam_overlay_{stem}.png"
    overlay.save(overlay_path)
    result["saved_files"].append(overlay_path)
    print(f"  Saved overlay      → {overlay_path}")

    # ── Save: tight crop of hottest artifact region ──────────────────────────
    crop, bbox = get_hottest_crop(pil_original, cam, threshold=0.5, padding=20)
    crop_path  = f"gradcam_crop_{stem}.png"
    crop.save(crop_path)
    result["saved_files"].append(crop_path)
    print(f"  Saved artifact crop→ {crop_path}")
    print(f"  Hottest region bbox: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")

    # ── Save: original with bounding box drawn ───────────────────────────────
    bbox_color  = (220, 50, 50) if predicted_class == "FAKE" else (50, 200, 50)
    annotated   = draw_bbox_on_image(pil_original, bbox, color=bbox_color, thickness=3)

    # Add a simple text banner at the top (pure PIL)
    banner_h    = 40
    banner      = Image.new("RGB", (annotated.width, banner_h), color=(30, 30, 30))
    annotated_with_banner = Image.new(
        "RGB", (annotated.width, annotated.height + banner_h)
    )
    annotated_with_banner.paste(banner, (0, 0))
    annotated_with_banner.paste(annotated, (0, banner_h))

    # Write text onto banner using PIL's built-in font
    draw       = ImageDraw.Draw(annotated_with_banner)
    label_text = f"{predicted_class}  |  {confidence:.1f}% confidence"
    text_color = (220, 80, 80) if predicted_class == "FAKE" else (80, 220, 80)
    draw.text((10, 12), label_text, fill=text_color)

    annotated_path = f"gradcam_annotated_{stem}.png"
    annotated_with_banner.save(annotated_path)
    result["saved_files"].append(annotated_path)
    print(f"  Saved annotated    → {annotated_path}")

    # ── Task 2 preparation info ───────────────────────────────────────────────
    if predicted_class == "FAKE":
        # Compute hotspot statistics for Task 2 pipeline
        cam_resized_orig = cam_for_overlay
        hotspot_intensity = float(cam_resized_orig.mean())
        peak_intensity    = float(cam_resized_orig.max())

        # Relative position of the hottest point (where the artifact is)
        flat_idx   = np.argmax(cam)
        peak_row   = flat_idx // cam.shape[1]
        peak_col   = flat_idx  % cam.shape[1]
        peak_row_pct = peak_row / cam.shape[0] * 100
        peak_col_pct = peak_col / cam.shape[1] * 100

        print(f"\n  ┌─ TASK 2 INFO (for artifact explanation) ────┐")
        print(f"  │  Crop saved at    : {crop_path}")
        print(f"  │  Hotspot peak     : ({peak_col_pct:.0f}% from left, {peak_row_pct:.0f}% from top)")
        print(f"  │  Mean activation  : {hotspot_intensity:.3f}")
        print(f"  │  Peak activation  : {peak_intensity:.3f}")
        print(f"  │")
        print(f"  │  Next step: pass '{crop_path}' into CLIP zero-shot")
        print(f"  │  to identify which of the 70 artifact types")
        print(f"  │  are most likely present in this region.")
        print(f"  └─────────────────────────────────────────────┘")

        result["task2"] = {
            "crop_path":       crop_path,
            "hotspot_bbox":    bbox,
            "peak_location":   (peak_col_pct, peak_row_pct),
            "mean_activation": hotspot_intensity,
            "peak_activation": peak_intensity,
        }

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n  ┌─ FILES SAVED ───────────────────────────────┐")
    for f in result["saved_files"]:
        print(f"  │  {f}")
    print(f"  └─────────────────────────────────────────────┘\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict REAL/FAKE + GradCAM artifact localisation (Adversarially Trained Model)"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the image file to analyse"
    )
    parser.add_argument(
        "--model", type=str, default=CFG["model_path"],
        help=f"Path to adversarially trained model (default: {CFG['model_path']})"
    )
    parser.add_argument(
        "--model_type", type=str, default="adversarial", choices=["adversarial", "original"],
        help="Which model to use: 'adversarial' (default) or 'original'"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print(f"       Make sure you have trained the model first.")
        sys.exit(1)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    CFG["model_path"] = args.model
    model = load_model(args.model, device)

    # ── Run prediction ────────────────────────────────────────────────────────
    result = predict(args.image, model, device)

    # ── Final one-line verdict ────────────────────────────────────────────────
    verdict = result["prediction"]
    conf    = result["confidence"]
    print("=" * 55)
    print(f"  VERDICT: This image is {verdict}  ({conf:.1f}% confidence)")
    if verdict == "FAKE":
        print(f"  🔍 Check gradcam_crop_{Path(args.image).stem}.png for artifact localization")
    print("=" * 55)