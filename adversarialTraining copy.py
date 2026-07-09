"""
adversarialTraining.py
=======================

PGD adversarial fine-tuning for the DINOv2 REAL/FAKE classifier.
"""

import os
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import timm


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

CFG_ADV = {
    "data_root": "stratified_dataset",
    "image_size": 518,
    "checkpoint_in": "checkpoints/best_model.pt",
    "checkpoint_out_dir": "checkpoints_adv",
    "backbone": "vit_base_patch14_dinov2.lvd142m",
    "freeze_blocks": 8,
    "lr_head": 2e-5,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-2,
    "epochs": 3,
    "micro_batch_size": 2,
    "accum_steps": 2,
    "num_workers": 0,
    "eps": 8.0 / 255.0,
    "alpha": 2.0 / 255.0,
    "pgd_steps": 10,
    "random_start": True,
    "adv_ratio": 0.5,
    "grad_checkpointing": True,
    "amp_dtype": "bf16",
    "seed": 42,
    "max_grad_norm": 1.0,
    "eval_max_batches": None,
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Repro + logging
# --------------------------------------------------------------------------- #

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("adv_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(out_dir, "adv_train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# --------------------------------------------------------------------------- #
# Dataset - handles stratified_dataset structure
# --------------------------------------------------------------------------- #

def build_dataloaders(cfg):
    train_tf = transforms.Compose([
        transforms.Resize((cfg["image_size"] + 32, cfg["image_size"] + 32)),
        transforms.RandomCrop(cfg["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    root = cfg["data_root"]
    train_dir = os.path.join(root, "train")
    
    # Check if we're using stratified_dataset (has val folders)
    val_in_dir = os.path.join(root, "val_in_distribution")
    val_out_dir = os.path.join(root, "val_out_of_distribution")
    
    if os.path.exists(val_in_dir) and os.path.exists(val_out_dir):
        # Use both val folders as test set (stratified_dataset)
        from torch.utils.data import ConcatDataset
        
        val_in_ds = datasets.ImageFolder(val_in_dir, eval_tf)
        val_out_ds = datasets.ImageFolder(val_out_dir, eval_tf)
        eval_ds = ConcatDataset([val_in_ds, val_out_ds])
        
        # Log info
        print(f"Using stratified_dataset: train={train_dir}, test=val_in_distribution + val_out_of_distribution")
        print(f"Train samples: {len(train_ds) if 'train_ds' in dir() else 'loading...'}")
    else:
        # Original dataset with test folder
        test_dir = os.path.join(root, "test")
        eval_ds = datasets.ImageFolder(test_dir, eval_tf)

    train_ds = datasets.ImageFolder(train_dir, train_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["micro_batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg["micro_batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    return train_ds, eval_ds, train_loader, eval_loader


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class DINOv2Classifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def build_model(cfg, device, log):
    backbone = timm.create_model(
        cfg["backbone"],
        pretrained=False,
        num_classes=0,
        global_pool="token",
        cache_dir=os.environ.get("TORCH_HOME", "/home/kartik/.cache/torch"),
    )

    # Enable memory-efficient attention
    if hasattr(backbone, "set_attn_impl"):
        try:
            backbone.set_attn_impl("flash_attention")
            log.info("Flash attention enabled")
        except Exception as e:
            log.warning(f"Could not enable flash attention: {e}")

    # Freeze layers
    for name, param in backbone.named_parameters():
        if "patch_embed" in name or "pos_embed" in name or "cls_token" in name:
            param.requires_grad = False

    for i, block in enumerate(backbone.blocks):
        if i < cfg["freeze_blocks"]:
            for param in block.parameters():
                param.requires_grad = False

    embed_dim = backbone.num_features
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )

    model = DINOv2Classifier(backbone, head)

    # Load checkpoint
    ckpt_path = cfg["checkpoint_in"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    log.info(f"Loaded weights from {ckpt_path}")

    model = model.to(device)

    # Gradient checkpointing
    if cfg["grad_checkpointing"]:
        if hasattr(model.backbone, "set_grad_checkpointing"):
            model.backbone.set_grad_checkpointing(enable=True)
            log.info("Gradient checkpointing enabled")

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: total={n_total:,} trainable={n_trainable:,}")

    return model


def build_optimizer(model, cfg):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg["lr_backbone"]},
            {"params": head_params, "lr": cfg["lr_head"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    return optimizer


# --------------------------------------------------------------------------- #
# PGD Attack
# --------------------------------------------------------------------------- #

class Normalizer:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def eps_in_norm_space(self, eps):
        return eps / self.std

    def valid_range(self):
        lower = (0.0 - self.mean) / self.std
        upper = (1.0 - self.mean) / self.std
        return lower, upper


def pgd_attack(model, images, labels, cfg, normalizer, amp_dtype):
    eps_t = normalizer.eps_in_norm_space(cfg["eps"])
    alpha_t = normalizer.eps_in_norm_space(cfg["alpha"])
    lower, upper = normalizer.valid_range()

    if cfg["random_start"]:
        delta = torch.empty_like(images).uniform_(-1.0, 1.0) * eps_t
    else:
        delta = torch.zeros_like(images)

    delta = torch.max(torch.min(delta, upper - images), lower - images)
    delta.requires_grad_(True)

    for _ in range(cfg["pgd_steps"]):
        adv = images + delta
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                           dtype=amp_dtype, enabled=True):
            logits = model(adv)
            loss = F.cross_entropy(logits, labels)

        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            delta = delta + alpha_t * grad.sign()
            delta = torch.max(torch.min(delta, eps_t), -eps_t)
            delta = torch.max(torch.min(delta, upper - images), lower - images)
        delta = delta.detach().requires_grad_(True)

    return (images + delta).detach()


# --------------------------------------------------------------------------- #
# VRAM probe
# --------------------------------------------------------------------------- #

def run_probe(cfg, device, log):
    log.info("Running VRAM probe...")
    
    if device.type != "cuda":
        log.warning("CUDA not available - running on CPU (this will be slow)")
    
    model = build_model(cfg, device, log)
    optimizer = build_optimizer(model, cfg)
    normalizer = Normalizer(MEAN, STD, device)
    amp_dtype = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16

    x = torch.randn(cfg["micro_batch_size"], 3, cfg["image_size"], cfg["image_size"], device=device)
    y = torch.randint(0, 2, (cfg["micro_batch_size"],), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    model.eval()
    adv_x = pgd_attack(model, x, y, cfg, normalizer, amp_dtype)
    model.train()

    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                       dtype=amp_dtype, enabled=True):
        logits = model(adv_x)
        loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    elapsed = time.time() - t0

    log.info(f"Probe finished in {elapsed:.2f}s")
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        log.info(f"Peak GPU memory: {peak_gb:.2f} GB")
    else:
        log.info("Run with CUDA for accurate memory measurements")


# --------------------------------------------------------------------------- #
# Train/Eval loops
# --------------------------------------------------------------------------- #

def split_clean_adv(images, labels, adv_ratio):
    n = images.size(0)
    n_adv = max(1, int(round(n * adv_ratio))) if n > 1 else (1 if adv_ratio >= 0.5 else 0)
    idx = torch.randperm(n)
    adv_idx, clean_idx = idx[:n_adv], idx[n_adv:]
    return images[clean_idx], labels[clean_idx], images[adv_idx], labels[adv_idx]


def train_one_epoch(model, loader, optimizer, cfg, device, normalizer, amp_dtype, log, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    accum = cfg["accum_steps"]
    t_epoch = time.time()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        clean_x, clean_y, adv_src_x, adv_src_y = split_clean_adv(images, labels, cfg["adv_ratio"])

        if adv_src_x.size(0) > 0:
            model.eval()
            adv_x = pgd_attack(model, adv_src_x, adv_src_y, cfg, normalizer, amp_dtype)
            model.train()
        else:
            adv_x = adv_src_x

        batch_x = torch.cat([clean_x, adv_x], dim=0)
        batch_y = torch.cat([clean_y, adv_src_y], dim=0)

        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                           dtype=amp_dtype, enabled=True):
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y) / accum

        loss.backward()

        if (step + 1) % accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            total_loss += loss.item() * accum

        if (step + 1) % 100 == 0:
            log.info(f"  epoch {epoch} step {step+1} | loss={total_loss/(step+1):.4f} acc={correct/total:.4f}")

    if len(loader) % accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["max_grad_norm"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / len(loader), correct / total, time.time() - t_epoch


@torch.no_grad()
def evaluate_clean(model, loader, device, amp_dtype, cfg, log, tag="clean"):
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(loader):
        if cfg["eval_max_batches"] and i >= cfg["eval_max_batches"]:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                           dtype=amp_dtype, enabled=True):
            logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / max(total, 1)
    log.info(f"  [{tag}] acc={acc:.4f} over {total} samples")
    return acc


def evaluate_adversarial(model, loader, cfg, device, normalizer, amp_dtype, log, tag="adv"):
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(loader):
        if cfg["eval_max_batches"] and i >= cfg["eval_max_batches"]:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.enable_grad():
            adv_images = pgd_attack(model, images, labels, cfg, normalizer, amp_dtype)

        with torch.no_grad():
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                               dtype=amp_dtype, enabled=True):
                logits = model(adv_images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(total, 1)
    log.info(f"  [{tag}] PGD-{cfg['pgd_steps']} acc={acc:.4f} over {total} samples")
    return acc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="PGD adversarial fine-tuning")
    p.add_argument("--data_root", type=str, default=CFG_ADV["data_root"])
    p.add_argument("--checkpoint_in", type=str, default=CFG_ADV["checkpoint_in"])
    p.add_argument("--checkpoint_out_dir", type=str, default=CFG_ADV["checkpoint_out_dir"])
    p.add_argument("--epochs", type=int, default=CFG_ADV["epochs"])
    p.add_argument("--micro_batch_size", type=int, default=CFG_ADV["micro_batch_size"])
    p.add_argument("--accum_steps", type=int, default=CFG_ADV["accum_steps"])
    p.add_argument("--pgd_steps", type=int, default=CFG_ADV["pgd_steps"])
    p.add_argument("--alpha", type=float, default=CFG_ADV["alpha"])
    p.add_argument("--adv_ratio", type=float, default=CFG_ADV["adv_ratio"])
    p.add_argument("--eval_max_batches", type=int, default=CFG_ADV.get("eval_max_batches", None))
    p.add_argument("--probe", action="store_true", help="Run VRAM probe and exit")
    p.add_argument("--baseline_only", action="store_true", help="Only evaluate baseline")
    p.add_argument("--seed", type=int, default=CFG_ADV["seed"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(CFG_ADV)
    cfg.update({
        "data_root": args.data_root,
        "checkpoint_in": args.checkpoint_in,
        "checkpoint_out_dir": args.checkpoint_out_dir,
        "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "accum_steps": args.accum_steps,
        "pgd_steps": args.pgd_steps,
        "alpha": args.alpha,
        "adv_ratio": args.adv_ratio,
        "eval_max_batches": args.eval_max_batches,
        "seed": args.seed,
    })

    set_seed(cfg["seed"])
    log = build_logger(cfg["checkpoint_out_dir"])
    log.info("=" * 70)
    log.info("Adversarial fine-tuning run")
    log.info(f"Config: {cfg}")
    log.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    if args.probe:
        run_probe(cfg, device, log)
        return

    # If CUDA not available, exit
    if device.type != "cuda":
        log.error("CUDA is required for adversarial training. Exiting.")
        sys.exit(1)

    amp_dtype = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16

    model = build_model(cfg, device, log)
    normalizer = Normalizer(MEAN, STD, device)
    _, _, train_loader, eval_loader = build_dataloaders(cfg)
    log.info(f"Train batches/epoch: {len(train_loader)} | Eval batches: {len(eval_loader)}")

    # Baseline evaluation
    log.info("-" * 70)
    log.info("BASELINE evaluation")
    log.info("-" * 70)
    baseline_clean_acc = evaluate_clean(model, eval_loader, device, amp_dtype, cfg, log, "baseline-clean")
    baseline_adv_acc = evaluate_adversarial(model, eval_loader, cfg, device, normalizer, amp_dtype, log, "baseline-adv")
    log.info(f"BASELINE: clean={baseline_clean_acc:.4f} adv={baseline_adv_acc:.4f}")

    if args.baseline_only:
        log.info("--baseline_only set, exiting.")
        return

    optimizer = build_optimizer(model, cfg)
    os.makedirs(cfg["checkpoint_out_dir"], exist_ok=True)
    best_adv_acc = baseline_adv_acc
    history = []

    log.info("-" * 70)
    log.info("Starting adversarial fine-tuning")
    log.info("-" * 70)

    for epoch in range(1, cfg["epochs"] + 1):
        try:
            train_loss, train_acc, elapsed = train_one_epoch(
                model, train_loader, optimizer, cfg, device, normalizer, amp_dtype, log, epoch
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                log.error(f"OOM at epoch {epoch}: {e}")
                emergency_path = os.path.join(cfg["checkpoint_out_dir"], f"adv_model_epoch{epoch}_oom.pt")
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "cfg": cfg}, emergency_path)
                log.info(f"Emergency checkpoint saved to {emergency_path}")
                raise
            raise

        clean_acc = evaluate_clean(model, eval_loader, device, amp_dtype, cfg, log, f"epoch{epoch}-clean")
        adv_acc = evaluate_adversarial(model, eval_loader, cfg, device, normalizer, amp_dtype, log, f"epoch{epoch}-adv")

        log.info(f"Epoch {epoch}/{cfg['epochs']} | {elapsed:.0f}s | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | clean={clean_acc:.4f} adv={adv_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "clean_acc": clean_acc, "adv_acc": adv_acc})

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "cfg": cfg,
            "clean_acc": clean_acc,
            "adv_acc": adv_acc,
            "baseline_clean_acc": baseline_clean_acc,
            "baseline_adv_acc": baseline_adv_acc,
        }
        epoch_path = os.path.join(cfg["checkpoint_out_dir"], f"adv_model_epoch{epoch}.pt")
        torch.save(ckpt, epoch_path)
        log.info(f"  saved {epoch_path}")

        if adv_acc > best_adv_acc:
            best_adv_acc = adv_acc
            best_path = os.path.join(cfg["checkpoint_out_dir"], "adv_best_model.pt")
            torch.save(ckpt, best_path)
            log.info(f"  * new best: {best_adv_acc:.4f} -> saved {best_path}")

    log.info("=" * 70)
    log.info("Complete.")
    log.info(f"Baseline: clean={baseline_clean_acc:.4f} adv={baseline_adv_acc:.4f}")
    for h in history:
        log.info(f"Epoch {h['epoch']}: clean={h['clean_acc']:.4f} adv={h['adv_acc']:.4f}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

