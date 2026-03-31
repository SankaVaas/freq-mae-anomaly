"""
train.py
--------
Training script for Frequency-Aware MAE (FreqMAE) on MVTec-AD.

Features:
    - Loads config from configs/default.yaml
    - Trains self-supervised MAE on normal images only
    - Saves checkpoints per epoch + best model
    - Logs loss to console (and W&B if enabled)
    - Auto-detects GPU (works on Colab T4)
    - Resumes from checkpoint if available

Usage:
    # Train on single category
    python src/train.py --category bottle

    # Train on all 15 categories
    python src/train.py --all

    # Resume from checkpoint
    python src/train.py --category bottle --resume checkpoints/bottle/last.pth

    # Override config values
    python src/train.py --category bottle --epochs 200 --batch_size 64
"""

import os
import sys
import time
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model
from dataset import build_dataloaders, MVTEC_CATEGORIES


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("freqmae")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/default.yaml") -> Dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def override_config(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Apply CLI overrides on top of yaml config."""
    overrides = ["epochs", "batch_size", "lr", "weight_decay",
                 "mask_ratio", "mask_mode", "img_size"]
    for key in overrides:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(state: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler=None) -> int:
    """Load checkpoint. Returns start epoch."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    return start_epoch


# ---------------------------------------------------------------------------
# Single-category training
# ---------------------------------------------------------------------------

def train_one_category(
    category: str,
    cfg: Dict,
    resume: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
):
    if logger is None:
        logger = setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{'='*60}")
    logger.info(f"Category : {category}")
    logger.info(f"Device   : {device}")
    logger.info(f"Epochs   : {cfg['epochs']}")
    logger.info(f"Batch    : {cfg['batch_size']}")
    logger.info(f"Mask mode: {cfg['mask_mode']}  ratio={cfg['mask_ratio']}")
    logger.info(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, _ = build_dataloaders(
        root=cfg["data_root"],
        category=category,
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    logger.info(f"Train batches: {len(train_loader)}  "
                f"({len(train_loader.dataset)} images)")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Parameters: {n_params:.1f}M")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),          # MAE paper recommendation
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=cfg["lr"] * 0.01,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    ckpt_dir = Path(cfg["checkpoint_dir"]) / category
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, scheduler)
        logger.info(f"Resumed from {resume}  (epoch {start_epoch})")

    # ── Training loop ─────────────────────────────────────────────────────
    best_loss = float("inf")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)

            optimizer.zero_grad()
            loss, _, _ = model(images)
            loss.backward()

            # Gradient clipping — stabilizes ViT training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

            # Step-level log every N steps
            if (step + 1) % cfg.get("log_every", 20) == 0:
                logger.info(
                    f"  [ep {epoch+1:03d} | step {step+1:04d}] "
                    f"loss={loss.item():.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1:03d}/{cfg['epochs']}  "
            f"avg_loss={avg_loss:.4f}  "
            f"time={elapsed:.1f}s  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Checkpointing ─────────────────────────────────────────────────
        state = {
            "epoch"    : epoch,
            "model"    : model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss"     : avg_loss,
            "cfg"      : cfg,
            "category" : category,
        }

        # Always save last
        save_checkpoint(state, ckpt_dir / "last.pth")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(state, ckpt_dir / "best.pth")
            logger.info(f"  ✓ Best model saved (loss={best_loss:.4f})")

        # Periodic checkpoint every N epochs
        if (epoch + 1) % cfg.get("save_every", 50) == 0:
            save_checkpoint(state, ckpt_dir / f"epoch_{epoch+1:03d}.pth")

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")
    return best_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FreqMAE on MVTec-AD"
    )
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--category",   type=str, default=None,
                        choices=MVTEC_CATEGORIES,
                        help="Single MVTec category to train on")
    parser.add_argument("--all",        action="store_true",
                        help="Train on all 15 MVTec categories sequentially")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Config overrides
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--mask_ratio",   type=float, default=None)
    parser.add_argument("--mask_mode",    type=str,   default=None,
                        choices=["freq", "random", "low"])
    parser.add_argument("--img_size",     type=int,   default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Load and override config
    cfg = load_config(args.config)
    cfg = override_config(cfg, args)

    # Determine categories to train
    if args.all:
        categories = MVTEC_CATEGORIES
    elif args.category:
        categories = [args.category]
    else:
        print("Error: specify --category <name> or --all")
        print(f"Available: {MVTEC_CATEGORIES}")
        sys.exit(1)

    # Setup logger (shared across categories)
    log_dir = Path(cfg.get("log_dir", "logs"))
    logger = setup_logger(log_dir / "train.log")

    logger.info(f"FreqMAE Training — categories: {categories}")
    logger.info(f"Config: {cfg}")

    results = {}
    for cat in categories:
        resume = args.resume if len(categories) == 1 else None
        best = train_one_category(cat, cfg, resume=resume, logger=logger)
        results[cat] = best

    if len(categories) > 1:
        logger.info("\n=== Final Results ===")
        for cat, loss in results.items():
            logger.info(f"  {cat:<15} best_loss={loss:.4f}")