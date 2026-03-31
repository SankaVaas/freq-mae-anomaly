"""
evaluate.py
-----------
Evaluation script for FreqMAE on MVTec-AD benchmark.

Metrics (standard for MVTec-AD papers):
    1. Image-level AUROC  — can the model distinguish normal vs anomaly images?
    2. Pixel-level AUROC  — does the reconstruction error map align with GT masks?
    3. PRO Score          — Per-Region Overlap, rewards detecting small defects
                           (more robust than pixel AUROC for imbalanced GT masks)

Usage:
    # Evaluate single category
    python src/evaluate.py --category bottle --checkpoint checkpoints/bottle/best.pth

    # Evaluate all categories (full benchmark table)
    python src/evaluate.py --all --checkpoint_dir checkpoints/

    # Ablation: compare masking modes
    python src/evaluate.py --category bottle --checkpoint checkpoints/bottle/best.pth --mask_mode random
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.ndimage import label as scipy_label

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model, FreqMAE
from dataset import MVTecDataset, build_test_transform, MVTEC_CATEGORIES


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("freqmae.eval")
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
# Anomaly Map Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_anomaly_map(
    model: FreqMAE,
    images: torch.Tensor,
    n_passes: int = 5,
    patch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate per-pixel anomaly map via multi-pass reconstruction error.

    Running multiple forward passes with different random masks (freq mode
    always picks top-k, so we vary by running the stochastic components)
    and averaging gives a more stable anomaly map.

    Args:
        model    : trained FreqMAE
        images   : (B, C, H, W)
        n_passes : number of forward passes to average
        patch_size: patch size used during training

    Returns:
        image_scores : (B,)       per-image anomaly score
        pixel_maps   : (B, 1, H, W) per-pixel anomaly heatmap
    """
    model.eval()
    B, C, H, W = images.shape
    N = (H // patch_size) * (W // patch_size)
    grid = H // patch_size

    accumulated_err = torch.zeros(B, N, device=images.device)
    accumulated_mask = torch.zeros(B, N, device=images.device)

    for _ in range(n_passes):
        _, pred, mask = model(images)               # pred: (B, N, patch_dim)
        target = model.patchify(images)             # (B, N, patch_dim)

        if model.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        err = ((pred - target) ** 2).mean(dim=-1)   # (B, N)
        accumulated_err  += err * mask
        accumulated_mask += mask

    # Average error over passes, only on masked patches
    avg_err = accumulated_err / accumulated_mask.clamp(min=1.0)  # (B, N)

    # Image-level score = max patch error (sensitive to localized defects)
    image_scores = avg_err.max(dim=1).values        # (B,)

    # Reshape to spatial map: (B, 1, grid, grid)
    pixel_maps_small = avg_err.reshape(B, 1, grid, grid)

    # Upsample to full resolution
    pixel_maps = F.interpolate(
        pixel_maps_small.float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )                                               # (B, 1, H, W)

    return image_scores, pixel_maps


# ---------------------------------------------------------------------------
# PRO Score
# ---------------------------------------------------------------------------

def compute_pro_score(
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray,
    n_thresholds: int = 100,
    fpr_limit: float = 0.3,
) -> float:
    """
    Per-Region Overlap (PRO) score.

    For each threshold, compute the mean overlap between predicted anomaly
    regions and each individual connected ground-truth defect region.
    The PRO score is the area under the PRO-FPR curve up to fpr_limit.

    Args:
        anomaly_maps : (N, H, W) float — predicted anomaly scores
        gt_masks     : (N, H, W) binary — ground-truth defect pixels
        n_thresholds : number of thresholds to sweep
        fpr_limit    : integrate PRO curve only up to this FPR

    Returns:
        pro_auc: float in [0, 1]
    """
    thresholds = np.linspace(anomaly_maps.min(), anomaly_maps.max(), n_thresholds)

    pros = []
    fprs = []

    for thresh in thresholds:
        pred_binary = (anomaly_maps >= thresh).astype(np.uint8)

        # Per-region overlap
        region_overlaps = []
        for pred, gt in zip(pred_binary, gt_masks):
            labeled, n_regions = scipy_label(gt)
            if n_regions == 0:
                continue
            for region_id in range(1, n_regions + 1):
                region = (labeled == region_id)
                overlap = (pred[region] > 0).sum() / region.sum()
                region_overlaps.append(overlap)

        if len(region_overlaps) == 0:
            pros.append(0.0)
        else:
            pros.append(np.mean(region_overlaps))

        # FPR on normal pixels
        normal_pixels = gt_masks == 0
        if normal_pixels.sum() > 0:
            fpr = pred_binary[normal_pixels].sum() / normal_pixels.sum()
        else:
            fpr = 0.0
        fprs.append(fpr)

    pros = np.array(pros)
    fprs = np.array(fprs)

    # Sort by FPR for integration
    sort_idx = np.argsort(fprs)
    fprs = fprs[sort_idx]
    pros = pros[sort_idx]

    # Clip to fpr_limit and integrate (trapezoid)
    valid = fprs <= fpr_limit
    if valid.sum() < 2:
        return 0.0

    pro_auc = np.trapz(pros[valid], fprs[valid]) / fpr_limit
    return float(np.clip(pro_auc, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-Category Evaluation
# ---------------------------------------------------------------------------

def evaluate_category(
    model: FreqMAE,
    category: str,
    cfg: Dict,
    logger: logging.Logger,
) -> Dict:
    """
    Evaluate on one MVTec category. Returns dict of metrics.
    """
    device = next(model.parameters()).device

    test_ds = MVTecDataset(
        root=cfg["data_root"],
        category=category,
        split="test",
        img_size=cfg["img_size"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    logger.info(f"  Evaluating {category} ({len(test_ds)} test images)...")

    all_image_scores = []
    all_image_labels = []
    all_pixel_maps   = []
    all_gt_masks     = []

    model.eval()
    for batch in test_loader:
        images = batch["image"].to(device)
        gt_mask = batch["mask"]                            # (1, 1, H, W)
        label   = batch["label"].item()

        img_score, pixel_map = generate_anomaly_map(
            model, images,
            n_passes=cfg.get("n_passes", 5),
            patch_size=cfg["patch_size"],
        )

        all_image_scores.append(img_score.cpu().item())
        all_image_labels.append(label)
        all_pixel_maps.append(pixel_map.cpu().squeeze().numpy())   # (H, W)
        all_gt_masks.append(gt_mask.cpu().squeeze().numpy())       # (H, W)

    all_image_scores = np.array(all_image_scores)
    all_image_labels = np.array(all_image_labels)
    all_pixel_maps   = np.stack(all_pixel_maps)     # (N, H, W)
    all_gt_masks     = np.stack(all_gt_masks)       # (N, H, W)

    # ── Image-level AUROC ────────────────────────────────────────────────
    if all_image_labels.sum() == 0 or all_image_labels.sum() == len(all_image_labels):
        logger.warning(f"  {category}: only one class in test set, skipping AUROC")
        image_auroc = float("nan")
    else:
        image_auroc = roc_auc_score(all_image_labels, all_image_scores)

    # ── Pixel-level AUROC ────────────────────────────────────────────────
    gt_flat    = all_gt_masks.flatten()
    pred_flat  = all_pixel_maps.flatten()
    if gt_flat.sum() == 0:
        pixel_auroc = float("nan")
    else:
        pixel_auroc = roc_auc_score(gt_flat, pred_flat)

    # ── PRO Score ─────────────────────────────────────────────────────────
    pro = compute_pro_score(all_pixel_maps, all_gt_masks)

    metrics = {
        "category"    : category,
        "image_auroc" : round(image_auroc * 100, 1),
        "pixel_auroc" : round(pixel_auroc * 100, 1),
        "pro"         : round(pro * 100, 1),
        "n_test"      : len(test_ds),
    }

    logger.info(
        f"  {category:<15} "
        f"Img-AUROC={metrics['image_auroc']:5.1f}%  "
        f"Pix-AUROC={metrics['pixel_auroc']:5.1f}%  "
        f"PRO={metrics['pro']:5.1f}%"
    )
    return metrics


# ---------------------------------------------------------------------------
# Results Table
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict], logger: logging.Logger):
    logger.info("\n" + "="*65)
    logger.info(f"{'Category':<15} {'Img-AUROC':>10} {'Pix-AUROC':>10} {'PRO':>8}")
    logger.info("-"*65)
    img_scores, pix_scores, pro_scores = [], [], []
    for r in results:
        logger.info(
            f"{r['category']:<15} {r['image_auroc']:>9.1f}%"
            f" {r['pixel_auroc']:>9.1f}%  {r['pro']:>7.1f}%"
        )
        if not np.isnan(r["image_auroc"]): img_scores.append(r["image_auroc"])
        if not np.isnan(r["pixel_auroc"]): pix_scores.append(r["pixel_auroc"])
        if not np.isnan(r["pro"]):         pro_scores.append(r["pro"])
    logger.info("-"*65)
    logger.info(
        f"{'Mean':<15} {np.mean(img_scores):>9.1f}%"
        f" {np.mean(pix_scores):>9.1f}%  {np.mean(pro_scores):>7.1f}%"
    )
    logger.info("="*65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FreqMAE on MVTec-AD")
    parser.add_argument("--config",         type=str, default="configs/default.yaml")
    parser.add_argument("--category",       type=str, default=None,
                        choices=MVTEC_CATEGORIES)
    parser.add_argument("--all",            action="store_true",
                        help="Evaluate all 15 categories")
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Path to .pth checkpoint (single category)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Root dir for checkpoints (--all mode)")
    parser.add_argument("--mask_mode",      type=str, default=None,
                        choices=["freq", "random", "low"],
                        help="Override mask mode for ablation")
    parser.add_argument("--output",         type=str, default=None,
                        help="Save results table to this .txt file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mask_mode:
        cfg["mask_mode"] = args.mask_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = Path(cfg.get("log_dir", "logs"))
    logger = setup_logger(log_dir / "eval.log")
    logger.info(f"Device: {device}  |  mask_mode: {cfg['mask_mode']}")

    # Determine categories
    if args.all:
        categories = MVTEC_CATEGORIES
    elif args.category:
        categories = [args.category]
    else:
        print("Error: specify --category <name> or --all")
        sys.exit(1)

    results = []
    for cat in categories:
        # Resolve checkpoint path
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            ckpt_path = str(Path(args.checkpoint_dir) / cat / "best.pth")

        if not Path(ckpt_path).exists():
            logger.warning(f"Checkpoint not found: {ckpt_path} — skipping {cat}")
            continue

        # Load model
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_cfg = ckpt.get("cfg", cfg)
        if args.mask_mode:
            model_cfg["mask_mode"] = args.mask_mode

        model = build_model(model_cfg).to(device)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded checkpoint: {ckpt_path}")

        metrics = evaluate_category(model, cat, cfg, logger)
        results.append(metrics)

    if len(results) > 1:
        print_results_table(results, logger)

    # Optionally save to file
    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(f"mask_mode={cfg['mask_mode']}\n\n")
            f.write(f"{'Category':<15} {'Img-AUROC':>10} {'Pix-AUROC':>10} {'PRO':>8}\n")
            f.write("-"*50 + "\n")
            for r in results:
                f.write(f"{r['category']:<15} {r['image_auroc']:>9.1f}%"
                        f" {r['pixel_auroc']:>9.1f}%  {r['pro']:>7.1f}%\n")
        logger.info(f"Results saved to {out_path}")