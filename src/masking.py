"""
masking.py
----------
Frequency-Aware Masking Strategy for MAE-based Anomaly Detection.

Core Novelty:
    Instead of random patch masking (vanilla MAE), we rank patches by their
    high-frequency energy (via 2D FFT) and preferentially mask the top-k
    high-frequency patches. This forces the decoder to reconstruct fine-grained
    texture details, making the model more sensitive to subtle industrial defects.

Masking Modes:
    - 'freq'   : Top-k high-frequency patches masked (proposed method)
    - 'random' : Standard MAE random masking (baseline)
    - 'low'    : Top-k LOW-frequency patches masked (ablation)

Reference:
    He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Frequency Energy Scoring
# ---------------------------------------------------------------------------

def compute_patch_freq_energy(patches: torch.Tensor) -> torch.Tensor:
    """
    Compute high-frequency energy score for each patch via 2D FFT.

    Args:
        patches: (B, N, C, Ph, Pw)
                 B  = batch size
                 N  = number of patches
                 C  = channels
                 Ph = patch height
                 Pw = patch width

    Returns:
        scores: (B, N) — higher score means more high-frequency content
    """
    B, N, C, Ph, Pw = patches.shape

    # Average over channels for frequency analysis
    x = patches.mean(dim=2)                         # (B, N, Ph, Pw)

    # 2D FFT → shift zero-freq to center
    fft = torch.fft.fft2(x, norm="ortho")           # (B, N, Ph, Pw) complex
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    magnitude = fft_shifted.abs()                   # (B, N, Ph, Pw)

    # Build high-frequency mask: exclude center (low-freq) region
    cy, cx = Ph // 2, Pw // 2
    radius = min(Ph, Pw) // 4                       # inner 25% = low-freq zone

    yy, xx = torch.meshgrid(
        torch.arange(Ph, device=patches.device),
        torch.arange(Pw, device=patches.device),
        indexing="ij"
    )
    dist = ((yy - cy) ** 2 + (xx - cx) ** 2).float().sqrt()
    high_freq_mask = (dist > radius).float()        # (Ph, Pw)

    # High-frequency energy = sum of magnitude in high-freq zone
    scores = (magnitude * high_freq_mask).sum(dim=(-2, -1))  # (B, N)

    return scores


# ---------------------------------------------------------------------------
# Patch Extraction Helper
# ---------------------------------------------------------------------------

def extract_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Split image tensor into non-overlapping patches.

    Args:
        x          : (B, C, H, W)
        patch_size : int, side length of each square patch

    Returns:
        patches: (B, N, C, patch_size, patch_size)
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        f"Image size ({H}x{W}) must be divisible by patch_size ({patch_size})"
    )

    x = x.unfold(2, patch_size, patch_size)         # (B, C, nH, W, Ph)
    x = x.unfold(3, patch_size, patch_size)         # (B, C, nH, nW, Ph, Pw)
    B, C, nH, nW, Ph, Pw = x.shape
    x = x.contiguous().view(B, C, nH * nW, Ph, Pw)  # (B, C, N, Ph, Pw)
    x = x.permute(0, 2, 1, 3, 4)                    # (B, N, C, Ph, Pw)

    return x


# ---------------------------------------------------------------------------
# Core Masking Module
# ---------------------------------------------------------------------------

class FrequencyAwareMasking(nn.Module):
    """
    Generates mask indices for MAE training using frequency-aware strategy.

    Args:
        mask_ratio  : float, fraction of patches to mask (default: 0.75)
        patch_size  : int, patch side length in pixels (default: 16)
        mode        : 'freq'   → top-k high-freq masked   [proposed]
                      'random' → standard MAE masking      [baseline]
                      'low'    → top-k low-freq masked     [ablation]
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        patch_size: int = 16,
        mode: str = "freq",
    ):
        super().__init__()
        assert 0.0 < mask_ratio < 1.0, "mask_ratio must be in (0, 1)"
        assert mode in ("freq", "random", "low"), (
            f"mode must be 'freq', 'random', or 'low', got '{mode}'"
        )
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.mode = mode

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate mask for a batch of images.

        Args:
            x: (B, C, H, W) — input images

        Returns:
            ids_keep   : (B, N_visible) — indices of unmasked patches
            ids_masked : (B, N_masked)  — indices of masked patches
            mask       : (B, N)         — binary mask (1 = masked, 0 = visible)
        """
        B, C, H, W = x.shape
        P = self.patch_size
        N = (H // P) * (W // P)
        n_masked = int(np.ceil(N * self.mask_ratio))
        n_visible = N - n_masked

        if self.mode == "random":
            ids_shuffle = self._random_indices(B, N, x.device)

        elif self.mode == "freq":
            patches = extract_patches(x, P)             # (B, N, C, P, P)
            scores = compute_patch_freq_energy(patches)  # (B, N)
            # Sort descending: highest freq energy first → these get masked
            ids_shuffle = scores.argsort(dim=1, descending=True)

        elif self.mode == "low":
            patches = extract_patches(x, P)
            scores = compute_patch_freq_energy(patches)
            # Sort ascending: lowest freq energy first → these get masked
            ids_shuffle = scores.argsort(dim=1, descending=False)

        ids_masked = ids_shuffle[:, :n_masked]           # (B, N_masked)
        ids_keep = ids_shuffle[:, n_masked:]             # (B, N_visible)

        # Binary mask in original patch order (for loss computation)
        mask = torch.zeros(B, N, device=x.device)
        mask.scatter_(1, ids_masked, 1.0)                # 1 = masked

        return ids_keep, ids_masked, mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _random_indices(B: int, N: int, device: torch.device) -> torch.Tensor:
        """Per-sample random permutation of patch indices."""
        noise = torch.rand(B, N, device=device)
        return noise.argsort(dim=1)


# ---------------------------------------------------------------------------
# Quick Sanity Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    batch = torch.randn(4, 3, 224, 224)   # 4 images, 3ch, 224x224
    # 224/16 = 14 → 14x14 = 196 patches

    for mode in ("freq", "random", "low"):
        masker = FrequencyAwareMasking(mask_ratio=0.75, patch_size=16, mode=mode)
        ids_keep, ids_masked, mask = masker(batch)

        print(f"[{mode:>6}] keep={ids_keep.shape} | masked={ids_masked.shape} "
              f"| mask_sum={mask.sum(dim=1).mean().item():.1f} / 196")

    print("\nmasking.py OK")