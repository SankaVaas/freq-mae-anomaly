"""
model.py
--------
Frequency-Aware Masked Autoencoder (FreqMAE) for Texture Anomaly Detection.

Architecture:
    Encoder : ViT-Small (lightweight, T4-friendly) — processes only visible patches
    Decoder : Shallow Transformer (4 layers) — reconstructs all patches from
              visible tokens + learned mask tokens
    Masking : FrequencyAwareMasking from masking.py (core novelty)

Anomaly Score (inference):
    Reconstruction error is computed ONLY on masked (high-frequency) patches.
    High error on those patches = likely anomaly.

Reference:
    He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
"""

import torch
import torch.nn as nn
from functools import partial
from typing import Tuple, Dict

from masking import FrequencyAwareMasking, extract_patches


# ---------------------------------------------------------------------------
# Helpers: Positional Embedding
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    Sinusoidal 2D positional embedding.
    Returns: (grid_size^2, embed_dim)
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    half = embed_dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
    omega = 1.0 / (10000 ** omega)                       # (half//2,)

    g = torch.arange(grid_size, dtype=torch.float32)
    out_h = torch.einsum("i,j->ij", g, omega)            # (gs, half//2)
    out_w = out_h.clone()

    emb_h = torch.cat([out_h.sin(), out_h.cos()], dim=1) # (gs, half)
    emb_w = torch.cat([out_w.sin(), out_w.cos()], dim=1) # (gs, half)

    # Combine H and W grids
    emb_h = emb_h.unsqueeze(1).expand(-1, grid_size, -1) # (gs, gs, half)
    emb_w = emb_w.unsqueeze(0).expand(grid_size, -1, -1) # (gs, gs, half)
    emb = torch.cat([emb_h, emb_w], dim=-1)              # (gs, gs, embed_dim)
    return emb.view(grid_size * grid_size, embed_dim)


# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# FreqMAE
# ---------------------------------------------------------------------------

class FreqMAE(nn.Module):
    """
    Frequency-Aware Masked Autoencoder.

    Args:
        img_size      : input image resolution (default: 224)
        patch_size    : patch side length in pixels (default: 16)
        in_chans      : number of input channels (default: 3)
        mask_ratio    : fraction of patches to mask (default: 0.75)
        mask_mode     : 'freq' | 'random' | 'low'  (default: 'freq')
        enc_embed_dim : encoder embedding dim (default: 384, ViT-Small)
        enc_depth     : encoder transformer depth (default: 12)
        enc_num_heads : encoder attention heads (default: 6)
        dec_embed_dim : decoder embedding dim (default: 192)
        dec_depth     : decoder transformer depth (default: 4)
        dec_num_heads : decoder attention heads (default: 3)
        mlp_ratio     : MLP hidden ratio (default: 4.0)
        norm_pix_loss : normalize pixel targets per patch (default: True)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        mask_ratio: float = 0.75,
        mask_mode: str = "freq",
        enc_embed_dim: int = 384,
        enc_depth: int = 12,
        enc_num_heads: int = 6,
        dec_embed_dim: int = 192,
        dec_depth: int = 4,
        dec_num_heads: int = 3,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = True,
    ):
        super().__init__()

        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_dim = in_chans * patch_size * patch_size
        self.enc_embed_dim = enc_embed_dim
        self.norm_pix_loss = norm_pix_loss

        # ── Masking ────────────────────────────────────────────────────────
        self.masker = FrequencyAwareMasking(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            mode=mask_mode,
        )

        # ── Encoder ────────────────────────────────────────────────────────
        self.patch_embed = nn.Linear(self.patch_dim, enc_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_embed_dim))

        enc_pos = get_2d_sincos_pos_embed(enc_embed_dim, self.grid_size)
        self.register_buffer("enc_pos_embed",
                             enc_pos.unsqueeze(0))           # (1, N, D)

        self.encoder = nn.ModuleList([
            TransformerBlock(enc_embed_dim, enc_num_heads, mlp_ratio)
            for _ in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(enc_embed_dim)

        # ── Decoder ────────────────────────────────────────────────────────
        self.dec_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))

        dec_pos = get_2d_sincos_pos_embed(dec_embed_dim, self.grid_size)
        self.register_buffer("dec_pos_embed",
                             dec_pos.unsqueeze(0))           # (1, N, D)

        self.decoder = nn.ModuleList([
            TransformerBlock(dec_embed_dim, dec_num_heads, mlp_ratio)
            for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(dec_embed_dim)
        self.dec_pred = nn.Linear(dec_embed_dim, self.patch_dim, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Patch utilities
    # ------------------------------------------------------------------

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) → (B, N, patch_dim)"""
        P = self.patch_size
        patches = extract_patches(x, P)                 # (B,N,C,P,P)
        B, N, C, Ph, Pw = patches.shape
        return patches.reshape(B, N, C * Ph * Pw)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_dim) → (B, C, H, W)"""
        P = self.patch_size
        G = self.grid_size
        B, N, _ = x.shape
        C = self.patch_dim // (P * P)
        x = x.reshape(B, G, G, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.reshape(B, C, G * P, G * P)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode visible patches, decode all patches.

        Returns:
            loss       : scalar reconstruction loss
            pred       : (B, N, patch_dim) full reconstruction
            mask       : (B, N) binary mask (1 = masked patch)
        """
        # 1. Mask
        ids_keep, ids_masked, mask = self.masker(x)    # all (B, *)

        # 2. Encode visible patches only
        tokens = self._encode(x, ids_keep)             # (B, N_vis, D_enc)

        # 3. Decode all patches (pass ids_masked for clean token placement)
        pred = self._decode(tokens, ids_keep, ids_masked)  # (B, N, patch_dim)

        # 4. Loss on masked patches only
        loss = self._loss(x, pred, mask)

        return loss, pred, mask

    def _encode(self, x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = self.patchify(x)                     # (B, N, patch_dim)

        # Keep only visible patches
        ids_exp = ids_keep.unsqueeze(-1).expand(-1, -1, self.patch_dim)
        visible = patches.gather(1, ids_exp)           # (B, N_vis, patch_dim)

        tokens = self.patch_embed(visible)             # (B, N_vis, D_enc)

        # Add positional embedding for visible positions
        pos = self.enc_pos_embed.expand(B, -1, -1)    # (B, N, D_enc)
        pos_vis = pos.gather(
            1, ids_keep.unsqueeze(-1).expand(-1, -1, self.enc_embed_dim)
        )
        tokens = tokens + pos_vis

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)       # (B, 1+N_vis, D_enc)

        for blk in self.encoder:
            tokens = blk(tokens)
        tokens = self.enc_norm(tokens)

        return tokens[:, 1:, :]                        # drop CLS → (B, N_vis, D_enc)

    def _decode(
        self, tokens: torch.Tensor, ids_keep: torch.Tensor,
        ids_masked: torch.Tensor
    ) -> torch.Tensor:
        B, N_vis, _ = tokens.shape
        N = self.num_patches

        tokens = self.dec_embed(tokens)                # (B, N_vis, D_dec)
        D = tokens.shape[-1]

        # Build full sequence by scattering visible + mask tokens
        full = torch.zeros(B, N, D, device=tokens.device)

        # Place visible tokens at their original positions
        ids_vis_exp = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        full.scatter_(1, ids_vis_exp, tokens)

        # Place learned mask token at each masked position
        mask_tok = self.mask_token.expand(B, ids_masked.shape[1], -1)  # (B, N_masked, D)
        ids_msk_exp = ids_masked.unsqueeze(-1).expand(-1, -1, D)
        full.scatter_(1, ids_msk_exp, mask_tok)

        # Add full positional embedding
        full = full + self.dec_pos_embed.expand(B, -1, -1)

        for blk in self.decoder:
            full = blk(full)
        full = self.dec_norm(full)

        return self.dec_pred(full)                     # (B, N, patch_dim)

    def _loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target = self.patchify(x)                      # (B, N, patch_dim)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
        loss = (loss_per_patch * mask).sum() / mask.sum()     # masked-only MSE
        return loss

    # ------------------------------------------------------------------
    # Inference: per-patch anomaly score
    # ------------------------------------------------------------------

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-image anomaly score for inference.

        Strategy: run multiple forward passes with different masks,
        average reconstruction error on HIGH-FREQUENCY patches.

        Args:
            x: (B, C, H, W)

        Returns:
            scores: (B,) — higher = more anomalous
        """
        self.eval()
        _, pred, mask = self(x)

        target = self.patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        err = ((pred - target) ** 2).mean(dim=-1)      # (B, N)
        # Score = mean error over masked (high-freq) patches
        scores = (err * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return scores                                  # (B,)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_model(cfg: Dict) -> FreqMAE:
    """Build FreqMAE from a config dict (loaded from default.yaml)."""
    return FreqMAE(
        img_size=cfg.get("img_size", 224),
        patch_size=cfg.get("patch_size", 16),
        in_chans=cfg.get("in_chans", 3),
        mask_ratio=cfg.get("mask_ratio", 0.75),
        mask_mode=cfg.get("mask_mode", "freq"),
        enc_embed_dim=cfg.get("enc_embed_dim", 384),
        enc_depth=cfg.get("enc_depth", 12),
        enc_num_heads=cfg.get("enc_num_heads", 6),
        dec_embed_dim=cfg.get("dec_embed_dim", 192),
        dec_depth=cfg.get("dec_depth", 4),
        dec_num_heads=cfg.get("dec_num_heads", 3),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        norm_pix_loss=cfg.get("norm_pix_loss", True),
    )


# ---------------------------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FreqMAE(
        img_size=224, patch_size=16, enc_depth=12, dec_depth=4,
        mask_mode="freq"
    ).to(device)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total:.1f}M")

    x = torch.randn(2, 3, 224, 224, device=device)
    loss, pred, mask = model(x)

    print(f"Loss      : {loss.item():.4f}")
    print(f"Pred shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}  (masked patches: {mask.sum(dim=1).mean():.0f}/196)")

    scores = model.anomaly_score(x)
    print(f"Anomaly scores: {scores}")

    print("\nmodel.py OK")