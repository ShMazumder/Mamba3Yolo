"""
Mamba3ODSSBlock and helpers for Mamba3Yolo.

Upgrades the original ODSSBlock (Mamba-YOLO, AAAI 2025) by replacing the S6/SS2D
selective scan with Mamba-3 (complex-valued states, exponential-trapezoidal
discretization, optional MIMO).

Keeps LSBlock and RGBlock for vision-specific local inductive bias
(MambaOut, CVPR 2025 findings).

References:
- Mamba-YOLO: https://arxiv.org/abs/2406.05835
- Mamba-3: https://arxiv.org/abs/2603.15569
- Official kernels: https://github.com/state-spaces/mamba
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Official Mamba-3 (preferred) or pure-PyTorch reference fallback
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.modules.mamba3 import Mamba3
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class LSBlock(nn.Module):
    """Local Spatial Block (from original Mamba-YOLO). Compensates for SSM's
    weaker local modeling with depthwise conv + residual."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw(x)
        return x + residual


class RGBlock(nn.Module):
    """Residual Gated Block (original Mamba-YOLO). Multi-branch gated channel mixing
    after the global SSM path."""

    def __init__(self, dim: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.fc1 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.fc2 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.fc_out = nn.Conv2d(hidden, dim, 1, bias=False)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x2 = self.dw(x2) + x2
        gate = self.act(x2)
        out = self.fc_out(x1 * gate)
        return self.norm(out + residual)


class Mamba3Reference(nn.Module):
    """Lightweight pure-PyTorch stand-in for Mamba3 when official kernels are
    unavailable. Correct shapes and residual structure only; not for final accuracy."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, D)
        residual = x
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        y = self.act(z) * self.norm(x)
        return self.out_proj(y) + residual


class Mamba3SS2D(nn.Module):
    """
    2D multi-directional selective scan driven by Mamba-3.

    Follows the VMamba / original Mamba-YOLO SS2D pattern:
    four directional sequences → shared Mamba3 mixer → merge.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        if HAS_MAMBA3:
            self.mamba = Mamba3(
                d_model=dim,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
            )
        else:
            self.mamba = Mamba3Reference(
                d_model=dim,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _four_dir_scan(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape

        # Horizontal
        seq_h = x.flatten(2).transpose(1, 2)          # B, H*W, C
        seq_hf = torch.flip(seq_h, dims=[1])

        # Vertical
        seq_v = x.permute(0, 1, 3, 2).contiguous().flatten(2).transpose(1, 2)
        seq_vf = torch.flip(seq_v, dims=[1])

        out_h = self.mamba(seq_h)
        out_hf = self.mamba(seq_hf)
        out_v = self.mamba(seq_v)
        out_vf = self.mamba(seq_vf)

        # Reshape back
        out_h = out_h.transpose(1, 2).view(B, C, H, W)
        out_hf = torch.flip(out_hf, dims=[1]).transpose(1, 2).view(B, C, H, W)
        out_v = out_v.transpose(1, 2).view(B, C, W, H).permute(0, 1, 3, 2)
        out_vf = torch.flip(out_vf, dims=[1]).transpose(1, 2).view(B, C, W, H).permute(0, 1, 3, 2)

        return (out_h + out_hf + out_v + out_vf) * 0.25

    def forward(self, x: Tensor) -> Tensor:
        # x: B C H W
        residual = x
        x = x.permute(0, 2, 3, 1).contiguous()  # B H W C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # B C H W
        x = self._four_dir_scan(x)
        return residual + self.drop_path(x)


class Mamba3ODSSBlock(nn.Module):
    """
    Object-Detection State-Space Block powered by Mamba-3.

    Drop-in replacement for original ODSSBlock / C2f.
    Flow:
        Conv1x1 + BN + SiLU
        → LSBlock (local)
        → Mamba3SS2D (global complex/MIMO)
        → RGBlock (channel gated)
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        drop_path: float = 0.0,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.ls = LSBlock(dim)
        self.ss2d = Mamba3SS2D(
            dim=dim,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            drop_path=drop_path,
        )
        self.rg = RGBlock(dim, expansion=mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.ls(x)
        x = self.ss2d(x)
        x = self.rg(x)
        return x


def build_mamba3_odss(c1: int, c2: int, n: int = 1, **kwargs) -> nn.Module:
    """Factory used by YAML parsers (Ultralytics-style)."""
    if c1 != c2:
        return nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            *[Mamba3ODSSBlock(c2, **kwargs) for _ in range(n)],
        )
    return nn.Sequential(*[Mamba3ODSSBlock(c2, **kwargs) for _ in range(n)])


# Convenience export for registration
__all__ = [
    "Mamba3ODSSBlock",
    "Mamba3SS2D",
    "LSBlock",
    "RGBlock",
    "build_mamba3_odss",
    "HAS_MAMBA3",
]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Official Mamba3 kernels: {HAS_MAMBA3}")
    block = Mamba3ODSSBlock(dim=128, d_state=32, headdim=32, is_mimo=True).to(device)
    x = torch.randn(2, 128, 40, 40, device=device)
    y = block(x)
    print(f"Shape check: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"Params: {sum(p.numel() for p in block.parameters()):,}")
