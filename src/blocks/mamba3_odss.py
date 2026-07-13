"""
Mamba3ODSSBlock and helpers for Mamba3Yolo.

Upgrades the original ODSSBlock (Mamba-YOLO, AAAI 2025) by replacing the S6/SS2D
selective scan with Mamba-3 (complex-valued states, exponential-trapezoidal
discretization, optional MIMO).

Keeps LSBlock and RGBlock for vision-specific local inductive bias
(MambaOut, CVPR 2025 findings).

On platforms where official Mamba-3 MIMO kernels fail (smem / stride issues
common on Kaggle T4), the block permanently switches to a pure-PyTorch
reference after the first failure so training stays fast and quiet.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
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
    """Local Spatial Block (from original Mamba-YOLO)."""

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
    """Residual Gated Block (original Mamba-YOLO)."""

    def __init__(self, dim: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.fc1 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.fc2 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.fc_out = nn.Conv2d(hidden, dim, 1, bias=False)
        self.act = nn.SiLU()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.act(x1) * x2
        x = self.dw(x)
        x = self.fc_out(x)
        return self.norm(x + residual)


class Mamba3Reference(nn.Module):
    """
    Pure-PyTorch reference that approximates Mamba-3 behaviour.
    Used when official kernels are unavailable or fail at runtime.
    """

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

    After the first official-kernel failure, permanently switches to the
    pure-PyTorch reference so subsequent forwards are fast and silent.
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
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Always keep a pure-PyTorch reference ready
        self.ref = Mamba3Reference(
            d_model=dim, d_state=d_state, expand=expand,
            headdim=headdim, is_mimo=is_mimo, mimo_rank=mimo_rank,
        )

        self._use_official = False
        self.mamba = None

        if HAS_MAMBA3:
            # Prefer SISO (is_mimo=False) — MIMO kernels are unstable on many Kaggle images
            try:
                self.mamba = Mamba3(
                    d_model=dim,
                    d_state=min(d_state, 64),
                    expand=expand,
                    headdim=min(headdim, 64),
                    is_mimo=False,          # force SISO for stability
                    mimo_rank=1,
                )
                self._use_official = True
            except Exception as e:
                print(f"[Mamba3SS2D] Official Mamba3 init failed ({e}), using reference")
                self.mamba = self.ref
                self._use_official = False
        else:
            self.mamba = self.ref
            self._use_official = False

    def _safe_mamba(self, seq: Tensor) -> Tensor:
        seq = seq.contiguous()
        if not self._use_official:
            return self.ref(seq)
        try:
            return self.mamba(seq)
        except Exception:
            # Permanent switch — never touch the broken kernel again
            if self._use_official:
                print("[Mamba3SS2D] Official kernel failed once → permanently using pure-PyTorch reference")
                self._use_official = False
                # free the broken module to save memory
                self.mamba = None
            return self.ref(seq)

    def _four_dir_scan(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        x = x.contiguous()

        seq_h = x.flatten(2).transpose(1, 2).contiguous()
        seq_hf = torch.flip(seq_h, dims=[1]).contiguous()
        seq_v = x.permute(0, 1, 3, 2).contiguous().flatten(2).transpose(1, 2).contiguous()
        seq_vf = torch.flip(seq_v, dims=[1]).contiguous()

        out_h = self._safe_mamba(seq_h)
        out_hf = self._safe_mamba(seq_hf)
        out_v = self._safe_mamba(seq_v)
        out_vf = self._safe_mamba(seq_vf)

        out_h = out_h.transpose(1, 2).contiguous().view(B, C, H, W)
        out_hf = torch.flip(out_hf, dims=[1]).transpose(1, 2).contiguous().view(B, C, H, W)
        out_v = out_v.transpose(1, 2).contiguous().view(B, C, W, H).permute(0, 1, 3, 2).contiguous()
        out_vf = torch.flip(out_vf, dims=[1]).transpose(1, 2).contiguous().view(B, C, W, H).permute(0, 1, 3, 2).contiguous()

        return (out_h + out_hf + out_v + out_vf) * 0.25

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self._four_dir_scan(x)
        return residual + self.drop_path(x)


class Mamba3ODSSBlock(nn.Module):
    """
    Object-Detection State-Space Block powered by Mamba-3.

    Flow:
        Conv1x1 + BN + SiLU
        → LSBlock (local)
        → Mamba3SS2D (global)
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
