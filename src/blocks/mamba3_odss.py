"""
Mamba3ODSSBlock - NaN-safe + permanent fallback version
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt
from torch import Tensor

try:
    from mamba_ssm.modules.mamba3 import Mamba3
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False

# Faithful pure-PyTorch Mamba-3 selective SSM (real recurrence, complex/RoPE states).
# Replaces the old gated-MLP Mamba3Reference as the reference/fallback path. Verified
# on the parity state-tracking gate (scripts/validate_parity.py).
from src.blocks.mamba3_ref import Mamba3RefSSM

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
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.act = nn.GELU()
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw(x); x = self.bn(x); x = self.act(x); x = self.pw(x)
        return x + residual

class RGBlock(nn.Module):
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
        x1 = self.fc1(x); x2 = self.fc2(x)
        x = self.act(x1) * x2; x = self.dw(x); x = self.fc_out(x)
        return self.norm(x + residual)


class Mamba3SS2D(nn.Module):
    def __init__(self, dim: int, d_state: int = 64, expand: int = 2, headdim: int = 64, is_mimo: bool = True, mimo_rank: int = 4, drop_path: float = 0.0, use_rope: bool = True, trapezoidal: bool = True):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # Real Mamba-3 SSM reference (verified on parity gate), not the old gated MLP.
        # d_state forced even for the 2-D rotary blocks. use_rope/trapezoidal are the
        # mechanism-ablation toggles (complex-state and trapezoidal discretization).
        self.ref = Mamba3RefSSM(dim, d_state=d_state + (d_state % 2), expand=expand, headdim=headdim, use_rope=use_rope, trapezoidal=trapezoidal)
        self._use_official = False
        self._official_mamba = None   # stored outside nn.Module to avoid doubling params
        if HAS_MAMBA3:
            try:
                self._official_mamba = Mamba3(d_model=dim, d_state=min(d_state, 64), expand=expand, headdim=min(headdim, 64), is_mimo=False, mimo_rank=1)
                self._use_official = True
            except Exception:
                self._official_mamba = None
                self._use_official = False
    def _safe_mamba(self, seq: Tensor) -> Tensor:
        seq = seq.contiguous()
        if not self._use_official:
            return self.ref(seq)
        try:
            return self._official_mamba(seq)
        except Exception:
            if self._use_official:
                print("[Mamba3SS2D] Official kernel failed once → permanently using pure-PyTorch reference")
                self._use_official = False
                self._official_mamba = None
            return self.ref(seq)
    def _four_dir_scan(self, x: Tensor) -> Tensor:
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

    @torch.no_grad()
    def spatial_saliency(self, x: Tensor) -> Tensor:
        """Intrinsic controllability-Gramian saliency map for this block. (B,H,W).
        Runs the 4 scan directions and folds each token-saliency back to 2D."""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        seq_h = x.flatten(2).transpose(1, 2).contiguous()
        seq_hf = torch.flip(seq_h, dims=[1]).contiguous()
        seq_v = x.permute(0, 1, 3, 2).contiguous().flatten(2).transpose(1, 2).contiguous()
        seq_vf = torch.flip(seq_v, dims=[1]).contiguous()
        sal = self.ref.token_saliency if hasattr(self.ref, "token_saliency") else None
        if sal is None:      # official-kernel path has no intrinsic saliency
            return x.new_zeros(B, H, W)
        mh = sal(seq_h).view(B, H, W)
        mhf = torch.flip(sal(seq_hf), dims=[1]).view(B, H, W)
        mv = sal(seq_v).view(B, W, H).permute(0, 2, 1)
        mvf = torch.flip(sal(seq_vf), dims=[1]).view(B, W, H).permute(0, 2, 1)
        return (mh + mhf + mv + mvf) * 0.25

class Mamba3ODSSBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 64, expand: int = 2, headdim: int = 64, is_mimo: bool = True, mimo_rank: int = 4, drop_path: float = 0.0, mlp_ratio: float = 2.0, use_rope: bool = True, trapezoidal: bool = True):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU(inplace=True))
        self.ls = LSBlock(dim)
        self.ss2d = Mamba3SS2D(dim=dim, d_state=d_state, expand=expand, headdim=headdim, is_mimo=is_mimo, mimo_rank=mimo_rank, drop_path=drop_path, use_rope=use_rope, trapezoidal=trapezoidal)
        self.rg = RGBlock(dim, expansion=mlp_ratio)
        self.use_checkpoint = True   # recompute the SSM scan in backward -> big memory saving
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x); x = self.ls(x)
        if self.use_checkpoint and self.training and x.requires_grad:
            x = _ckpt.checkpoint(self.ss2d, x, use_reentrant=False)  # don't store scan trajectory
        else:
            x = self.ss2d(x)
        x = self.rg(x)
        return x

def build_mamba3_odss(c1: int, c2: int, n: int = 1, **kwargs) -> nn.Module:
    if c1 != c2:
        return nn.Sequential(nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True), *[Mamba3ODSSBlock(c2, **kwargs) for _ in range(n)])
    return nn.Sequential(*[Mamba3ODSSBlock(c2, **kwargs) for _ in range(n)])


class Mamba3ODSS(nn.Module):
    """Ultralytics-compatible wrapper (c1, c2, n, ...) so a YOLO yaml can reference
    `Mamba3ODSS` where it would use C3k2. Matches the channel+repeat calling
    convention Ultralytics' parse_model uses for CSP blocks. Leaner defaults
    (expand=1, mlp_ratio=1.0, d_state=32) keep the -s variant near ~12-13M."""

    def __init__(self, c1: int, c2: int, n: int = 1, d_state: int = 32,
                 expand: int = 1, mlp_ratio: float = 1.0,
                 use_rope: bool = True, trapezoidal: bool = True):
        super().__init__()
        self.block = build_mamba3_odss(c1, c2, n=max(int(n), 1), d_state=d_state,
                                       expand=expand, mlp_ratio=mlp_ratio,
                                       use_rope=bool(use_rope), trapezoidal=bool(trapezoidal))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
