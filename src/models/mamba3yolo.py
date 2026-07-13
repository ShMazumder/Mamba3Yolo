"""
Minimal but complete Mamba3Yolo detector.

Self-contained YOLO-style architecture that does not require the full
Ultralytics codebase. Designed so the same Mamba3ODSSBlock can be dropped
into the original Mamba-YOLO repo with a 5-line registration patch.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.blocks.mamba3_odss import Mamba3ODSSBlock, HAS_MAMBA3


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (Ultralytics style)."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Detect(nn.Module):
    """Simplified decoupled detection head (box + cls)."""

    def __init__(self, nc: int = 80, ch: Tuple[int, ...] = (256, 512, 1024)):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16  # DFL
        self.no = nc + self.reg_max * 4

        self.cv2 = nn.ModuleList(
            nn.Sequential(ConvBNAct(x, x, 3), ConvBNAct(x, x, 3), nn.Conv2d(x, 4 * self.reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(ConvBNAct(x, x, 3), ConvBNAct(x, x, 3), nn.Conv2d(x, nc, 1))
            for x in ch
        )
        self.dfl = nn.Identity()  # placeholder; real DFL can be added later

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        # Training: return raw predictions; inference post-process separately
        outs = []
        for i in range(self.nl):
            box = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])
            outs.append(torch.cat((box, cls), 1))
        return outs


class Mamba3Yolo(nn.Module):
    """
    Full Mamba3Yolo detector (Tiny scale by default).

    Can be used stand-alone for research or as reference for patching
    the original Mamba-YOLO Ultralytics codebase.
    """

    def __init__(
        self,
        nc: int = 80,
        width_mult: float = 0.25,
        depth_mult: float = 0.33,
        d_state: int = 64,
        is_mimo: bool = True,
        mimo_rank: int = 4,
    ):
        super().__init__()
        self.nc = nc
        base = [64, 128, 256, 512, 1024]
        ch = [max(int(c * width_mult), 16) for c in base]
        n = max(round(2 * depth_mult), 1)  # number of blocks

        # Backbone
        self.stem = nn.Sequential(
            ConvBNAct(3, ch[0], 3, 2),
            ConvBNAct(ch[0], ch[1], 3, 2),
        )
        self.stage1 = nn.Sequential(
            *[Mamba3ODSSBlock(ch[1], d_state=d_state, is_mimo=is_mimo, mimo_rank=mimo_rank) for _ in range(n)]
        )
        self.down1 = ConvBNAct(ch[1], ch[2], 3, 2)
        self.stage2 = nn.Sequential(
            *[Mamba3ODSSBlock(ch[2], d_state=d_state, is_mimo=is_mimo, mimo_rank=mimo_rank) for _ in range(n)]
        )
        self.down2 = ConvBNAct(ch[2], ch[3], 3, 2)
        self.stage3 = nn.Sequential(
            *[Mamba3ODSSBlock(ch[3], d_state=d_state, is_mimo=is_mimo, mimo_rank=mimo_rank) for _ in range(n)]
        )
        self.down3 = ConvBNAct(ch[3], ch[4], 3, 2)
        self.stage4 = nn.Sequential(
            *[Mamba3ODSSBlock(ch[4], d_state=d_state, is_mimo=is_mimo, mimo_rank=mimo_rank) for _ in range(1)]
        )
        self.sppf = SPPF(ch[4], ch[4])

        # Neck (simplified PAFPN)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.lat1 = ConvBNAct(ch[4] + ch[3], ch[3], 1)
        self.neck1 = Mamba3ODSSBlock(ch[3], d_state=d_state, is_mimo=is_mimo, mimo_rank=mimo_rank)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.lat2 = ConvBNAct(ch[3] + ch[2], ch[2], 1)
        self.neck2 = Mamba3ODSSBlock(ch[2], d_state=d_state, is_mimo=is_mimo, mimo_rank=mimo_rank)

        self.down_n1 = ConvBNAct(ch[2], ch[2], 3, 2)
        self.lat3 = ConvBNAct(ch[2] + ch[3], ch[3], 1)
        self.neck3 = Mamba3ODSSBlock(ch[3], d_state=d_state, is_mimo=is_mimo, mimo_rank=mimo_rank)

        self.down_n2 = ConvBNAct(ch[3], ch[3], 3, 2)
        self.lat4 = ConvBNAct(ch[3] + ch[4], ch[4], 1)
        self.neck4 = Mamba3ODSSBlock(ch[4], d_state=d_state, is_mimo=is_mimo, mimo_rank=mimo_rank)

        # Head
        self.detect = Detect(nc=nc, ch=(ch[2], ch[3], ch[4]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        # Backbone
        x = self.stem(x)
        c2 = self.stage1(x)
        x = self.down1(c2)
        c3 = self.stage2(x)
        x = self.down2(c3)
        c4 = self.stage3(x)
        x = self.down3(c4)
        c5 = self.stage4(x)
        c5 = self.sppf(c5)

        # Neck
        p5 = c5
        p4 = self.neck1(self.lat1(torch.cat([self.up1(p5), c4], 1)))
        p3 = self.neck2(self.lat2(torch.cat([self.up2(p4), c3], 1)))

        p4 = self.neck3(self.lat3(torch.cat([self.down_n1(p3), p4], 1)))
        p5 = self.neck4(self.lat4(torch.cat([self.down_n2(p4), p5], 1)))

        return self.detect([p3, p4, p5])

    def get_feature_maps(self, x: Tensor) -> Dict[str, Tensor]:
        """For XAI / Grad-CAM hooks."""
        feats = {}
        x = self.stem(x)
        c2 = self.stage1(x)
        feats["c2"] = c2
        x = self.down1(c2)
        c3 = self.stage2(x)
        feats["c3"] = c3
        x = self.down2(c3)
        c4 = self.stage3(x)
        feats["c4"] = c4
        x = self.down3(c4)
        c5 = self.stage4(x)
        c5 = self.sppf(c5)
        feats["c5"] = c5
        return feats


def build_mamba3yolo(cfg: str = "T", nc: int = 80, **kwargs) -> Mamba3Yolo:
    """Factory."""
    if cfg.upper() == "T":
        return Mamba3Yolo(nc=nc, width_mult=0.25, depth_mult=0.33, **kwargs)
    elif cfg.upper() == "M":
        return Mamba3Yolo(nc=nc, width_mult=0.50, depth_mult=0.67, **kwargs)
    elif cfg.upper() == "L":
        return Mamba3Yolo(nc=nc, width_mult=1.00, depth_mult=1.00, **kwargs)
    else:
        raise ValueError(f"Unknown scale {cfg}")


if __name__ == "__main__":
    model = build_mamba3yolo("T", nc=80, is_mimo=True)
    print(f"Official Mamba3: {HAS_MAMBA3}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(1, 3, 640, 640)
    outs = model(x)
    print(f"Outputs: {[o.shape for o in outs]}")
