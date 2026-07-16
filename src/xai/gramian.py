"""
Intrinsic controllability-Gramian explainability for Mamba3Yolo.

Unlike Grad-CAM (gradient-based, saturates on recurrent nets) this reads the
attribution directly from the SSM's own dynamics -- one forward pass, no backprop,
O(L). It is closed-form precisely because Mamba-3's state is complex-diagonal.

  from src.xai.gramian import gramian_saliency
  heat = gramian_saliency(model, img)     # (B, H, W) in [0,1], per input pixel

This is the paper's novel XAI contribution. Compare against Grad-CAM++ (src/xai/gradcam.py)
on insertion/deletion + pointing-game -- the numbers competitors (e.g. AKCMamba-YOLO,
which only shows qualitative Grad-CAM) never report.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.blocks.mamba3_odss import Mamba3SS2D


def _norm(t: Tensor) -> Tensor:
    B = t.shape[0]
    flat = t.reshape(B, -1)
    lo = flat.min(1, keepdim=True).values
    hi = flat.max(1, keepdim=True).values
    return ((flat - lo) / (hi - lo + 1e-8)).view_as(t)


@torch.no_grad()
def gramian_saliency(model: torch.nn.Module, x: Tensor,
                     size: Optional[tuple] = None) -> Tensor:
    """Aggregate intrinsic saliency over every Mamba3SS2D block. Returns (B, H, W)."""
    blocks = [m for m in model.modules() if isinstance(m, Mamba3SS2D)]
    if not blocks:
        raise ValueError("no Mamba3SS2D blocks in model")
    if all(b.ref is None for b in blocks):
        raise ValueError("intrinsic saliency needs the pure-PyTorch reference core "
                         "(the official kernel exposes no internals)")
    captured: dict = {}
    hooks = [b.register_forward_pre_hook(
        lambda mod, inp, key=b: captured.__setitem__(key, inp[0].detach())) for b in blocks]
    was_training = model.training
    model.eval()
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    H0, W0 = size or x.shape[-2:]
    total = None
    for b, inp in captured.items():
        sal = b.spatial_saliency(inp)                      # (B,h,w)
        sal = F.interpolate(sal.unsqueeze(1).float(), size=(H0, W0),
                            mode="bilinear", align_corners=False).squeeze(1)
        sal = _norm(sal)
        total = sal if total is None else total + sal
    return _norm(total)


# ---------------------------------------------------------------- self-tests
if __name__ == "__main__":
    from src.blocks.mamba3_ref import Mamba3RefSSM

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) reverse-energy G matches brute force
    ssm = Mamba3RefSSM(d_model=24, d_state=8, headdim=12).to(dev).eval()
    B, L, H = 2, 40, ssm.nheads
    alpha = (torch.rand(B, L, H, device=dev) * 0.3 + 0.6)      # decays in [0.6, 0.9]
    G_fast = ssm._reverse_energy(alpha.log())
    G_brute = torch.zeros_like(alpha)
    for tau in range(L):                                       # G_tau = sum_{t>=tau} (prod alpha)^2
        D = torch.ones(B, H, device=dev)
        for t in range(tau, L):
            if t > tau:
                D = D * alpha[:, t]
            G_brute[:, tau] += D ** 2
    err = (G_fast - G_brute).abs().max().item()
    print(f"[reverse-energy] max|fast-brute| = {err:.2e}  ->", "OK" if err < 1e-3 else "MISMATCH")

    # 2) does the block's saliency localize a bright region?
    ss2d = Mamba3SS2D(dim=16, d_state=8, headdim=16).to(dev).eval()
    feat = torch.randn(1, 16, 16, 16, device=dev) * 0.1
    feat[:, :, 3:7, 10:14] += 3.0                              # bright patch top-right
    sal = ss2d.spatial_saliency(feat)[0]
    patch = sal[3:7, 10:14].mean().item()
    elsewhere = ((sal.sum() - sal[3:7, 10:14].sum()) / (sal.numel() - 16)).item()
    print(f"[block saliency] patch={patch:.3f} vs elsewhere={elsewhere:.3f}  ->",
          "LOCALIZES" if patch > 2 * elsewhere else "weak")

    # 3) end-to-end on the integrated YOLO11-mamba3
    try:
        from scripts.ultra_mamba3 import register, make_yaml
        from ultralytics import YOLO
        register(verbose=False)
        model = YOLO(make_yaml("s")).model.to(dev)
        img = torch.randn(1, 3, 128, 128, device=dev) * 0.1
        img[:, :, 20:44, 80:104] += 2.0                        # bright object
        heat = gramian_saliency(model, img)                    # (1,128,128)
        obj = heat[0, 20:44, 80:104].mean().item()
        bg = ((heat[0].sum() - heat[0, 20:44, 80:104].sum()) / (heat[0].numel() - 24 * 24)).item()
        print(f"[full model] heat shape {tuple(heat.shape)}, obj={obj:.3f} vs bg={bg:.3f}  ->",
              "OBJECT-FOCUSED" if obj > bg else "diffuse")
    except Exception as e:
        print("[full model] skipped:", type(e).__name__, e)
