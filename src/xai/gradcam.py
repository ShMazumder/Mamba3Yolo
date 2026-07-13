"""
Robust Grad-CAM++ for Mamba3Yolo (detection heads).

Works even when the model was trained with a proxy loss.
Produces publication-ready overlays for medical images.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layers: List[nn.Module], nc: int = 9, reg_max: int = 16):
        self.model = model
        self.target_layers = target_layers
        self.nc = nc
        self.reg_max = reg_max
        self.activations: List[Tensor] = []
        self.gradients: List[Tensor] = []
        self._handles = []

        def save_act(module, inp, out):
            # keep only the tensor if tuple
            o = out[0] if isinstance(out, (tuple, list)) else out
            self.activations.append(o.detach())

        def save_grad(module, gin, gout):
            g = gout[0] if isinstance(gout, (tuple, list)) else gout
            self.gradients.append(g.detach())

        for layer in target_layers:
            self._handles.append(layer.register_forward_hook(save_act))
            self._handles.append(layer.register_full_backward_hook(save_grad))

    def __call__(self, x: Tensor, class_idx: Optional[int] = None) -> List[Tensor]:
        self.activations.clear()
        self.gradients.clear()
        self.model.zero_grad()
        outs = self.model(x)

        # outs: list of (B, 4*reg_max + nc, H, W)
        scores = []
        for o in outs:
            cls = o[:, self.reg_max * 4 :, :, :]  # (B, nc, H, W)
            scores.append(cls.amax(dim=(2, 3)))   # (B, nc)
        score = torch.stack(scores, dim=0).amax(dim=0)  # (B, nc)

        if class_idx is None:
            class_idx = score.argmax(dim=1)  # (B,)

        selected = score[torch.arange(score.size(0), device=score.device), class_idx].sum()
        try:
            selected.backward(retain_graph=False)
        except RuntimeError:
            # fallback: pure activation map (no gradient)
            heatmaps = []
            for act in self.activations:
                cam = act.abs().mean(dim=1, keepdim=True)
                cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                heatmaps.append(cam)
            return heatmaps

        heatmaps = []
        for act, grad in zip(self.activations, self.gradients):
            if act is None or grad is None or act.shape != grad.shape:
                continue
            # Grad-CAM++
            grad2 = grad ** 2
            grad3 = grad ** 3
            alpha = grad2 / (2.0 * grad2 + (act * grad3).sum(dim=(2, 3), keepdim=True) + 1e-8)
            weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            heatmaps.append(cam)
        return heatmaps

    def close(self):
        for h in self._handles:
            h.remove()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """image: HWC uint8 RGB, heatmap: HW float [0,1]"""
    if HAS_CV2:
        hm = (heatmap * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    else:
        # pure numpy jet-ish
        hm = np.clip(heatmap, 0, 1)
        r = np.clip(1.5 - np.abs(4*hm - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4*hm - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4*hm - 1), 0, 1)
        hm_color = (np.stack([r, g, b], -1) * 255).astype(np.uint8)

    if image.shape[:2] != heatmap.shape:
        if HAS_CV2:
            image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
        else:
            from PIL import Image as PILImage
            image = np.array(PILImage.fromarray(image).resize((heatmap.shape[1], heatmap.shape[0])))
    out = (image.astype(np.float32) * (1 - alpha) + hm_color.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def find_target_layers(model: nn.Module, max_layers: int = 4) -> List[nn.Module]:
    """Auto-select last few Conv2d / Mamba blocks for Grad-CAM."""
    candidates = []
    for name, m in model.named_modules():
        if any(k in name.lower() for k in ["stage3", "stage4", "neck", "sppf", "mamba3", "odss"]):
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)) or "Mamba3" in type(m).__name__ or "ODSS" in type(m).__name__:
                candidates.append((name, m))
    # prefer later layers
    if not candidates:
        for name, m in list(model.named_modules())[-20:]:
            if isinstance(m, nn.Conv2d):
                candidates.append((name, m))
    selected = [m for _, m in candidates[-max_layers:]]
    return selected if selected else [list(model.modules())[-2]]
