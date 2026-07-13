"""
Robust Grad-CAM++ + Feature-Energy saliency for Mamba3Yolo.

Always produces a heatmap even when class scores are collapsed
(common after placeholder-loss pretraining).
"""

from __future__ import annotations
from typing import List, Optional
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
    """Grad-CAM++ with automatic feature-energy fallback."""

    def __init__(self, model: nn.Module, target_layers: List[nn.Module], nc: int = 9, reg_max: int = 16):
        self.model = model
        self.target_layers = target_layers
        self.nc = nc
        self.reg_max = reg_max
        self.activations: List[Tensor] = []
        self.gradients: List[Tensor] = []
        self._handles = []

        def save_act(module, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            self.activations.append(o.detach())

        def save_grad(module, gin, gout):
            g = gout[0] if isinstance(gout, (tuple, list)) else gout
            self.gradients.append(g.detach() if g is not None else None)

        for layer in target_layers:
            self._handles.append(layer.register_forward_hook(save_act))
            self._handles.append(layer.register_full_backward_hook(save_grad))

    def __call__(self, x: Tensor, class_idx: Optional[int] = None) -> List[Tensor]:
        self.activations.clear()
        self.gradients.clear()
        self.model.zero_grad()
        x = x.clone().requires_grad_(True)
        outs = self.model(x)

        heatmaps: List[Tensor] = []
        try:
            scores = []
            for o in outs:
                cls = o[:, self.reg_max * 4 :, :, :]
                scores.append(cls.amax(dim=(2, 3)))
            score = torch.stack(scores, dim=0).amax(dim=0)  # (B, nc)

            if class_idx is None:
                class_idx = score.argmax(dim=1)

            # add tiny residual so gradient never completely vanishes
            selected = score[torch.arange(score.size(0), device=score.device), class_idx].sum()
            selected = selected + 0.01 * score.sum()
            selected.backward(retain_graph=False)

            for act, grad in zip(self.activations, self.gradients):
                if act is None or grad is None or act.shape != grad.shape:
                    continue
                grad2 = grad ** 2
                grad3 = grad ** 3
                alpha = grad2 / (2.0 * grad2 + (act * grad3).sum(dim=(2, 3), keepdim=True) + 1e-8)
                weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * act).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                heatmaps.append(cam)
        except Exception:
            pass

        # ALWAYS provide feature-energy fallback
        if not heatmaps:
            for act in self.activations:
                cam = act.abs().mean(dim=1, keepdim=True)
                cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                heatmaps.append(cam)

        return heatmaps

    def close(self):
        for h in self._handles:
            h.remove()


def find_target_layers(model: nn.Module, max_layers: int = 3) -> List[nn.Module]:
    candidates = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and any(k in name.lower() for k in ["stage", "neck", "sppf", "cv", "mamba", "odss"]):
            candidates.append((name, m))
    if not candidates:
        for name, m in list(model.named_modules())[-20:]:
            if isinstance(m, nn.Conv2d):
                candidates.append((name, m))
    selected = [m for _, m in candidates[-max_layers:]]
    return selected if selected else [list(model.modules())[-3]]


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    """image: HWC uint8 RGB, heatmap: HW float [0,1] → HWC uint8"""
    if HAS_CV2:
        hm = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != heatmap.shape:
            image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
    else:
        hm = np.clip(heatmap, 0, 1)
        r = np.clip(1.5 - np.abs(4 * hm - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * hm - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * hm - 1), 0, 1)
        color = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
        if image.shape[:2] != heatmap.shape:
            from PIL import Image as PILImage
            image = np.array(PILImage.fromarray(image).resize((heatmap.shape[1], heatmap.shape[0])))
    out = image.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
