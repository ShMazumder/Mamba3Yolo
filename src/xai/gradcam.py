"""
Explainability for Mamba3Yolo.

Provides:
- Grad-CAM++ (works on any convolutional feature map)
- Simple SSM-state saliency (hook on Mamba3 intermediate activations)
- Overlay utilities suitable for medical images (fundus, polyps, cells)

Usage for paper figures and quantitative faithfulness (insertion/deletion).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class GradCAMPlusPlus:
    """Grad-CAM++ for detection feature maps.

    Target layers are typically the last Mamba3ODSSBlock or the neck outputs
    before the Detect head (c3/c4/c5 or p3/p4/p5).
    """

    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        self.model = model
        self.target_layers = target_layers
        self.activations: List[Tensor] = []
        self.gradients: List[Tensor] = []
        self._handles = []

        def save_activation(module, input, output):
            self.activations.append(output.detach())

        def save_gradient(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())

        for layer in target_layers:
            self._handles.append(layer.register_forward_hook(save_activation))
            self._handles.append(layer.register_full_backward_hook(save_gradient))

    def __call__(
        self,
        x: Tensor,
        class_idx: Optional[int] = None,
        retain_graph: bool = False,
    ) -> List[Tensor]:
        """Return list of heatmaps, one per target layer (B, 1, H, W)."""
        self.activations.clear()
        self.gradients.clear()

        self.model.zero_grad()
        outs = self.model(x)  # list of (B, no, H, W)

        # For simplicity take the highest-confidence class score across scales
        # (real detection Grad-CAM needs more careful target selection)
        scores = []
        for o in outs:
            # o: B, (4*reg_max + nc), H, W
            cls = o[:, -self.model.detect.nc :, :, :]
            scores.append(cls.amax(dim=(2, 3)))  # B, nc
        score = torch.stack(scores, dim=0).amax(dim=0)  # B, nc

        if class_idx is None:
            class_idx = score.argmax(dim=1)

        # Backprop the selected class score
        selected = score[torch.arange(score.size(0)), class_idx].sum()
        selected.backward(retain_graph=retain_graph)

        heatmaps = []
        for act, grad in zip(self.activations, self.gradients):
            # Grad-CAM++ weights
            grad_2 = grad ** 2
            grad_3 = grad ** 3
            alpha = grad_2 / (2 * grad_2 + act * grad_3 + 1e-8)
            weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            heatmaps.append(cam)

        return heatmaps

    def close(self):
        for h in self._handles:
            h.remove()


def overlay_heatmap(
    image: Tensor,
    heatmap: Tensor,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    image: (3, H, W) or (H, W, 3) in [0,1] or [0,255]
    heatmap: (1, H, W) or (H, W) in [0,1]
    Returns uint8 RGB overlay.
    """
    import cv2

    if image.ndim == 3 and image.shape[0] == 3:
        img = image.permute(1, 2, 0).cpu().numpy()
    else:
        img = image.cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    hm = heatmap.squeeze().cpu().numpy()
    hm = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1 - alpha, hm_color, alpha, 0)
    return overlay


def ssm_state_saliency(mamba_module: nn.Module, x_seq: Tensor) -> Tensor:
    """
    Crude but useful saliency from Mamba3 hidden state magnitude.
    For research: hook the internal state or use the output norm as proxy.
    Returns (B, L) importance.
    """
    # Placeholder: use L2 norm of the mamba output as importance proxy
    with torch.no_grad():
        y = mamba_module(x_seq)
        importance = y.norm(dim=-1)  # B, L
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    return importance


# Example usage for medical images
def generate_medical_xai(
    model: nn.Module,
    image: Tensor,
    target_layer_names: List[str] = ["stage3", "neck2"],
    save_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """High-level helper for paper figures on fundus / polyp / cell images."""
    # In practice resolve target_layer_names to actual modules via named_modules()
    # For now this is a template.
    results = {}
    # ... (user can expand with concrete layer references)
    return results
