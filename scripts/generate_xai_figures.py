#!/usr/bin/env python3
"""
Generate Grad-CAM++ and SSM-saliency figures for paper.

Works with real images (medical or natural). Saves high-resolution
overlays suitable for publication.

Example:
    python scripts/generate_xai_figures.py \
        --weights runs/paper_xxx/best.pt \
        --image /path/to/fundus.jpg \
        --out figures/xai \
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from src.models.mamba3yolo import build_mamba3yolo
from src.xai.gradcam import GradCAMPlusPlus, overlay_heatmap


def load_image(path: str, imgsz: int = 640) -> tuple[torch.Tensor, np.ndarray]:
    img = Image.open(path).convert("RGB")
    orig = np.array(img)
    img = img.resize((imgsz, imgsz), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0), orig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--out", type=str, default="figures/xai")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--class_idx", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.weights, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    model = build_mamba3yolo(
        cfg.get("scale", "T"),
        nc=cfg.get("nc", 80),
        is_mimo=cfg.get("is_mimo", True),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device).eval()
    print(f"Model loaded from {args.weights}")

    # Target layers for Grad-CAM (last stages + neck)
    target_layers = []
    for name, module in model.named_modules():
        if any(k in name for k in ["stage3", "stage4", "neck2", "neck3", "sppf"]):
            if isinstance(module, torch.nn.Conv2d) or "Mamba3" in type(module).__name__:
                target_layers.append(module)
                print(f"  Hooked: {name}")
    if not target_layers:
        # fallback: last few convs
        for name, module in list(model.named_modules())[-10:]:
            if isinstance(module, torch.nn.Conv2d):
                target_layers.append(module)
                break

    if not target_layers:
        print("No suitable target layers found. Using model itself as fallback.")
        target_layers = [model]

    cam = GradCAMPlusPlus(model, target_layers[:3])  # limit to 3

    # Collect images
    img_paths = []
    p = Path(args.image)
    if p.is_dir():
        img_paths = list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg"))
    else:
        img_paths = [p]

    print(f"Processing {len(img_paths)} image(s)...")

    for img_path in img_paths:
        try:
            tensor, orig = load_image(str(img_path), args.imgsz)
            tensor = tensor.to(args.device)

            heatmaps = cam(tensor, class_idx=args.class_idx)

            # Create multi-panel figure
            n = min(len(heatmaps), 3)
            fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
            if n == 0:
                axes = [axes]

            # Original
            axes[0].imshow(orig)
            axes[0].set_title("Original")
            axes[0].axis("off")

            for i, hm in enumerate(heatmaps[:n]):
                overlay = overlay_heatmap(tensor[0], hm[0], alpha=0.45)
                axes[i + 1].imshow(overlay)
                axes[i + 1].set_title(f"Grad-CAM++ L{i}")
                axes[i + 1].axis("off")

            plt.tight_layout()
            stem = img_path.stem
            fig.savefig(out_dir / f"{stem}_gradcam.png", dpi=300, bbox_inches="tight")
            fig.savefig(out_dir / f"{stem}_gradcam.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {stem}_gradcam.png/pdf")

            # Also save individual high-res overlays
            for i, hm in enumerate(heatmaps[:n]):
                ov = overlay_heatmap(tensor[0], hm[0], alpha=0.5)
                Image.fromarray(ov).save(out_dir / f"{stem}_layer{i}.png")

        except Exception as e:
            print(f"  Failed on {img_path}: {e}")

    cam.close()
    print(f"\nAll figures saved to {out_dir}")
    print("Use the PNGs/PDFs directly in your paper (Figure X).")


if __name__ == "__main__":
    main()
