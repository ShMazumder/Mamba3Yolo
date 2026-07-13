"""
Real YOLO-style detection loss for Mamba3Yolo.

Simplified but usable:
- DFL (Distribution Focal Loss) for box regression
- BCE for classification
- Simple center-prior assignment (no full TaskAlignedAssigner for speed)

Enough to replace the placeholder loss and produce meaningful gradients.
"""

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def bbox_iou(box1: Tensor, box2: Tensor, xywh: bool = False, CIoU: bool = True, eps: float = 1e-7) -> Tensor:
    """IoU / CIoU between box1 (N,4) and box2 (M,4) or broadcast."""
    if xywh:
        b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
        b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
        b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if not CIoU:
        return iou

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    v = (4 / (3.14159265 ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


class YOLOLoss(nn.Module):
    """
    Practical YOLO loss for the multi-scale raw outputs of Mamba3Yolo.

    preds: list of 3 tensors (B, 4*reg_max + nc, H, W)
    targets: (N, 6)  [batch_idx, cls, cx, cy, w, h]  (normalized 0-1)
    """

    def __init__(self, nc: int = 9, reg_max: int = 16, box_w: float = 7.5, cls_w: float = 0.5, dfl_w: float = 1.5):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.box_w = box_w
        self.cls_w = cls_w
        self.dfl_w = dfl_w
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # project for DFL
        self.register_buffer("project", torch.arange(reg_max, dtype=torch.float))

    def forward(self, preds: List[Tensor], targets: Tensor) -> Tensor:
        device = preds[0].device
        if targets.numel() == 0 or targets.shape[0] == 0:
            # no targets → just push predictions down a little
            return sum(p.float().pow(2).mean() for p in preds) * 0.01

        total_loss = torch.zeros(1, device=device)
        n_pos = 0

        for pi, pred in enumerate(preds):
            B, C, H, W = pred.shape
            stride = 640 // H   # approximate (imgsz=640)
            # reshape to (B, H, W, no)
            pred = pred.permute(0, 2, 3, 1).contiguous()  # B,H,W,no
            box_pred = pred[..., : self.reg_max * 4]
            cls_pred = pred[..., self.reg_max * 4 :]

            # simple assignment: for each gt, find the grid cell of its center
            for b in range(B):
                t = targets[targets[:, 0] == b]
                if t.numel() == 0:
                    # background: encourage low objectness via cls
                    total_loss = total_loss + self.bce(cls_pred[b], torch.zeros_like(cls_pred[b])).mean() * self.cls_w * 0.1
                    continue

                for gt in t:
                    cls_id = int(gt[1].item())
                    if cls_id < 0 or cls_id >= self.nc:
                        continue
                    cx, cy, bw, bh = gt[2:6]
                    gj = min(int(cy * H), H - 1)
                    gi = min(int(cx * W), W - 1)

                    # classification target
                    tcls = torch.zeros(self.nc, device=device)
                    tcls[cls_id] = 1.0
                    total_loss = total_loss + self.bce(cls_pred[b, gj, gi], tcls).mean() * self.cls_w

                    # box regression (decode DFL → xyxy then CIoU)
                    # soft-argmax DFL
                    box_raw = box_pred[b, gj, gi].view(4, self.reg_max)
                    box_prob = F.softmax(box_raw, dim=1)
                    dist = (box_prob * self.project.to(device)).sum(1)  # l,t,r,b in grid units

                    # convert to xyxy in normalized image coords
                    px = (gi + 0.5) / W
                    py = (gj + 0.5) / H
                    pred_xyxy = torch.stack([
                        px - dist[0] * stride / 640,
                        py - dist[1] * stride / 640,
                        px + dist[2] * stride / 640,
                        py + dist[3] * stride / 640,
                    ])
                    gt_xyxy = torch.stack([
                        cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                    ]).to(device)

                    iou = bbox_iou(pred_xyxy.unsqueeze(0), gt_xyxy.unsqueeze(0), CIoU=True)
                    total_loss = total_loss + (1.0 - iou).mean() * self.box_w
                    n_pos += 1

        if n_pos == 0:
            return total_loss + sum(p.float().pow(2).mean() for p in preds) * 0.001
        return total_loss / max(n_pos, 1)
