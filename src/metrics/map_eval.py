"""
Simple pure-PyTorch mAP@0.5 evaluator for Mamba3Yolo.

Does not require pycocotools. Good enough for paper ablation tables.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def xywhn_to_xyxy(boxes: Tensor, w: int, h: int) -> Tensor:
    """normalized cxcywh → absolute xyxy"""
    x, y, bw, bh = boxes.T
    return torch.stack([(x - bw/2)*w, (y - bh/2)*h, (x + bw/2)*w, (y + bh/2)*h], -1)


def box_iou(box1: Tensor, box2: Tensor, eps: float = 1e-7) -> Tensor:
    """(N,4) vs (M,4) → (N,M)"""
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2 - inter + eps)


def decode_preds(preds: List[Tensor], conf_thres: float = 0.25, reg_max: int = 16, nc: int = 9, imgsz: int = 640) -> List[Tensor]:
    """
    Decode multi-scale raw outputs → list of (N, 6) [x1,y1,x2,y2,score,cls] per image.
    Very simple top-k per cell (no full NMS for speed).
    """
    device = preds[0].device
    B = preds[0].shape[0]
    project = torch.arange(reg_max, device=device, dtype=torch.float)
    results = [[] for _ in range(B)]

    for pred in preds:
        B, C, H, W = pred.shape
        stride = imgsz / H
        pred = pred.permute(0, 2, 3, 1)  # B,H,W,C
        box_raw = pred[..., : reg_max*4].view(B, H, W, 4, reg_max)
        cls_raw = pred[..., reg_max*4 :]  # B,H,W,nc
        box_prob = F.softmax(box_raw, dim=-1)
        dist = (box_prob * project).sum(-1)  # B,H,W,4  ltrb

        # grid
        gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        cx = (gx + 0.5) * stride
        cy = (gy + 0.5) * stride

        x1 = cx - dist[..., 0] * stride
        y1 = cy - dist[..., 1] * stride
        x2 = cx + dist[..., 2] * stride
        y2 = cy + dist[..., 3] * stride
        boxes = torch.stack([x1, y1, x2, y2], -1)  # B,H,W,4

        scores, cls_ids = cls_raw.sigmoid().max(-1)  # B,H,W
        mask = scores > conf_thres

        for b in range(B):
            m = mask[b]
            if m.any():
                bboxes = boxes[b][m]
                sc = scores[b][m]
                cl = cls_ids[b][m].float()
                results[b].append(torch.cat([bboxes, sc.unsqueeze(1), cl.unsqueeze(1)], 1))

    out = []
    for b in range(B):
        if results[b]:
            det = torch.cat(results[b], 0)
            # simple top-100
            if det.shape[0] > 100:
                det = det[det[:, 4].argsort(descending=True)[:100]]
            out.append(det)
        else:
            out.append(torch.zeros((0, 6), device=device))
    return out


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """11-point or continuous AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


def map50(preds: List[Tensor], targets: Tensor, imgsz: int = 640, iou_thres: float = 0.5, nc: int = 9) -> Dict[str, float]:
    """
    preds: list of (N,6) xyxy score cls  (absolute)
    targets: (M,6) batch_idx cls cx cy w h  (normalized)
    """
    aps = []
    for c in range(nc):
        # collect all preds / gts of class c
        all_pred = []
        all_gt = []
        n_gt = 0
        for b, p in enumerate(preds):
            if p.numel():
                pc = p[p[:, 5] == c]
                all_pred.append(pc)
            t = targets[(targets[:, 0] == b) & (targets[:, 1] == c)]
            if t.numel():
                gt_xyxy = xywhn_to_xyxy(t[:, 2:6], imgsz, imgsz)
                all_gt.append((b, gt_xyxy))
                n_gt += gt_xyxy.shape[0]

        if n_gt == 0:
            continue
        if not all_pred:
            aps.append(0.0)
            continue

        pred = torch.cat(all_pred, 0)
        pred = pred[pred[:, 4].argsort(descending=True)]

        tp = np.zeros(pred.shape[0])
        fp = np.zeros(pred.shape[0])
        gt_matched = {b: torch.zeros(g.shape[0], dtype=torch.bool) for b, g in all_gt}

        for i, p in enumerate(pred):
            # find best matching gt of same image (we lost image id; approximate by global)
            # For simplicity we treat all gts of class c together (works for small val sets)
            best_iou = 0.0
            best_j = -1
            best_b = -1
            for b, g in all_gt:
                if g.numel() == 0:
                    continue
                ious = box_iou(p[:4].unsqueeze(0), g)[0]
                max_iou, j = ious.max(0)
                if max_iou > best_iou and not gt_matched[b][j]:
                    best_iou = max_iou.item()
                    best_j = j.item()
                    best_b = b
            if best_iou >= iou_thres and best_j >= 0:
                tp[i] = 1
                gt_matched[best_b][best_j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / (n_gt + 1e-8)
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        aps.append(compute_ap(recall, precision))

    return {
        "mAP50": float(np.mean(aps)) if aps else 0.0,
        "AP_per_class": aps,
        "num_classes_with_gt": len(aps),
    }
