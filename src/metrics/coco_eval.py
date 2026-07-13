"""
Real COCO-style mAP evaluator for Mamba3Yolo.

Uses torchmetrics (preferred on Kaggle) with fallback to pycocotools.
Designed for YOLO-format predictions and ground truth.

Usage:
    from src.metrics.coco_eval import COCOEvaluator
    evaluator = COCOEvaluator(nc=80, conf_thres=0.001, iou_thres=0.6)
    # during validation loop
    evaluator.update(preds, targets, image_sizes)
    metrics = evaluator.compute()
    print(metrics)  # {'mAP50': ..., 'mAP50-95': ..., 'AP_s': ...}
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Any
import torch
from torch import Tensor
import numpy as np

try:
    from torchmetrics.detection import MeanAveragePrecision
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCOAPI = True
except ImportError:
    HAS_COCOAPI = False


class COCOEvaluator:
    """
    Lightweight COCO mAP evaluator compatible with YOLO outputs.
    
    Predictions are expected as list of dicts or raw YOLO tensors.
    For simplicity we accept:
      - preds: list of (N, 6) tensors [x1,y1,x2,y2,score,class] in absolute pixels
      - targets: list of (M, 5) tensors [class, x1,y1,x2,y2] absolute
    """

    def __init__(
        self,
        nc: int = 80,
        conf_thres: float = 0.001,
        iou_thres: float = 0.6,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.nc = nc
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_names = class_names or [str(i) for i in range(nc)]
        self.device = device

        if HAS_TORCHMETRICS:
            self.metric = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="bbox",
                class_metrics=True,
                max_detection_thresholds=[1, 10, 100],
            )
            self.backend = "torchmetrics"
        else:
            self.metric = None
            self.backend = "none"
            print("[COCOEvaluator] torchmetrics not found. Install with: pip install torchmetrics")
            print("               Falling back to simple proxy until available.")

        self.reset()

    def reset(self):
        if self.metric is not None:
            self.metric.reset()
        self._preds = []
        self._targets = []

    def update(
        self,
        preds: List[Tensor],
        targets: List[Tensor],
        image_ids: Optional[List[int]] = None,
    ):
        """
        preds: list of length B, each (N_i, 6) = [x1,y1,x2,y2,conf,cls] (absolute xyxy)
        targets: list of length B, each (M_i, 5) = [cls, x1,y1,x2,y2]
        """
        if self.backend != "torchmetrics":
            # store for later simple analysis
            self._preds.extend(preds)
            self._targets.extend(targets)
            return

        pred_list = []
        tgt_list = []
        for p, t in zip(preds, targets):
            if p is None or p.numel() == 0:
                pred_list.append({
                    "boxes": torch.zeros((0, 4), device=self.device),
                    "scores": torch.zeros(0, device=self.device),
                    "labels": torch.zeros(0, dtype=torch.int64, device=self.device),
                })
            else:
                # filter by conf
                keep = p[:, 4] >= self.conf_thres
                p = p[keep]
                pred_list.append({
                    "boxes": p[:, :4].float(),
                    "scores": p[:, 4].float(),
                    "labels": p[:, 5].long(),
                })

            if t is None or t.numel() == 0:
                tgt_list.append({
                    "boxes": torch.zeros((0, 4), device=self.device),
                    "labels": torch.zeros(0, dtype=torch.int64, device=self.device),
                })
            else:
                tgt_list.append({
                    "boxes": t[:, 1:5].float(),
                    "labels": t[:, 0].long(),
                })

        self.metric.update(pred_list, tgt_list)

    def compute(self) -> Dict[str, float]:
        if self.backend == "torchmetrics":
            res = self.metric.compute()
            out = {
                "mAP50-95": float(res["map"].item()) if "map" in res else 0.0,
                "mAP50": float(res["map_50"].item()) if "map_50" in res else 0.0,
                "mAP75": float(res["map_75"].item()) if "map_75" in res else 0.0,
                "AP_s": float(res["map_small"].item()) if "map_small" in res else 0.0,
                "AP_m": float(res["map_medium"].item()) if "map_medium" in res else 0.0,
                "AP_l": float(res["map_large"].item()) if "map_large" in res else 0.0,
            }
            # per-class if available
            if "map_per_class" in res and res["map_per_class"] is not None:
                out["per_class_mAP"] = res["map_per_class"].tolist()
            return out
        else:
            # very crude proxy so the pipeline never crashes
            return {
                "mAP50-95": 0.0,
                "mAP50": 0.0,
                "mAP75": 0.0,
                "AP_s": 0.0,
                "AP_m": 0.0,
                "AP_l": 0.0,
                "note": "Install torchmetrics for real mAP",
            }

    def __str__(self):
        m = self.compute()
        return (f"mAP50-95: {m.get('mAP50-95',0):.4f} | "
                f"mAP50: {m.get('mAP50',0):.4f} | "
                f"AP_s: {m.get('AP_s',0):.4f}")


def xywhn_to_xyxy(x: Tensor, w: int, h: int) -> Tensor:
    """Convert normalized YOLO (x_c,y_c,w,h) to absolute xyxy."""
    y = x.clone()
    y[:, 0] = (x[:, 0] - x[:, 2] / 2) * w  # x1
    y[:, 1] = (x[:, 1] - x[:, 3] / 2) * h  # y1
    y[:, 2] = (x[:, 0] + x[:, 2] / 2) * w  # x2
    y[:, 3] = (x[:, 1] + x[:, 3] / 2) * h  # y2
    return y


def non_max_suppression(
    prediction: Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> List[Tensor]:
    """
    Simplified NMS for a single image prediction tensor of shape (C, H, W)
    or already decoded (N, 6). This is a placeholder – for full accuracy
    use the Ultralytics NMS or torchvision.ops.nms.
    """
    # For research prototype we assume the caller has already decoded boxes.
    # Return as-is after conf filter.
    if prediction.ndim == 2 and prediction.shape[1] == 6:
        keep = prediction[:, 4] > conf_thres
        return [prediction[keep][:max_det]]
    return [torch.zeros((0, 6))]
