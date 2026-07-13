# Recommended Datasets for Mamba3Yolo Paper + Exact Kaggle Links

## 1. Exact Kaggle Dataset Links (as of July 2026)

### A. MS COCO 2017 (YOLO format) — **Primary benchmark**
| Name | Link | Notes |
|------|------|-------|
| COCO Dataset for Yolo | https://www.kaggle.com/datasets/sarkisshilgevorkyan/coco-dataset-for-yolo | Good starting point |
| Search on Kaggle | “COCO YOLO format” or “COCO 2017 YOLOv8” | Many mirrors; pick one with `images/train2017` + `labels/train2017` |

### B. VisDrone 2019 (YOLO format) — **Small objects / hard case**
| Name | Link | Notes |
|------|------|-------|
| VisDrone Dataset | https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset | Original + annotations |
| VisDrone YOLO format | https://www.kaggle.com/datasets/thnhvlngc/visdrone-dataset-yolo-format | Ready YOLO labels |
| Modified Aerial + VisDrone YOLO | https://www.kaggle.com/datasets/redzapdos123/modified-aerial-traffic-and-visdrone-dataset-yolo | Preprocessed |

Ultralytics also auto-downloads VisDrone when you use `data="VisDrone.yaml"`.

### C. Medical Multi-Domain
| Domain | Dataset | Kaggle Link | Notes |
|--------|---------|-------------|-------|
| **Polyp** | Kvasir-SEG | https://www.kaggle.com/datasets/debeshjha1/kvasirseg | Masks → convert to boxes |
| **Polyp YOLO** | YOLO-Kvasir | https://www.kaggle.com/datasets/arfanakbar/yolo-kvasir | Already YOLO labels |
| **Blood Cell** | BCCD | Search “BCCD YOLO” or “Blood Cell Detection YOLO” | Classic, easy |
| **Brain Tumor** | Brain_Tumor (YOLO) | https://www.kaggle.com/datasets/abhit007pandey/brain-tumor-yolo | Ready YOLO, 3 classes |
| **Brain Tumor** | Br35H | https://www.kaggle.com/datasets/sushreeswain/brain-tumour-br35h | Images + masks |

### Quick Add-Data commands on Kaggle
In the notebook sidebar → **Add Data** → search the names above → Add.

---

## 2. Ready Download / Setup Cells (copy-paste into notebook)

```python
# ============================================================
# CELL: Dataset paths (edit after adding datasets on Kaggle)
# ============================================================
from pathlib import Path

DATASETS = {
    # Primary
    "coco": Path("/kaggle/input/coco-dataset-for-yolo"),          # change to your actual folder name
    "visdrone": Path("/kaggle/input/visdrone-dataset-yolo-format"),
    
    # Medical
    "polyp": Path("/kaggle/input/yolo-kvasir"),                   # or kvasirseg
    "bccd": Path("/kaggle/input/bccd-yolo"),                     # create/search
    "br35h": Path("/kaggle/input/brain-tumor-yolo"),
}

# Check what is available
print("Available datasets:")
for name, p in DATASETS.items():
    exists = p.exists()
    print(f"  {name:12s}: {'✅' if exists else '❌'}  {p}")
```

```python
# ============================================================
# CELL: Create a unified multi-domain medical folder (optional)
# ============================================================
import shutil
from pathlib import Path

medical_root = Path("/kaggle/working/medical_multi")
medical_root.mkdir(exist_ok=True)

domains = {
    "polyp": DATASETS.get("polyp"),
    "bccd":  DATASETS.get("bccd"),
    "br35h": DATASETS.get("br35h"),
}

for dom, src in domains.items():
    if src is None or not src.exists():
        print(f"Skip {dom} (not found)")
        continue
    dst_img = medical_root / dom / "images"
    dst_lbl = medical_root / dom / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    
    # Copy (or symlink) a limited number for testing
    imgs = list(src.rglob("*.jpg")) + list(src.rglob("*.png"))
    print(f"{dom}: found {len(imgs)} images")
    # For full paper you can copy everything; for Kaggle speed limit to e.g. 500
    for img in imgs[:500]:
        shutil.copy(img, dst_img / img.name)
        # try to find matching label
        for lbl_cand in [img.with_suffix(".txt"), 
                         src / "labels" / (img.stem + ".txt"),
                         src.parent / "labels" / (img.stem + ".txt")]:
            if lbl_cand.exists():
                shutil.copy(lbl_cand, dst_lbl / (img.stem + ".txt"))
                break

print("Medical multi-domain ready at:", medical_root)
```

```python
# ============================================================
# CELL: Set CFG for different experiments
# ============================================================
# Experiment 1: COCO main results
CFG["data_root"] = str(DATASETS["coco"])
CFG["nc"] = 80
CFG["max_train_samples"] = 8000          # or None for full

# Experiment 2: VisDrone (small objects)
# CFG["data_root"] = str(DATASETS["visdrone"])
# CFG["nc"] = 10                          # VisDrone has 10 classes

# Experiment 3: Medical multi-domain
# CFG["data_root"] = "/kaggle/working/medical_multi"
# CFG["nc"] = 9                           # or whatever unified classes you map
```

---

## 3. Exact Table Templates for the Paper

### Table 1 — Main Results on COCO val2017
```latex
\begin{table}[t]
\centering
\caption{Main results on COCO val2017. FPS measured on NVIDIA T4 (TensorRT FP16, batch=1).}
\label{tab:coco_main}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccc}
\toprule
Method & Params (M) & FLOPs (G) & mAP$^{50:95}$ & mAP$^{50}$ & AP$_s$ & FPS \\
\midrule
YOLOv8n          & 3.2  & 8.7  & 37.3 & 52.6 & 18.9 & 312 \\
YOLOv8s          & 11.2 & 28.6 & 44.9 & 61.8 & 25.7 & 216 \\
RT-DETRv2-S      & 20.0 & 60.0 & 47.9 & 65.8 & --   & 108 \\
Mamba-YOLO-T     & 5.8  & 13.2 & 44.5 & 61.2 & 24.7 & 198 \\
Mamba-YOLO-B     & 19.1 & 45.4 & 49.1 & 66.5 & 30.6 & 97  \\
\midrule
Mamba3Yolo-T (SISO) & 6.1 & 13.8 & --.-- & --.-- & --.-- & --- \\
\textbf{Mamba3Yolo-T (MIMO)} & 6.8 & 14.5 & \textbf{--.--} & \textbf{--.--} & \textbf{--.--} & --- \\
Mamba3Yolo-M         & 20.4 & 47.1 & --.-- & --.-- & --.-- & --- \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 2 — VisDrone (Small Objects)
```latex
\begin{table}[t]
\centering
\caption{Results on VisDrone2019-val. Emphasis on small-object performance.}
\label{tab:visdrone}
\begin{tabular}{lcccc}
\toprule
Method & mAP$^{50:95}$ & mAP$^{50}$ & AP$_s$ & FPS \\
\midrule
YOLOv8n          & --.-- & --.-- & --.-- & --- \\
Mamba-YOLO-T     & --.-- & --.-- & --.-- & --- \\
\textbf{Mamba3Yolo-T} & \textbf{--.--} & \textbf{--.--} & \textbf{--.--} & --- \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 3 — Medical Multi-Domain
```latex
\begin{table}[t]
\centering
\caption{Multi-domain medical detection (mAP$^{50}$). One model trained jointly.}
\label{tab:medical}
\begin{tabular}{lcccc}
\toprule
Method & Polyp & Blood Cell & Brain Tumor & Average \\
\midrule
YOLOv8n          & --.-- & --.-- & --.-- & --.-- \\
Mamba-YOLO-T     & --.-- & --.-- & --.-- & --.-- \\
\textbf{Mamba3Yolo-T} & \textbf{--.--} & \textbf{--.--} & \textbf{--.--} & \textbf{--.--} \\
\quad + INT8 PTQ & --.-- & --.-- & --.-- & --.-- \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 4 — Ablation (on COCO or VisDrone)
```latex
\begin{table}[t]
\centering
\caption{Ablation of Mamba-3 components (Tiny scale).}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Variant & mAP$^{50:95}$ & AP$_s$ & FPS \\
\midrule
Mamba-YOLO baseline (S6)     & --.-- & --.-- & --- \\
+ Exp-trapezoidal only       & --.-- & --.-- & --- \\
+ Complex states             & --.-- & --.-- & --- \\
+ MIMO ($r=4$)               & \textbf{--.--} & \textbf{--.--} & --- \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 5 — Latency / Edge
```latex
\begin{table}[t]
\centering
\caption{Latency and throughput (batch=1, 640$\times$640).}
\label{tab:latency}
\begin{tabular}{lcccc}
\toprule
Platform & Backend & Latency (ms) & FPS & Peak Mem \\
\midrule
RTX 3090     & PyTorch AMP   & --.-- & --- & -- GB \\
T4 (Kaggle)  & TensorRT FP16 & --.-- & --- & -- GB \\
Orin NX      & TensorRT FP16 & --.-- & --- & -- GB \\
Orin NX      & TensorRT INT8 & --.-- & --- & -- GB \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Recommended Order of Experiments on Kaggle

1. **COCO** (even 5k–10k images) → get main table numbers
2. **VisDrone** → small-object table
3. **Medical multi-domain** (start with BCCD + Br35H + one polyp set)
4. Ablations on the fastest dataset
5. Quantization + latency on the final model
6. XAI figures on medical images

This combination gives you a complete, high-impact paper.
