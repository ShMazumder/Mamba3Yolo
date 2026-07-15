#!/usr/bin/env python3
"""Assemble a self-contained Kaggle notebook from the verified current source files.

Embeds the corrected core files (repo on GitHub still has the old code) via %%writefile
so the notebook runs the real, verified pipeline regardless of repo state.
Run:  python scripts/make_kaggle_notebook.py
Out:  notebooks/Mamba3Yolo_Kaggle_Runbook.ipynb
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMBED = [
    "src/blocks/mamba3_ref.py",
    "src/blocks/mamba3_odss.py",
    "scripts/ultra_mamba3.py",
    "src/xai/gramian.py",
    "scripts/validate_parity.py",
]

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text}
def code(text): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}
def writefile(relpath):
    content = (ROOT / relpath).read_text(encoding="utf-8")
    return code(f"%%writefile {relpath}\n{content}")

cells = []

cells.append(md(
"""# Mamba3Yolo — Kaggle Runbook

**First Mamba-3 (complex-state, exponential-trapezoidal) object detector + intrinsic controllability-Gramian explainability.**

Before running: **Settings → Accelerator: GPU (T4/P100)** and **Internet: ON**.

This notebook is self-contained. It clones the repo, then overwrites the core files with the
**verified** versions (the GitHub repo may lag behind). It then:
1. runs the parity state-tracking gate (proof the Mamba-3 core is real),
2. smoke-trains on `coco8` through the real Ultralytics pipeline,
3. trains **baseline vs ours** + mechanism ablations (RoPE / trapezoidal),
4. demos the Gramian explainability on the trained model,
5. collects results into tables.

> Honesty: report only numbers you measure. `official_mamba3_kernels=False` (no released kernel — the pure-PyTorch verified core runs). FLOPs are ~3× the baseline; state it."""))

cells.append(md(
"""## 0. Environment

**Run this cell FIRST.** `ultralytics==8.3.0` pins `numpy<2`, but Kaggle's image is built for
numpy 2.x — a normal install downgrades numpy to 1.26.4 and ABI-breaks cv2/tifffile/shap
(`numpy.dtype size changed, Expected 96 got 88`). So we install ultralytics with **`--no-deps`**:
it keeps Kaggle's numpy 2.x untouched, and ultralytics 8.3.0 runs fine on it.

> If you already ran a bad install this session (numpy shows 1.26.4 below), do
> **Run → Factory reset** first to restore the clean numpy-2.x image, then run this cell."""))
cells.append(code(
"""import importlib, subprocess, sys
def have(m):
    try:
        importlib.import_module(m); return True
    except Exception:
        return False
if not have("ultralytics"):
    # --no-deps: do NOT let pip downgrade Kaggle's numpy 2.x (that breaks cv2's ABI).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "ultralytics==8.3.0", "ultralytics-thop"], check=False)
import ultralytics, numpy, cv2
print("ultralytics", ultralytics.__version__, "| numpy", numpy.__version__, "| cv2", cv2.__version__)"""))
cells.append(code("!nvidia-smi -L\nimport torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"))
cells.append(code(
"""import os
if not os.path.exists('Mamba3Yolo'):
    !git clone -q https://github.com/ShMazumder/Mamba3Yolo.git
os.chdir('Mamba3Yolo')
os.makedirs('src/xai', exist_ok=True); os.makedirs('scripts', exist_ok=True)
print('cwd:', os.getcwd())"""))

cells.append(md("## 1. Patch in the verified source files\n"
                "The repo on GitHub predates this work; these cells write the real, verified code."))
for rel in EMBED:
    cells.append(writefile(rel))
cells.append(code("import sys; sys.path.insert(0, os.getcwd())\nprint('patched files in place')"))

cells.append(md("## 2. Honesty gate — parity / state-tracking\n"
                "Complex (RoPE) state must solve running-parity (~1.0); the real-valued control must fail (~0.5). "
                "This is the proof the Mamba-3 core actually tracks state."))
cells.append(code("!python scripts/validate_parity.py"))

cells.append(md("## 3. Smoke test — real Ultralytics pipeline on coco8\n"
                "Confirms the Mamba-3 model trains through the genuine DFL loss + assigner + NMS + COCO mAP val, "
                "and **survives** Ultralytics' yaml rebuild (surgery-on-object does not)."))
cells.append(code(
"""from scripts.ultra_mamba3 import register, make_yaml
from ultralytics import YOLO
from src.blocks.mamba3_ref import Mamba3RefSSM
register()
y = YOLO(make_yaml('s'))
print('Mamba blocks before train:', sum(isinstance(m, Mamba3RefSSM) for m in y.model.modules()))
y.train(data='coco8.yaml', epochs=3, imgsz=320, batch=8, device=0, workers=2,
        plots=False, verbose=False, exist_ok=True, name='coco8_smoke')
print('Mamba blocks after train :', sum(isinstance(m, Mamba3RefSSM) for m in y.model.modules()))"""))

cells.append(md(
"""## 4. Main comparison + ablations (configure below)

Same base (YOLO11-s), same data/schedule — **only the mixer differs**.

**Kaggle note:** this model is ~69 GFLOPs (the O(L) scan is heavy). Full COCO/300 epochs will not
fit a single Kaggle session. Defaults below use `coco128` for a *real but small* demo that produces
genuine numbers. For paper numbers, point `DATA` at your full dataset and raise `EPOCHS` on adequate
compute (or resume across sessions)."""))
cells.append(code(
"""DATA   = 'coco128.yaml'   # <-- your data.yaml for real numbers (COCO / medical)
EPOCHS = 60
IMGSZ  = 512              # 640 for paper; 512 is faster on Kaggle T4
BATCH  = 16
DEV    = 0"""))

cells.append(md("### 4a. Baseline — stock YOLO11-s (C3k2 mixer)"))
cells.append(code(
"""from ultralytics import YOLO
YOLO('yolo11s.yaml').train(data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEV,
                           cos_lr=True, plots=False, exist_ok=True, name='yolo11s_base')"""))

cells.append(md("### 4b. Ours + ablations — Mamba-3 full / no-RoPE / Euler(=Mamba-2)"))
cells.append(code(
"""from scripts.ultra_mamba3 import register, make_yaml
from ultralytics import YOLO
register()
variants = {
    'mamba3_full':   make_yaml('s'),                    # RoPE + trapezoidal
    'mamba3_norope': make_yaml('s', use_rope=False),    # - complex state
    'mamba3_euler':  make_yaml('s', trapezoidal=False), # - trapezoidal (= Mamba-2)
}
for name, ycfg in variants.items():
    print('=== training', name, '===')
    YOLO(ycfg).train(data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEV,
                     cos_lr=True, plots=False, exist_ok=True, name=name)"""))

cells.append(md("## 5. Results table\nPulls final val metrics from each run's `results.csv`."))
cells.append(code(
"""import pandas as pd, glob, os
rows = []
for run in ['yolo11s_base', 'mamba3_full', 'mamba3_norope', 'mamba3_euler']:
    csv = f'runs/detect/{run}/results.csv'
    if os.path.exists(csv):
        df = pd.read_csv(csv); df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        rows.append({'run': run,
                     'mAP50': round(float(last.get('metrics/mAP50(B)', float('nan'))), 4),
                     'mAP50-95': round(float(last.get('metrics/mAP50-95(B)', float('nan'))), 4)})
print(pd.DataFrame(rows).to_string(index=False) if rows else 'no runs yet')"""))

cells.append(md("## 6. Gramian explainability on the trained model\n"
                "Intrinsic controllability-energy saliency — one forward pass, no backprop. "
                "Only meaningful on a **trained** model."))
cells.append(code(
"""import torch, numpy as np, glob
import matplotlib.pyplot as plt
from scripts.ultra_mamba3 import register
from ultralytics import YOLO
from src.xai.gramian import gramian_saliency
register()
ckpt = 'runs/detect/mamba3_full/weights/best.pt'
model = YOLO(ckpt).model.eval()

# grab one val image
imgs = sorted(glob.glob('datasets/coco128/images/train2017/*.jpg'))[:1] or \\
       sorted(glob.glob('/kaggle/working/Mamba3Yolo/datasets/**/*.jpg', recursive=True))[:1]
from ultralytics.data.augment import LetterBox
import cv2
im0 = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2RGB)
im = LetterBox((512, 512))(image=im0)
x = torch.from_numpy(im).permute(2, 0, 1).float()[None] / 255.0
with torch.no_grad():
    heat = gramian_saliency(model, x)[0].cpu().numpy()

fig, ax = plt.subplots(1, 3, figsize=(14, 5))
ax[0].imshow(im); ax[0].set_title('input'); ax[0].axis('off')
ax[1].imshow(heat, cmap='jet'); ax[1].set_title('Gramian saliency'); ax[1].axis('off')
ax[2].imshow(im); ax[2].imshow(heat, cmap='jet', alpha=0.5); ax[2].set_title('overlay'); ax[2].axis('off')
plt.tight_layout(); plt.show()"""))

cells.append(md(
"""## 7. Integrity checklist
- Every reported number is measured here — no placeholders.
- `official_mamba3_kernels = False` (verified pure-PyTorch core).
- FLOPs ~3× baseline — report openly; scope any efficiency claim to what you measure.
- Ablations use the same schedule as the main model.
- Do **not** cite the old fabricated tables (Gemini report / `paper_summary.json`).

For the XAI faithfulness metrics (insertion/deletion, pointing-game) vs Grad-CAM++, and full-COCO
training, see `docs/EXPERIMENTS_RUNBOOK.md`."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = ROOT / "notebooks" / "Mamba3Yolo_Kaggle_Runbook.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "|", len(cells), "cells")
