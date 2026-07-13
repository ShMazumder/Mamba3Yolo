# Mamba3Yolo — Experiments Runbook

Turnkey steps to produce every number the paper needs, on GPU. Everything here is
wired and smoke-tested on CPU; this is the recipe to scale it up. **Report only
numbers you actually measure. No placeholder tables.** (See the integrity check at the end.)

Paper's two defensible claims:
1. **First Mamba-3 (complex-state, trapezoidal) detector.**
2. **Intrinsic controllability-Gramian explainability**, quantitatively more faithful than Grad-CAM.

---

## 0. Environment

```bash
pip install ultralytics==8.3.0 pymupdf
# torch with CUDA matching your GPU. mamba-ssm is NOT required (pure-PyTorch core runs).
```
Status of the official kernel: `HAS_MAMBA3=False` (no released Mamba-3 CUDA kernel).
The verified pure-PyTorch core is what runs. State this honestly in the paper.

**Every ultralytics command that uses a Mamba yaml must `register()` first** (it patches
`parse_model` in-process so the yaml parses). The one-liners below do it.

---

## 1. Data

Ultralytics YOLO format. Example `data.yaml`:
```yaml
path: /abs/path/dataset
train: images/train
val:   images/val
nc: 80
names: [ ... ]
```
Targets: COCO2017 (matches Mamba-YOLO / AKCMamba baselines) and/or the medical
multi-domain set (`configs/datasets/medical_multi.yaml`). Sanity-run on `coco8.yaml` first.

---

## 2. Core comparison (the main table)

Same base (YOLO11-s), same data, same schedule — **only the mixer differs**.

```bash
# Baseline: stock YOLO11-s (C3k2 mixer)
python -c "from ultralytics import YOLO; YOLO('yolo11s.yaml').train(\
data='data.yaml', epochs=300, imgsz=640, batch=32, device=0, name='yolo11s_base', cos_lr=True)"

# Ours: YOLO11-s with Mamba-3 mixer
python -c "from scripts.ultra_mamba3 import register,make_yaml; register(); \
from ultralytics import YOLO; YOLO(make_yaml('s')).train(\
data='data.yaml', epochs=300, imgsz=640, batch=32, device=0, name='yolo11s_mamba3', cos_lr=True)"
```
Compare against published Mamba-YOLO-T (45.4 mAP / 6.1M) and AKCMamba-YOLO (46.3 / 9.1M),
but the **controlled** row is baseline-vs-ours above (same framework).

Also worth a run: **Mamba-YOLO's exact base (YOLOv8)** as a secondary comparison if a
reviewer asks — swap `yolo11s` → `yolov8s` in both commands.

### Table 1 skeleton (fill from `runs/detect/*/results.csv`, val split)
| Model | mixer | Params(M) | GFLOPs | mAP50 | mAP50-95 | FPS(GPU) |
|---|---|---|---|---|---|---|
| YOLO11-s (baseline) | C3k2 | 9.46 | 21.7 | | | |
| YOLO11-s-Mamba3 (ours) | Mamba3ODSS | 12.78 | 69.4 | | | |
| Mamba-YOLO-T [ref] | ODSS(S6) | 6.1 | 14.3 | 45.4 | — | |
| AKCMamba-YOLO [ref] | AKSS2D | 9.1 | 14.9 | 46.3 | — | |

⚠ **FLOPs are 69.4 vs 21.7 — ~3×.** Confront this: report it, and either frame the
efficiency claim around params/accuracy (not FLOPs/latency), or reduce block cost.
Do not claim "efficient real-time" on latency without measuring it (Section 5).

---

## 3. Mechanism ablations (proves it's Mamba-3, not just "a mixer")

Three variants build from one function; toggles are baked into the yaml:

```bash
python - <<'PY'
from scripts.ultra_mamba3 import register, make_yaml
from ultralytics import YOLO
register()
runs = {
  "mamba3_full":   make_yaml("s"),                       # RoPE + trapezoidal (full Mamba-3)
  "mamba3_norope": make_yaml("s", use_rope=False),       # real-valued state (no complex/RoPE)
  "mamba3_euler":  make_yaml("s", trapezoidal=False),    # lambda=1 = exp-Euler = Mamba-2
}
for name, yaml_path in runs.items():
    YOLO(yaml_path).train(data="data.yaml", epochs=300, imgsz=640, batch=32,
                          device=0, name=name, cos_lr=True)
PY
```

### Table 2 skeleton
| Variant | complex(RoPE) | trapezoidal | mAP50-95 | Δ vs full |
|---|---|---|---|---|
| Mamba-3 full (ours) | ✓ | ✓ | | 0 |
| − complex state | ✗ | ✓ | | |
| − trapezoidal (= Mamba-2) | ✓ | ✗ | | |

Expected story: removing either mechanism drops mAP → each contributes. If they don't,
that's an honest finding to report, not hide.

State-tracking sanity (already passing, cite in paper): `python scripts/validate_parity.py`
(complex solves parity ~1.0, real-valued ~0.5).

---

## 4. Explainability eval (the novel contribution)

Method: `src/xai/gramian.py::gramian_saliency(model, img) -> (B,H,W)`. Math verified
(`python src/xai/gramian.py`). It only becomes a *good* explainer on a **trained** model.

Run on the trained `yolo11s_mamba3/weights/best.pt`:
```python
from scripts.ultra_mamba3 import register; register()
from ultralytics import YOLO
from src.xai.gramian import gramian_saliency
import torch
m = YOLO("runs/detect/yolo11s_mamba3/weights/best.pt").model.eval()
img = ...  # (1,3,H,W) preprocessed
heat = gramian_saliency(m, img)   # (1,H,W) in [0,1]
```

Compare against Grad-CAM++ (`src/xai/gradcam.py`) with faithfulness metrics AKCMamba
never reports:
- **Insertion / Deletion AUC** — add/remove pixels by saliency rank, measure confidence curve.
- **Pointing Game** — does the saliency peak land inside the GT box.
- Report both for Gramian vs Grad-CAM++ on a val subset (~500 imgs).

### Table 3 skeleton
| Method | Deletion AUC↓ | Insertion AUC↑ | Pointing Game↑ | Backprop? | Passes/img |
|---|---|---|---|---|---|
| Grad-CAM++ | | | | yes | 1 fwd + 1 bwd |
| Gramian (ours) | | | | no | 1 fwd |

(You'll need to write the insertion/deletion + pointing-game loop — standard XAI eval,
~150 lines. Flag if you want me to build it.)

---

## 5. Latency / FLOPs (measure, don't guess)

```bash
# GFLOPs + params print during training header, or:
python -c "from scripts.ultra_mamba3 import register,make_yaml; register(); \
from ultralytics import YOLO; YOLO(make_yaml('s')).info(detailed=False)"
```
Latency: `scripts/measure_latency.py` (PyTorch / ONNX / TensorRT). Measure on your target
(V100 for parity with AKCMamba; Jetson Orin if you keep the edge framing). **The O(L) scan
is FLOP-heavy — expect the latency gap. Report the real number.** If it's too slow, that's
the motivation for a future Triton kernel (say so; don't fake the speed).

---

## 6. What to log / keep
- `results.csv` per run (ultralytics writes it) — the source of truth for tables.
- `best.pt` for each run (base, mamba3_full, norope, euler) — XAI + latency need them.
- Enable W&B if you want curves: `yolo settings wandb=True`.
- Seeds: ultralytics `seed=0` default; run ≥2 seeds for the main comparison if time allows.

---

## 7. Integrity checklist (before writing anything)
- [ ] Every table cell is a number you measured. No carried-over/estimated values.
- [ ] `official_mamba3_kernels` reported as **False** (it is). Never hardcode True.
- [ ] FLOPs (3× baseline) stated openly; efficiency claim scoped to what's measured.
- [ ] Ablations run to the same schedule as the main model.
- [ ] Old fabricated artifacts deleted: `runs/mamba3yolo/medical_0713_1230/paper_summary.json`
      and any table using its numbers. The Gemini-report tables were fabricated — do not cite.
- [ ] Baseline and ours differ ONLY in the mixer (same base/data/schedule/aug).

---

## Gotchas
- **`register()` before `YOLO(<mamba yaml>)`** every session — the module patch is in-process.
- Surgery-on-object (`build_yolo11_mamba3.py`) is for eval/inspection only; **training must
  go through the registered yaml** or ultralytics rebuilds to stock (silently loses Mamba).
- Scan speed: fine to train at `imgsz=512` to cut cost (note it; small-object mAP may dip).
- Param target: block floors ~11.5M at -s; exact 9.1M match needs a narrower custom yaml.
