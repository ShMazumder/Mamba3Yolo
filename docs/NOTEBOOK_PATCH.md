# Mamba3Yolo runbook — what broke and what to change

## What the log actually says

| Symptom in the log | Root cause |
|---|---|
| `box_loss nan` from epoch 2 in **all three** Mamba variants; baseline fine | `_scan_chunked` computes `Pin = cumprod(alpha).clamp(min=1e-6)` then `intra = Pin * cumsum(U / Pin)`. Once `alpha = exp(dt·A)` drops below ~0.6, `alpha**32` underflows fp16, the clamp bites, and `U / 1e-6` overflows fp16 → `inf` → `NaN`. |
| `mamba3_norope` reports the **identical** val metric (`1.73e-05 / 0.00255`) at every epoch from 2 to 100 | The weights never changed: AMP's GradScaler skipped *every* step because the grads were non-finite. The model in that table is the epoch-1 model. |
| XAI cell: `size of tensor a (16383) must match tensor b (16384)` | `F.pad(beta, (0,0,0,0,0,1))` — `beta` is 3-D `(B,L,H)`, so a 6-element pad tuple pads the **batch** dim. `beta_next` came out length `L-1`. |
| Baseline mAP50 only 0.083 on coco128 (train == val, 100 epochs) | `batch=2` with the default `nbs=64` → `accumulate = 32` → **2 optimizer steps per epoch**, 200 total. Nothing could learn, ours *or* baseline. |
| 2.06 h for 100 epochs on 128 images | The scan materialised the full `(B, L, H, P, N)` state trajectory (≈33 M elements for the stride-4 block alone) four times per block. |
| Params "12,777,932" | `self._official_mamba = Mamba3(...)` registers as a submodule regardless of the underscore, so if the kernel ever existed the reference + kernel would both be counted. (Moot at `HAS_MAMBA3=False`, but the comment claimed otherwise.) |
| XAI cell would have taken hours even without the crash | `_reverse_energy` is a Python loop over `L` — 16384 steps × 4 directions × 8 blocks. |

## Files to drop in (replace the `%%writefile` cell bodies in §1)

- `src/blocks/mamba3_ref.py` — bounded segment-sum ("SSD") chunked scan, fp32-forced core,
  scalar-per-head decay so the readout contracts analytically (no `(B,L,H,P,N)` tensor),
  bounded + wrapped rotary phase, fixed `token_saliency` padding, vectorised
  `_reverse_energy`, and a kept `_ssm_reference` O(L) loop as ground truth.
- `src/blocks/mamba3_odss.py` — one scan path constructed, not two.
- `src/xai/gramian.py` — self-tests updated, guards added.
- `scripts/validate_parity.py` — unchanged behaviour, documents the rotation bound.
- `scripts/validate_core.py` — **new**: the gate that would have caught all of this.

`scripts/ultra_mamba3.py` needs no change.

---

## §2a — NEW cell, run it before anything expensive

```python
!python scripts/validate_core.py
```

Expected: equivalence `rel < 1e-4` for all four (trapezoidal × rope) combos, finite
forward/backward at every decay setting, `token_saliency` → `(B, L)`. If this is red,
stop — nothing downstream means anything.

Then §2 (`validate_parity.py`) as before.

---

## §4 — replace the config cell

```python
DATA   = 'coco128.yaml'   # <-- your data.yaml for real numbers (COCO / medical)
EPOCHS = 100
IMGSZ  = 256              # keep BOTH models here for a fair compare
BATCH  = 8                # see note below; raise if memory allows
DEV    = 0

# nbs=BATCH is NOT cosmetic. Ultralytics accumulates gradients to a nominal batch of
# nbs (default 64): at batch=2 that is accumulate=32, i.e. TWO optimizer steps per
# epoch on coco128. The old run trained the baseline for ~200 steps and ours for ~0
# (every step was skipped on non-finite grads). accumulate=1 makes an epoch an epoch.
TKW = dict(optimizer='AdamW', lr0=2e-3, nbs=BATCH, warmup_epochs=5, cos_lr=True,
           plots=False, exist_ok=True)
```

**Memory note.** The old scan built the `(B,L,H,P,N)` state trajectory; the SSD scan
never does, so the ceiling that forced `batch=2, imgsz=256` is gone. Don't take my word
for it — probe before launching a 4-run sweep:

```python
import torch
from scripts.ultra_mamba3 import register, make_yaml
from ultralytics import YOLO
register()
m = YOLO(make_yaml('s')).model.cuda().train()
torch.cuda.reset_peak_memory_stats()
x = torch.randn(BATCH, 3, IMGSZ, IMGSZ, device='cuda', requires_grad=True)
with torch.autocast('cuda', dtype=torch.float16):
    out = m(x)
sum(o.float().pow(2).mean() for o in (out if isinstance(out, (list, tuple)) else [out])).backward()
print(f'peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB at batch={BATCH}, imgsz={IMGSZ}')
del m, x, out; torch.cuda.empty_cache()
```

Leave ~30% headroom over that number (the val loop and mosaic add some). If it's tight,
drop `BATCH` or pass `d_state=16` to every `make_yaml(...)`.

Everything else in §4a/§4b is unchanged — baseline and ours must keep sharing `TKW`,
`IMGSZ` and `BATCH`.

---

## §5 — replace the results cell

Report the **best** epoch, not the last one, and surface NaN explicitly instead of
letting `round(2.7e-05, 4)` print `0.0000` as if it were a real zero.

```python
import pandas as pd, os

rows = []
for run in ['yolo11s_base', 'mamba3_full', 'mamba3_norope', 'mamba3_euler']:
    csv = f'runs/detect/{run}/results.csv'
    if not os.path.exists(csv):
        continue
    df = pd.read_csv(csv); df.columns = [c.strip() for c in df.columns]
    loss_cols = [c for c in df.columns if c.startswith('train/')]
    nan_ep = df.index[df[loss_cols].isna().any(axis=1)]
    best = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
    rows.append({
        'run': run,
        'best_epoch': int(best['epoch']),
        'mAP50': f"{best['metrics/mAP50(B)']:.4g}",
        'mAP50-95': f"{best['metrics/mAP50-95(B)']:.4g}",
        'train_loss_nan': 'NONE' if len(nan_ep) == 0 else f'from ep{int(df.loc[nan_ep[0], "epoch"])}',
        'frozen': bool(df['metrics/mAP50-95(B)'].nunique() <= 2 and len(df) > 10),
    })
print(pd.DataFrame(rows).to_string(index=False) if rows else 'no runs yet')
```

`train_loss_nan` must read `NONE` and `frozen` must read `False` for every row. Any other
value means the run is not a result, it is a bug report.

---

## §6 — replace the XAI cell

Match the training resolution (the model was trained at 256; 512 also makes the stride-4
scan 16 384 tokens for no reason), run on GPU, and fix the dataset path.

```python
import torch, glob, cv2
import matplotlib.pyplot as plt
from ultralytics.data.augment import LetterBox
from scripts.ultra_mamba3 import register
from ultralytics import YOLO
from src.xai.gramian import gramian_saliency

register()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('runs/detect/mamba3_full/weights/best.pt').model.to(dev).eval()

imgs = sorted(glob.glob('/kaggle/working/datasets/coco128/images/train2017/*.jpg'))
assert imgs, 'no images found - check the datasets dir from `yolo settings`'
im0 = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2RGB)
im = LetterBox((IMGSZ, IMGSZ))(image=im0)
x = torch.from_numpy(im).permute(2, 0, 1).float()[None].to(dev) / 255.0

heat = gramian_saliency(model, x)[0].cpu().numpy()

fig, ax = plt.subplots(1, 3, figsize=(14, 5))
ax[0].imshow(im);                       ax[0].set_title('input')
ax[1].imshow(heat, cmap='jet');         ax[1].set_title('Gramian saliency')
ax[2].imshow(im); ax[2].imshow(heat, cmap='jet', alpha=0.5); ax[2].set_title('overlay')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()
```

---

## §7 — integrity checklist, amended

Two claims in the old checklist are no longer supportable as written:

- **"FLOPs ~3× the baseline."** 69.4 vs 21.7 GFLOPs is what `thop` counts, and `thop`
  only sees the `nn.Linear`/`nn.Conv2d` calls — it never counted the scan at all, in
  either version. The number didn't change because it was never measuring the thing you
  were apologising for. If you want an efficiency claim, measure wall-clock latency and
  peak memory at fixed `imgsz`/`batch` and report those instead.
- **"Every reported number is measured here."** True, and that is exactly why the old
  table had to be thrown away: the measured numbers were of a frozen model. Re-run
  everything after `validate_core.py` is green.

Add to the checklist:

- `validate_core.py` passes (chunked scan == O(L) recurrence; AMP forward/backward finite).
- No run has a NaN in `train/*` of `results.csv`.
- coco128 has train == val — it is a pipeline/overfit sanity check, never a paper number.
  With `nbs=BATCH` the baseline should now climb well past mAP50 ≈ 0.08; if it doesn't,
  the schedule is still starved and the ablation contrast is meaningless.
