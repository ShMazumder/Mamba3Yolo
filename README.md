# Mamba3Yolo

**Mamba-3 enhanced real-time object detection**  
GitHub: https://github.com/ShMazumder/Mamba3Yolo

Upgrade of Mamba-YOLO (AAAI 2025) that replaces the SSM core with **Mamba-3** (complex-valued states, exponential-trapezoidal discretization, MIMO). Includes full paper-experiment tooling.

## Quick Start (Kaggle / Local)

```bash
git clone https://github.com/ShMazumder/Mamba3Yolo.git
cd Mamba3Yolo
python scripts/train.py --scale T --epochs 10 --batch 4 --imgsz 640 --device cuda
```

## Paper Experiment Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/Mamba3Yolo_Kaggle_Paper_Experiments.ipynb` | Full COCO-style training, ablations, curves, quant, summary |
| `notebooks/Mamba3Yolo_Medical_MultiDomain.ipynb` | Joint polyp + blood-cell + brain-tumor training |

## Key Scripts

- `scripts/train.py` – self-contained trainer
- `scripts/measure_latency.py` – PyTorch / ONNX / TensorRT FPS + Jetson notes
- `scripts/generate_xai_figures.py` – Grad-CAM++ paper figures on real images

## Metrics & Modules

- `src/metrics/coco_eval.py` – real mAP50 / mAP50-95 / AP_s via torchmetrics
- `src/xai/gradcam.py` – Grad-CAM++ + SSM saliency
- `src/quant/ptq.py` – PTQ helpers + ONNX export

## LaTeX for Paper

- `docs/METHODS_SECTION.md` – complete Methods
- `docs/latex/related_work_and_results.tex` – Related Work + Results with tables (ready for Overleaf)

## Integration with Original Mamba-YOLO

See `docs/INTEGRATION.md` (5-line Ultralytics registration patch).

## Citation

```bibtex
@misc{mamba3yolo2026,
  title={Mamba3Yolo: Mamba-3 Enhanced Real-Time Object Detection},
  author={ShMazumder et al.},
  year={2026},
  url={https://github.com/ShMazumder/Mamba3Yolo}
}
```
