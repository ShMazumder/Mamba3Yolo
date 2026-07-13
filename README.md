# Mamba3Yolo

**Mamba-3 enhanced real-time object detection** – drop-in upgrade of [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO) (AAAI 2025) that replaces the older SSM core with Mamba-3 (complex-valued states, exponential-trapezoidal discretization, MIMO).

Includes:
- Full integration patch for the original Ultralytics-based Mamba-YOLO repo
- Self-contained pure-PyTorch trainer that actually runs (tested)
- Medical multi-dataset configuration
- Grad-CAM++ + SSM-state XAI modules
- PTQ / QAT / ONNX export stubs ready for TensorRT
- Ready-to-copy **Methods** section written in human academic style

## Quick Start (self-contained, no external YOLO install)

```bash
cd Mamba3Yolo
python scripts/train.py --scale T --epochs 2 --batch 2 --imgsz 320 --device cpu
# or --device cuda if you have a GPU
```

This trains a ~6.8 M parameter model on synthetic data and saves checkpoints. Point `--data` at a real YOLO-format folder (images/ + labels/) for real experiments.

## Integration with Official Mamba-YOLO

See **docs/INTEGRATION.md** for the exact 5-line registration patch, YAML placement, and training command that re-uses the full Ultralytics pipeline + official Mamba-3 CUDA kernels.

## Medical Multi-Domain

`configs/datasets/medical_multi.yaml` defines a unified label space for polyp, blood-cell, brain-tumor and (optionally) DR-lesion detection. Convert your public medical detection sets to YOLO txt format and train jointly or via staged fine-tuning. The same model can then be deployed on heterogeneous clinical hardware.

## XAI

```python
from src.xai.gradcam import GradCAMPlusPlus, overlay_heatmap
# hook the last Mamba3ODSSBlock or neck layers
# produce heatmaps + overlays suitable for fundus / endoscopic / microscopy figures
```

## Quantization

```python
from src.quant.ptq import collect_activation_stats, quantize_model_dynamic, export_onnx_int8
# PTQ4VM / OuroMamba inspired activation handling for Mamba-3 layers
# full-detector INT8 export path
```

## Paper Methods Section

A complete, humanized Methods section (architecture, Mamba3SS2D equations, medical multi-domain, XAI, quantization, implementation notes) is in:

**docs/METHODS_SECTION.md**

Copy-paste ready for a CVPR/ICCV/MICCAI submission. Ablation plan is in `docs/RESEARCH_PLAN.md`.

## Project Layout

```
Mamba3Yolo/
├── README.md
├── configs/
│   ├── models/Mamba3Yolo-T.yaml
│   └── datasets/medical_multi.yaml
├── docs/
│   ├── INTEGRATION.md          # patch for original repo
│   ├── METHODS_SECTION.md      # paper-ready text
│   └── RESEARCH_PLAN.md
├── scripts/train.py            # runnable self-contained trainer
└── src/
    ├── blocks/mamba3_odss.py   # core Mamba3ODSSBlock
    ├── models/mamba3yolo.py    # full detector
    ├── xai/gradcam.py
    └── quant/ptq.py
```

## Citation

If you build on this, please cite Mamba-YOLO, Mamba-3, and this work once published.

```bibtex
@misc{mamba3yolo2026,
  title={Mamba3Yolo: Mamba-3 Enhanced Real-Time Object Detection with Explainability and Edge Quantization},
  author={...},
  year={2026},
  note={Code: https://github.com/...}
}
```

## License

AGPL-3.0 for the detection framework (inherited) + Apache-2.0 for the new Mamba-3 blocks and research modules.
