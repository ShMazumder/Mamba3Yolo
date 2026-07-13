# Full Integration Patch for Original Mamba-YOLO Repo

The original Mamba-YOLO is built on Ultralytics. To use the real Mamba-3 kernels and the full training/validation/export pipeline, apply this minimal patch.

## 1. Prerequisites

```bash
git clone https://github.com/HZAI-ZJNU/Mamba-YOLO.git
cd Mamba-YOLO

# Install base (as in their README)
conda create -n mamba3yolo python=3.11 -y
conda activate mamba3yolo
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install seaborn thop timm einops ultralytics
cd selective_scan && pip install . && cd ..
pip install -e .

# Critical: install official Mamba-3
pip install mamba-ssm   # or build from https://github.com/state-spaces/mamba
```

## 2. Drop the new block

```bash
# From this Mamba3Yolo repo
cp src/blocks/mamba3_odss.py /path/to/Mamba-YOLO/ultralytics/nn/modules/
```

## 3. Register the module (the actual patch)

Edit `ultralytics/nn/tasks.py` (or `ultralytics/nn/modules/__init__.py` + parse_model).

**A. In the imports section of tasks.py (or modules/__init__.py):**
```python
from ultralytics.nn.modules.mamba3_odss import Mamba3ODSSBlock, build_mamba3_odss
```

**B. Inside the parse_model function (the big if-elif that handles C2f, SPPF, etc.):**

Find a similar block (e.g. for C2f or ODSS) and add:

```python
elif m is Mamba3ODSSBlock:
    c1, c2 = ch[f], args[0]
    # args example from YAML: [256, 64, 2, 64, True, 4]  # dim, d_state, expand, headdim, is_mimo, mimo_rank
    args = [c2 if c2 != c1 else c1, *args[1:]] if len(args) > 0 else [c2]
    m_ = Mamba3ODSSBlock(*args) if c1 == c2 else nn.Sequential(
        nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(),
        Mamba3ODSSBlock(c2, *args[1:])
    )
```

Also add to the module name map if they have one:
```python
"Mamba3ODSSBlock": Mamba3ODSSBlock,
```

## 4. Use the provided YAML

Copy `configs/models/Mamba3Yolo-T.yaml` into `ultralytics/cfg/models/mamba-yolo/` and adjust any from/to indices if needed.

Then train exactly as the original README:

```bash
python mbyolo_train.py --task train \
    --data ultralytics/cfg/datasets/coco.yaml \
    --config ultralytics/cfg/models/mamba-yolo/Mamba3Yolo-T.yaml \
    --amp --project ./output_dir/mscoco --name mamba3yolo_t
```

## 5. Medical multi-dataset

Use `configs/datasets/medical_multi.yaml` (or convert your medical detection datasets into YOLO format folders and point the data yaml at them). For multi-domain sampling you can either:

- Concatenate the label folders and use a domain-balanced sampler, or
- Train sequentially (COCO pretrain → medical fine-tune).

## 6. XAI + Quant after training

- XAI: `from src.xai.gradcam import GradCAMPlusPlus, overlay_heatmap`
- Quant: `from src.quant.ptq import collect_activation_stats, quantize_model_dynamic, export_onnx_int8`

See the respective modules for usage examples.

## Verification

After the patch, a quick shape test:

```python
from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/mamba-yolo/Mamba3Yolo-T.yaml")
print(model.model)
x = torch.randn(1, 3, 640, 640)
y = model.model(x)
print([o.shape for o in y])
```

This is the complete, minimal integration path.
