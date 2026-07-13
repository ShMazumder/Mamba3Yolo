#!/usr/bin/env python3
"""
Latency / FPS / memory measurement for Mamba3Yolo.

Supports:
- Pure PyTorch (CUDA / CPU)
- ONNX Runtime
- TensorRT (if available)
- Jetson notes (use tegrastats + trtexec)

Usage on Kaggle / local:
    python scripts/measure_latency.py --weights runs/.../best.pt --imgsz 640 --device cuda
    python scripts/measure_latency.py --onnx mamba3yolo.onnx --backend onnx

For Jetson:
    # First export TensorRT engine
    trtexec --onnx=mamba3yolo.onnx --saveEngine=mamba3yolo.trt --fp16 --workspace=4096
    # Then measure with this script or nsys / tegrastats
"""

from __future__ import annotations

import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np


def measure_pytorch(
    model: torch.nn.Module,
    imgsz: int = 640,
    batch: int = 1,
    warmup: int = 50,
    iters: int = 200,
    device: str = "cuda",
    amp: bool = True,
) -> Dict[str, float]:
    model = model.to(device).eval()
    x = torch.randn(batch, 3, imgsz, imgsz, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    fps = 1000.0 / times.mean() * batch

    mem = 0.0
    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

    return {
        "backend": "pytorch",
        "device": device,
        "amp": amp,
        "batch": batch,
        "imgsz": imgsz,
        "latency_mean_ms": float(times.mean()),
        "latency_std_ms": float(times.std()),
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_p99_ms": float(np.percentile(times, 99)),
        "fps": float(fps),
        "peak_mem_mb": float(mem),
    }


def measure_onnx(
    onnx_path: str,
    imgsz: int = 640,
    batch: int = 1,
    warmup: int = 30,
    iters: int = 100,
    providers: List[str] = None,
) -> Dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed. pip install onnxruntime-gpu"}

    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    x = np.random.randn(batch, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(warmup):
        _ = sess.run(None, {input_name: x})

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: x})
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000
    fps = 1000.0 / times.mean() * batch
    return {
        "backend": "onnxruntime",
        "providers": sess.get_providers(),
        "batch": batch,
        "imgsz": imgsz,
        "latency_mean_ms": float(times.mean()),
        "latency_std_ms": float(times.std()),
        "latency_p50_ms": float(np.percentile(times, 50)),
        "fps": float(fps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="PyTorch .pt checkpoint")
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--out", type=str, default="latency_results.json")
    args = parser.parse_args()

    results = []

    if args.weights:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.models.mamba3yolo import build_mamba3yolo

        ckpt = torch.load(args.weights, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        model = build_mamba3yolo(
            cfg.get("scale", "T"),
            nc=cfg.get("nc", 80),
            is_mimo=cfg.get("is_mimo", True),
        )
        model.load_state_dict(ckpt["model"])
        print(f"Loaded {args.weights}")
        r = measure_pytorch(
            model, args.imgsz, args.batch, args.warmup, args.iters, args.device, args.amp
        )
        results.append(r)
        print(json.dumps(r, indent=2))

    if args.onnx:
        r = measure_onnx(args.onnx, args.imgsz, args.batch, args.warmup, args.iters)
        results.append(r)
        print(json.dumps(r, indent=2))

    if not results:
        # Demo with random model
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.models.mamba3yolo import build_mamba3yolo
        model = build_mamba3yolo("T")
        r = measure_pytorch(model, args.imgsz, args.batch, 20, 50, args.device, args.amp)
        results.append(r)
        print(json.dumps(r, indent=2))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")

    print("\n" + "="*60)
    print("Jetson / TensorRT notes:")
    print("  1. Export: torch.onnx.export(...) or the quant script")
    print("  2. Build engine: trtexec --onnx=model.onnx --saveEngine=model.trt --fp16")
    print("  3. Measure: trtexec --loadEngine=model.trt --shapes=images:1x3x640x640")
    print("  4. Power: tegrastats --interval 100")
    print("="*60)


if __name__ == "__main__":
    main()
