#!/usr/bin/env python3
"""
Register Mamba3ODSS with Ultralytics so a YOLO yaml can use it, and generate a
yolo11-mamba3 config (C3k2 mixers -> Mamba3ODSS). This makes the Mamba-3 block
survive Ultralytics' yaml rebuild during .train() -- the surgery-on-object route
does NOT (train() re-parses from cfg).

    from scripts.ultra_mamba3 import register, make_yaml
    register()
    from ultralytics import YOLO
    model = YOLO(make_yaml("s"))          # yolo11s with Mamba-3 mixers
    model.train(data="coco.yaml", ...)    # real DFL loss + assigner + NMS + mAP

No site-packages files are modified: parse_model is re-bound in-process.

ATTRIBUTION: the ODSS / LSBlock / RGBlock design this plugs into is Mamba-YOLO's
(Wang et al., AAAI 2025, arXiv:2406.05835, https://github.com/HZAI-ZJNU/Mamba-YOLO,
AGPL-3.0). The C3k2 -> SSM-mixer swap on a YOLO11 yaml is also not new: see
Xray-YOLO-Mamba (Zhao et al., Sci Rep 15:13171, 2025), which replaces C3k2 with a
VSS/SS2D block in the YOLOv11-n backbone. What is new here is the SSM *inside* the
block (Mamba-3: complex/rotational state + exponential-trapezoidal discretization)
and the intrinsic Gramian saliency. Say exactly that and nothing more.

`where="backbone"` exists because no prior work swaps the neck: Xray-YOLO-Mamba
keeps the YOLOv11 neck, Mamba-YOLO tunes a stage ratio. Swapping all 8 mixers is
what takes this model to ~69 GFLOPs vs Mamba-YOLO-T's 13.2G. Measure both.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def register(verbose: bool = True) -> None:
    """Patch ultralytics.nn.tasks.parse_model so it treats Mamba3ODSS as a
    channel-changing, repeatable CSP block (like C3k2)."""
    import ultralytics.nn.tasks as T
    from src.blocks.mamba3_odss import Mamba3ODSS
    if getattr(T, "_mamba3_registered", False):
        return

    src = inspect.getsource(T.parse_model)
    a_rep = "args.insert(2, n)"           # the repeat-insert set sits just before this
    a_chan = "c1, c2 = ch[f], args[0]"    # the channel set sits just before this
    if a_rep not in src or a_chan not in src:
        raise RuntimeError("parse_model anchors not found; ultralytics layout changed "
                           "(pinned 8.3.0). Update scripts/ultra_mamba3.py.")

    def inject(text: str, anchor: str) -> str:
        i = text.index(anchor)
        brace = text.rindex("}", 0, i)    # closing brace of the set literal before anchor
        return text[:brace] + "    Mamba3ODSS,\n            " + text[brace:]

    src = inject(src, a_rep)
    src = inject(src, a_chan)

    T.Mamba3ODSS = Mamba3ODSS                            # visible to the set literal at call time
    exec(compile(src, T.__file__, "exec"), T.__dict__)   # re-bind parse_model in-place
    T._mamba3_registered = True
    if verbose:
        print("registered Mamba3ODSS with ultralytics parse_model")


def make_yaml(scale: str = "s", d_state: int = 32, expand: int = 1,
              mlp_ratio: float = 1.0, use_rope: bool = True, trapezoidal: bool = True,
              where: str = "all", tag: str | None = None, out: str | None = None) -> str:
    """Generate a yolo11-mamba3 yaml from the stock yolo11 config, swapping C3k2 mixers
    for Mamba3ODSS. Ablation flags are baked into each block's args:
    [c2, d_state, expand, mlp_ratio, use_rope, trapezoidal].

    where: "all"      -> swap backbone + neck (8 mixers; ~69 GFLOPs at -s)
           "backbone" -> swap backbone only, keep the stock YOLO11 neck (what every
                         prior Mamba-YOLO variant does). Cheaper; ablate against "all".
    Returns the path."""
    import ultralytics
    import yaml
    assert where in {"all", "backbone"}, "where must be 'all' or 'backbone'"
    stock = Path(ultralytics.__file__).parent / "cfg/models/11/yolo11.yaml"
    d = yaml.safe_load(stock.read_text())
    extra = [int(d_state), int(expand), float(mlp_ratio), bool(use_rope), bool(trapezoidal)]

    def swap(layers):
        for layer in layers:                       # layer = [from, number, module, args]
            if layer[2] == "C3k2":
                layer[2] = "Mamba3ODSS"
                layer[3] = [layer[3][0], *extra]   # out-channels + ablation args
        return layers

    d["backbone"] = swap(d["backbone"])
    if where == "all":
        d["head"] = swap(d["head"])

    if tag is None:
        tag = ""
        if where == "backbone":
            tag += "-bb"
        if not use_rope:
            tag += "-norope"
        if not trapezoidal:
            tag += "-euler"
    out_dir = Path(__file__).resolve().parents[1] / "configs/models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out or str(out_dir / f"yolo11{scale}-mamba3{tag}.yaml")
    Path(out).write_text(yaml.safe_dump(d, sort_keys=False))
    return out


if __name__ == "__main__":
    import torch
    from ultralytics import YOLO
    from src.blocks.mamba3_ref import Mamba3RefSSM
    register()
    for where in ("all", "backbone"):
        path = make_yaml("s", where=where)
        m = YOLO(path)
        n = sum(isinstance(mm, Mamba3RefSSM) for mm in m.model.modules())
        p = sum(pp.numel() for pp in m.model.parameters()) / 1e6
        print(f"{Path(path).name}: Mamba3RefSSM blocks={n}, params={p:.2f}M")
        with torch.no_grad():
            out = m.model(torch.randn(1, 3, 320, 320))
        print("  forward ok" if out is not None else "  forward FAILED")
