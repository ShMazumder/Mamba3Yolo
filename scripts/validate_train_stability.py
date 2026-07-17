#!/usr/bin/env python3
"""
Training-scale stability gate. Reproduces the epoch-2 NaN that validate_core misses.

WHY validate_core is green but training NaNs
--------------------------------------------
validate_core.gate_stability tests the BARE Mamba3RefSSM:
  - max L = 1024                (training hits L = H*W = 64*64 = 4096 at 256px, stride-4 block)
  - one forward+backward         (no optimizer STEP, so it never sees weights move)
  - no gradient checkpointing     (real block: checkpoint(ss2d, use_reentrant=False))
  - no 4-directional scan         (real block runs the SSM 4x with flips)
token_saliency IS tested at L=4096 but under @torch.no_grad -> forward only.

The gap is exactly the regime training runs in. This gate closes it: it drives the
REAL Mamba3SS2D block (4-dir + checkpoint) at L=4096 under AMP, takes TWO optimizer
steps, and asserts weights/grads stay finite across the step -- because the run log
shows epoch 1 finite, epoch 2 NaN, i.e. the first AdamW step is what detonates.

It also bisects: for each config it toggles checkpoint / AMP / trapezoidal / rope so
the FIRST failing combination names the culprit instead of just "it's NaN".

Run:  python scripts/validate_train_stability.py
Exit 0 = every config survived 2 steps finite. Exit 1 = at least one detonated
(the printed table says which toggle first breaks it).
"""
import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.blocks.mamba3_odss import Mamba3SS2D

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OK = True


def _fail(msg):
    global OK
    OK = False
    print("  FAIL:", msg)


def _finite_params(m):
    return all(torch.isfinite(p).all().item() for p in m.parameters())


def _finite_grads(m):
    return all(torch.isfinite(p.grad).all().item()
               for p in m.parameters() if p.grad is not None)


def run_config(dim, hw, checkpoint, amp, trapezoidal, rope, steps=2, seed=0):
    """Drive the real 4-dir block at feature-map size hw x hw for `steps` optimizer
    steps. Returns (survived, first_bad_step, detail)."""
    torch.manual_seed(seed)
    block = Mamba3SS2D(dim=dim, d_state=32, expand=1, headdim=min(64, dim),
                       use_rope=rope, trapezoidal=trapezoidal).to(DEV)
    # Mirror the real ODSS block: recompute the scan in backward.
    import torch.utils.checkpoint as _ckpt
    opt = torch.optim.AdamW(block.parameters(), lr=2e-3)
    scaler = torch.amp.GradScaler(DEV, enabled=amp and DEV == "cuda")

    for step in range(steps):
        block.train()
        x = torch.randn(2, dim, hw, hw, device=DEV, requires_grad=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEV, dtype=torch.float16,
                            enabled=amp and DEV == "cuda"):
            if checkpoint and DEV == "cuda":
                y = _ckpt.checkpoint(block, x, use_reentrant=False)
            else:
                y = block(x)
            loss = y.float().pow(2).mean()
        if not torch.isfinite(loss).item():
            return False, step, f"loss non-finite BEFORE step (fwd), L={hw*hw}"
        scaler.scale(loss).backward()
        # unscale so grad-finiteness is meaningful even under AMP
        scaler.unscale_(opt)
        gfin = _finite_grads(block)
        gmax = max((p.grad.abs().max().item()
                    for p in block.parameters() if p.grad is not None), default=0.0)
        scaler.step(opt)
        scaler.update()
        pfin = _finite_params(block)
        if not (gfin and pfin):
            return False, step, f"grad_finite={gfin} param_finite={pfin} max|grad|={gmax:.2e}"
    return True, None, "ok"


def gate():
    # L = hw*hw. 64 -> 4096 tokens (the stride-4 block at 256px). 96 -> 9216 (192px-ish).
    sizes = [(64, 64), (128, 96)]      # (channels, feature-map side)
    print(f"[train-stability] device={DEV}  (real Mamba3SS2D, 2 optimizer steps)\n")
    header = f"  {'dim':>4} {'L':>6} {'ckpt':>5} {'amp':>4} {'trap':>5} {'rope':>5}  result"
    print(header)
    for dim, hw in sizes:
        for checkpoint, amp, trap, rope in itertools.product(
                (False, True), (False, True), (True, False), (True, False)):
            survived, bad_step, detail = run_config(dim, hw, checkpoint, amp, trap, rope)
            tag = "OK" if survived else f"NaN@step{bad_step} ({detail})"
            print(f"  {dim:>4} {hw*hw:>6} {str(checkpoint):>5} {str(amp):>4} "
                  f"{str(trap):>5} {str(rope):>5}  {tag}")
            if not survived:
                _fail(f"dim={dim} L={hw*hw} ckpt={checkpoint} amp={amp} "
                      f"trap={trap} rope={rope}: {detail}")


if __name__ == "__main__":
    gate()
    print()
    print("  PASS: real block survives 2 optimizer steps at training scale." if OK else
          "  FAIL: reproduced the training NaN. First failing row names the culprit.")
    sys.exit(0 if OK else 1)
