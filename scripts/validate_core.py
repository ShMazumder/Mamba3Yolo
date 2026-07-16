#!/usr/bin/env python3
"""
Correctness + stability gates for Mamba3RefSSM. Run this BEFORE spending GPU hours.

Three gates:
  1. EQUIVALENCE  - the chunked SSD scan must equal the O(L) reference recurrence,
                    for trapezoidal on/off and RoPE on/off. If this fails, every
                    number downstream is meaningless.
  2. STABILITY    - forward + backward under AMP autocast must stay finite at the
                    decay/step sizes training actually reaches. This is the gate that
                    the old `Pin.clamp(min=1e-6)` + `U / Pin` scan failed: alpha**chunk
                    underflows, the clamp bites, and U/1e-6 overflows fp16 -> NaN on
                    every batch (the frozen, all-NaN detector runs in the log).
  3. SHAPES       - token_saliency returns (B, L) for odd/even/long sequences. The old
                    F.pad(beta, (0,0,0,0,0,1)) padded the batch dim of a 3-D tensor and
                    returned L-1 -> the 16383-vs-16384 crash in the XAI cell.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.blocks.mamba3_ref import Mamba3RefSSM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OK = True


def _fail(msg):
    global OK
    OK = False
    print("  FAIL:", msg)


def gate_equivalence():
    print("[1] chunked SSD scan  vs  O(L) reference recurrence")
    torch.manual_seed(0)
    for trap in (True, False):
        for rope in (True, False):
            m = Mamba3RefSSM(d_model=32, d_state=16, expand=2, headdim=16,
                             use_rope=rope, trapezoidal=trap, chunk=8).to(DEV).eval()
            # push the decay well past the fp16-underflow regime that used to NaN
            with torch.no_grad():
                m.A_log.fill_(1.5)                      # A = -exp(1.5) ~ -4.5
                m.dt_proj.bias.fill_(0.5)
            x = torch.randn(3, 37, 32, device=DEV)      # L not a multiple of chunk
            with torch.no_grad():
                uf = m.norm(m.in_proj(x).chunk(2, -1)[0]).float()
                la, ga, be = m._coeffs(uf)
                Br, Cr = m._rotate(uf, m.B_proj(uf), m.C_proj(uf))
                uh = uf.view(3, 37, m.nheads, m.headdim)
                y_fast = m._ssd(la, ga, be, uh, Br, Cr)
                y_ref = m._ssm_reference(la, ga, be, uh, Br, Cr)
            err = (y_fast - y_ref).abs().max().item()
            scale = y_ref.abs().max().item() + 1e-12
            tag = f"trapezoidal={trap} rope={rope}"
            print(f"  {tag:34s} max|fast-ref| = {err:.3e}  (rel {err/scale:.2e})")
            if not (err / scale < 1e-4):
                _fail(f"{tag}: chunked scan does not match the reference recurrence")


def gate_stability():
    print("[2] AMP forward/backward stays finite at training-scale decays")
    torch.manual_seed(0)
    amp = DEV == "cuda"
    for A_log, dt_bias, L in [(0.0, 0.0, 512), (2.0, 1.0, 512), (4.0, 3.0, 1024)]:
        m = Mamba3RefSSM(d_model=64, d_state=32, expand=1, headdim=64).to(DEV)
        with torch.no_grad():
            m.A_log.fill_(A_log)                        # alpha = exp(-dt*exp(A_log))
            m.dt_proj.bias.fill_(dt_bias)
            m.w_proj.weight.mul_(50.0)                  # try to blow up the rotary phase
        x = torch.randn(2, L, 64, device=DEV, requires_grad=True)
        with torch.autocast(device_type=DEV, dtype=torch.float16, enabled=amp):
            y = m(x)
        loss = y.float().pow(2).mean()
        loss.backward()
        gmax = max(p.grad.abs().max().item() for p in m.parameters() if p.grad is not None)
        finite = torch.isfinite(y).all().item() and torch.isfinite(loss).item()
        gfinite = all(torch.isfinite(p.grad).all().item()
                      for p in m.parameters() if p.grad is not None)
        alpha = float(torch.exp(-torch.nn.functional.softplus(torch.tensor(dt_bias))
                                * torch.exp(torch.tensor(A_log))))
        print(f"  A_log={A_log:<4} dt_bias={dt_bias:<4} L={L:<5} alpha~{alpha:.2e}  "
              f"y_finite={finite} grad_finite={gfinite} max|grad|={gmax:.2e}")
        if not (finite and gfinite):
            _fail(f"non-finite forward/backward at A_log={A_log}, dt_bias={dt_bias}")


def gate_shapes():
    print("[3] token_saliency shapes")
    m = Mamba3RefSSM(d_model=32, d_state=16, expand=1, headdim=32).to(DEV).eval()
    for L in (1, 7, 64, 4096):
        s = m.token_saliency(torch.randn(2, L, 32, device=DEV))
        ok = tuple(s.shape) == (2, L) and torch.isfinite(s).all().item() and (s >= 0).all().item()
        print(f"  L={L:<6} -> {tuple(s.shape)}  finite&nonneg={ok}")
        if not ok:
            _fail(f"token_saliency wrong at L={L}: got {tuple(s.shape)}")


if __name__ == "__main__":
    print(f"device={DEV}\n")
    gate_equivalence()
    print()
    gate_stability()
    print()
    gate_shapes()
    print()
    print("  PASS: core is correct and AMP-stable." if OK else
          "  FAIL: do not train until the gates above are green.")
    sys.exit(0 if OK else 1)
