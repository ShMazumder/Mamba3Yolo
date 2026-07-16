#!/usr/bin/env python3
"""
Honesty gate for Mamba3RefSSM: the parity / state-tracking test.

Mamba-3 claims complex-valued (rotational) states solve running parity
  y_t = (sum_{i<=t} x_i) mod 2
-- a task real-valued diagonal SSMs provably CANNOT (Grazzi et al., Thm 1: real
eigenvalues can't do rotation).

Decisive result is the CONTRAST, not the absolute:
  - use_rope=True  (complex/rotational)  -> should learn parity  (acc -> ~1.0)
  - use_rope=False (real diagonal SSM)   -> should FAIL           (acc ~ 0.5)
If both pass, the rotation isn't doing the work (wiring bug). If the True model
fails, there is no Mamba-3 state-tracking claim. Either way we learn the truth.

Note on the rotation rate: it is now bounded to rot_max * tanh(w) with
rot_max = 2*pi, so pi rad/step (what parity needs) sits at tanh(w) = 0.5 and is
comfortably reachable, while the cumulative phase over a 16k-token image scan can no
longer run away. If you lower rot_max below pi this gate MUST fail -- that is the
point of the bound being an explicit, documented hyperparameter.

CPU-friendly: small model, short sequences, a few hundred steps.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.blocks.mamba3_ref import Mamba3RefSSM


class ParityNet(nn.Module):
    def __init__(self, use_rope: bool, d_model=32, d_state=16, headdim=16):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.ssm = Mamba3RefSSM(d_model, d_state=d_state, headdim=headdim, use_rope=use_rope)
        self.head = nn.Linear(d_model, 1)

    def forward(self, bits):                 # bits: (B, L) in {0,1}
        h = self.embed(bits.unsqueeze(-1))   # (B, L, d_model)
        h = self.ssm(h)
        return self.head(h).squeeze(-1)      # (B, L) per-position logit


def batch(B, L, device):
    bits = torch.randint(0, 2, (B, L), device=device).float()
    target = torch.cumsum(bits, dim=1) % 2   # running parity, per position
    return bits, target


def run(use_rope, steps=800, B=128, L=24, seed=0, device="cpu"):
    torch.manual_seed(seed)
    net = ParityNet(use_rope).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        bits, tgt = batch(B, L, device)
        opt.zero_grad()
        loss = lossf(net(bits), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    net.eval()
    with torch.no_grad():
        accs = {}
        for evalL in (L, 2 * L):             # same length + longer (length generalization)
            bits, tgt = batch(512, evalL, device)
            accs[evalL] = ((net(bits) > 0).float() == tgt).float().mean().item()
    return accs


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}\nrunning parity (cumsum mod 2), per-position accuracy\n")
    on = run(use_rope=True, device=dev)
    off = run(use_rope=False, device=dev)
    L = 24
    print(f"  RoPE ON  (complex): L={L} acc={on[L]:.3f}   L={2*L} acc={on[2*L]:.3f}")
    print(f"  RoPE OFF (real)   : L={L} acc={off[L]:.3f}   L={2*L} acc={off[2*L]:.3f}\n")
    gap = on[L] - off[L]
    if on[L] > 0.9 and off[L] < 0.75:
        print(f"  PASS: complex solves parity, real fails (gap {gap:+.3f}). Mamba-3 state-tracking is REAL.")
    elif on[L] > 0.9 and off[L] > 0.9:
        print(f"  SUSPECT: both solve it (gap {gap:+.3f}). Rotation may not be load-bearing -- inspect wiring.")
    elif on[L] < 0.75:
        print(f"  FAIL: complex model cannot track parity (acc {on[L]:.3f}). Claim #2 (Mamba-3 core) is NOT supported.")
    else:
        print(f"  WEAK: gap {gap:+.3f}. Inconclusive -- tune steps/lr and rerun before trusting.")
