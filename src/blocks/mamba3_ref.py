"""
Faithful pure-PyTorch reference for the Mamba-3 selective SSM.

Implements the actual Mamba-3 recurrence (Lahoti et al., 2026, arXiv:2603.15569),
NOT a gated-MLP stand-in. Sequential O(L) scan: correct and honest, not fast.
Use it as the reference path when the official CUDA kernel is unavailable, and to
run the paper's ablations (trapezoidal on/off via lambda, RoPE on/off).

Recurrence (Eq. 5-6, 11):
    alpha_t = exp(dt_t * A)                         # decay, A < 0
    beta_t  = (1 - lambda_t) * dt_t * alpha_t       # previous-input (trapezoidal) weight
    gamma_t = lambda_t * dt_t                        # current-input weight
    h_t = alpha_t * h_{t-1}
          + beta_t  * outer(x_{t-1}, RoPE(B_{t-1}, Phi_{t-1}))
          + gamma_t * outer(x_t,     RoPE(B_t,     Phi_t))
    y_t = < h_t , RoPE(C_t, Phi_t) >                # sum over state dim
where Phi_t is the cumulative, dt-scaled rotary angle (the product of R_i in Eq 11,
realised via the RoPE trick: rotate B at write-time, C at read-time).

lambda_t = 1  ->  exponential-Euler  ==  Mamba-2 (single input term). This is the
knob for the trapezoidal ablation. SISO here; MIMO (rank-r B/C) is a documented
extension below, deliberately not enabled so we never over-claim what runs.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _rope(v: Tensor, phi: Tensor) -> Tensor:
    """Rotate the state dimension of v in 2-D blocks by angles phi.

    v:   (..., N)      N even; consecutive pairs form the 2-D rotation blocks.
    phi: (..., N // 2) rotation angle per block.
    Returns v rotated, same shape as v. This is the standard RoPE trick, i.e. a
    real realisation of the complex-valued state transition (Prop. Complex-to-Real).
    """
    v2 = v.unflatten(-1, (-1, 2))          # (..., N/2, 2)
    cos, sin = torch.cos(phi), torch.sin(phi)
    x0, x1 = v2[..., 0], v2[..., 1]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    return torch.stack((r0, r1), dim=-1).flatten(-2)   # (..., N)


class Mamba3RefSSM(nn.Module):
    """Selective SSM with exponential-trapezoidal discretization + complex (RoPE) states.

    Diagonal (SISO) state per channel. Input/output are (B, L, D).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_state % 2 == 0, "d_state must be even for 2-D rotary blocks"
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.headdim = min(headdim, self.d_inner)
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        self.d_state = d_state

        # input / gate / output projections
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)   # x, z(gate)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # data-dependent params: dt (per head), B/C (shared d_state), lambda (per head)
        self.dt_proj = nn.Linear(self.d_inner, self.nheads, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=True)   # explicit B bias term
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=True)   # explicit C bias term
        self.lam_proj = nn.Linear(self.d_inner, self.nheads, bias=True)

        # A < 0 via -exp(A_log); one decay rate per (head, state)
        self.A_log = nn.Parameter(torch.zeros(self.nheads, d_state))
        self.norm = nn.LayerNorm(self.d_inner)

        # fixed rotary base frequencies per 2-D block (data-independent part of the angle)
        theta = rope_base ** (-torch.arange(0, d_state, 2).float() / d_state)  # (N/2,)
        self.register_buffer("theta", theta, persistent=False)

        # init dt bias so softplus(dt) lands in [dt_min, dt_max]
        with torch.no_grad():
            dt = torch.empty(self.nheads).uniform_(math.log(dt_min), math.log(dt_max)).exp()
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def forward(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        H, P, N = self.nheads, self.headdim, self.d_state

        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)                     # (B, L, d_inner) each
        u = self.norm(u)

        dt = F.softplus(self.dt_proj(u))               # (B, L, H)  > 0
        lam = torch.sigmoid(self.lam_proj(u))          # (B, L, H)  in [0,1]
        Bt = self.B_proj(u)                            # (B, L, N)
        Ct = self.C_proj(u)                            # (B, L, N)
        A = -torch.exp(self.A_log)                     # (H, N)  < 0

        uh = u.view(B, L, H, P)                        # per-head channels

        # cumulative rotary angle Phi_t = sum_{i<=t} dt_i * theta   (per head, per block)
        # dt: (B,L,H) ; theta: (N/2,)  ->  (B,L,H,N/2)
        ang = dt.unsqueeze(-1) * self.theta            # (B, L, H, N/2)
        phi = torch.cumsum(ang, dim=1)                 # (B, L, H, N/2)

        h = x.new_zeros(B, H, P, N)                    # SSM state
        prev_input = x.new_zeros(B, H, P)              # x_{t-1}
        prev_Brot = x.new_zeros(B, H, N)               # RoPE(B_{t-1}, Phi_{t-1})
        ys = []
        for t in range(L):
            dt_t = dt[:, t]                            # (B, H)
            lam_t = lam[:, t]                          # (B, H)
            alpha = torch.exp(dt_t.unsqueeze(-1) * A)  # (B, H, N)
            beta = ((1 - lam_t) * dt_t).unsqueeze(-1) * alpha   # (B, H, N)
            gamma = (lam_t * dt_t).unsqueeze(-1)                # (B, H, 1)

            phi_t = phi[:, t]                          # (B, H, N/2)
            Brot = _rope(Bt[:, t].unsqueeze(1).expand(B, H, N), phi_t)   # (B, H, N)
            Crot = _rope(Ct[:, t].unsqueeze(1).expand(B, H, N), phi_t)   # (B, H, N)

            xt = uh[:, t]                              # (B, H, P)
            cur = gamma.unsqueeze(2) * (xt.unsqueeze(-1) * Brot.unsqueeze(2))   # (B,H,P,N)
            prv = beta.unsqueeze(2) * (prev_input.unsqueeze(-1) * prev_Brot.unsqueeze(2))
            h = alpha.unsqueeze(2) * h + prv + cur     # (B, H, P, N)

            y_t = (h * Crot.unsqueeze(2)).sum(-1)      # (B, H, P)
            ys.append(y_t)
            prev_input, prev_Brot = xt, Brot

        y = torch.stack(ys, dim=1).reshape(B, L, self.d_inner)   # (B, L, d_inner)
        y = y * F.silu(z)                              # gate
        return self.out_proj(y)

# MIMO extension (not enabled): make B_proj / C_proj emit (N * r) and reshape to
# (N, r) matrices, replace the outer product `xt (x) Brot` with a matmul over the
# rank-r channel group, and read out y_t = Crot^T h. Enable only once validated
# against the paper's state-size vs. perplexity ablation -- until then, SISO is
# what runs, and that is what any paper text should claim.
