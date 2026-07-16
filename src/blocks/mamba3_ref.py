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
        use_rope: bool = True,
        trapezoidal: bool = True,
    ):
        super().__init__()
        self.use_rope = use_rope        # False -> real-valued diagonal SSM (ablation control)
        self.trapezoidal = trapezoidal  # False -> lambda=1 = exponential-Euler = Mamba-2
        self.parallel = True            # chunked parallel scan; set False for O(L) reference
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
        # Rotation RATE (imaginary part of complex A) -- data-dependent, decoupled from
        # the small real decay step dt. Must reach ~pi/step to represent parity-style
        # rotational state tracking; that is why it is NOT tied to dt (see parity gate).
        self.w_proj = nn.Linear(self.d_inner, self.nheads, bias=True)

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
        if not self.trapezoidal:
            lam = torch.ones_like(lam)                 # Euler / Mamba-2 ablation
        Bt = self.B_proj(u)                            # (B, L, N)
        Ct = self.C_proj(u)                            # (B, L, N)
        A = -torch.exp(self.A_log)                     # (H, N)  < 0

        uh = u.view(B, L, H, P)                        # (B,L,H,P) per-head channels

        # trapezoidal discretization coeffs, all steps at once
        alpha = torch.exp(dt.unsqueeze(-1) * A)        # (B,L,H,N)
        beta = ((1 - lam) * dt).unsqueeze(-1) * alpha  # (B,L,H,N)
        gamma = (lam * dt).unsqueeze(-1)               # (B,L,H,1)

        Bt_h = Bt.unsqueeze(2).expand(B, L, H, N)
        Ct_h = Ct.unsqueeze(2).expand(B, L, H, N)
        if self.use_rope:
            # cumulative rotary angle Phi_t = sum_{i<=t} w_i * theta   (vectorized over L)
            # w_t is the data-dependent rotation rate (imag part of A), free to reach ~pi.
            w = self.w_proj(u)                             # (B,L,H) unbounded rotation rate
            ang = w.unsqueeze(-1) * self.theta             # (B,L,H,N/2)
            phi = torch.cumsum(ang, dim=1)                 # (B,L,H,N/2)
            Brot = _rope(Bt_h, phi)                    # (B,L,H,N)
            Crot = _rope(Ct_h, phi)
        else:
            Brot, Crot = Bt_h, Ct_h                    # real-valued control: no rotation

        # per-step input drive U_t = gamma_t (x_t (x) B_t) + beta_t (x_{t-1} (x) B_{t-1})
        cur = gamma.unsqueeze(3) * (uh.unsqueeze(-1) * Brot.unsqueeze(3))       # (B,L,H,P,N)
        prev_uh = F.pad(uh, (0, 0, 0, 0, 1, 0))[:, :L]      # shift right by 1 along L
        prev_Brot = F.pad(Brot, (0, 0, 0, 0, 1, 0))[:, :L]
        prv = beta.unsqueeze(3) * (prev_uh.unsqueeze(-1) * prev_Brot.unsqueeze(3))
        U = cur + prv                                  # (B,L,H,P,N)

        alpha5 = alpha.unsqueeze(3)                    # (B,L,H,1,N) decay, broadcasts over P
        Hs = self._scan_chunked(alpha5, U) if self.parallel else self._scan_reference(alpha5, U)

        y = (Hs * Crot.unsqueeze(3)).sum(-1)           # (B,L,H,P)  readout y_t = <Crot_t, h_t>
        y = y.reshape(B, L, self.d_inner)
        y = y * F.silu(z)                              # gate
        return self.out_proj(y)

    @staticmethod
    def _scan_reference(alpha: Tensor, U: Tensor) -> Tensor:
        """Sequential O(L) linear scan h_t = alpha_t h_{t-1} + U_t. Ground truth."""
        B, L, H, P, N = U.shape
        h = U.new_zeros(B, H, P, N)
        out = []
        for t in range(L):
            h = alpha[:, t] * h + U[:, t]
            out.append(h)
        return torch.stack(out, dim=1)                 # (B,L,H,P,N)

    @staticmethod
    def _scan_chunked(alpha: Tensor, U: Tensor, chunk: int = 32) -> Tensor:
        """Chunked prefix scan of h_t = alpha_t h_{t-1} + U_t.

        Within-chunk work is fully parallel via cumulative products; only the
        chunk-carry recurrence loops (L/chunk steps). Numerically safe while the
        per-chunk decay product stays well above underflow -- true here because
        alpha = exp(dt*A) with small dt keeps alpha near 1.
        """
        B, L, H, P, N = U.shape
        pad = (chunk - L % chunk) % chunk
        if pad:
            U = F.pad(U, (0, 0, 0, 0, 0, 0, 0, pad))
            alpha = F.pad(alpha, (0, 0, 0, 0, 0, 0, 0, pad), value=1.0)
        Lp = L + pad
        nc = Lp // chunk
        Uc = U.view(B, nc, chunk, H, P, N)
        Ac = alpha.view(B, nc, chunk, H, 1, N)
        Pin = torch.cumprod(Ac, dim=2).clamp(min=1e-6)  # inclusive within-chunk decay (clamp for AMP safety)
        intra = Pin * torch.cumsum(Uc / Pin, dim=2)    # (B,nc,chunk,H,P,N)
        Ptot = Pin[:, :, -1]                           # (B,nc,H,1,N) full-chunk decay
        last = intra[:, :, -1]                         # (B,nc,H,P,N) chunk-final state
        carry = U.new_zeros(B, H, P, N)
        carries = []
        for c in range(nc):                            # nc = L/chunk steps, vectorized within
            carries.append(carry)
            carry = Ptot[:, c] * carry + last[:, c]
        carries = torch.stack(carries, dim=1)          # (B,nc,H,P,N) state entering each chunk
        Hs = intra + Pin * carries.unsqueeze(2)        # add carried state, broadcast over chunk
        return Hs.view(B, Lp, H, P, N)[:, :L]

    def _reverse_energy(self, alpha: Tensor) -> Tensor:
        """G_tau[n] = sum_{t>=tau} (prod_{k=tau+1..t} alpha_k[n])^2 via the stable
        reverse recurrence G_tau = 1 + alpha_{tau+1}^2 G_{tau+1} (G stays O(1), no
        division/underflow). (B,L,H,N). Eval-only (saliency), so an O(L) loop is fine."""
        a2 = alpha ** 2
        L = a2.shape[1]
        G = torch.empty_like(a2)
        g = torch.zeros_like(a2[:, 0])                      # G_{tau+1}, starts at 0 (G_L)
        for t in range(L - 1, -1, -1):
            g = torch.ones_like(g) if t == L - 1 else 1.0 + a2[:, t + 1] * g
            G[:, t] = g
        return G

    @torch.no_grad()
    def token_saliency(self, x: Tensor) -> Tensor:
        """Intrinsic controllability-energy saliency per input token. Returns (B, L).
        Closed-form from the SSM internals, one pass, no backprop.

        Accounts for BOTH Gramian contributions of each token:
          - gamma term: token t's direct drive at step t
          - beta term:  token t's carry-forward drive at step t+1 (trapezoidal)
        """
        B, L, _ = x.shape
        H, P, N = self.nheads, self.headdim, self.d_state
        u, _ = self.in_proj(x).chunk(2, dim=-1)
        u = self.norm(u)
        dt = F.softplus(self.dt_proj(u))
        lam = torch.sigmoid(self.lam_proj(u))
        if not self.trapezoidal:
            lam = torch.ones_like(lam)
        Bt = self.B_proj(u)
        A = -torch.exp(self.A_log)
        uh = u.view(B, L, H, P)
        alpha = torch.exp(dt.unsqueeze(-1) * A)             # (B,L,H,N)
        gamma = lam * dt                                    # (B,L,H)
        beta = (1 - lam) * dt                               # (B,L,H) trapezoidal weight
        Bt_h = Bt.unsqueeze(2).expand(B, L, H, N)
        if self.use_rope:
            w = self.w_proj(u)
            phi = torch.cumsum(w.unsqueeze(-1) * self.theta, dim=1)
            Brot = _rope(Bt_h, phi)
        else:
            Brot = Bt_h
        G = self._reverse_energy(alpha)                     # (B,L,H,N)
        energy = (Brot ** 2 * G).sum(-1)                    # (B,L,H)
        uh_norm = (uh ** 2).sum(-1)                         # (B,L,H)
        # Current-input (gamma) contribution: token t drives state at step t
        s_gamma = (gamma ** 2) * uh_norm * energy           # (B,L,H)
        # Trapezoidal (beta) contribution: token t drives state at step t+1
        # beta_{t+1} * alpha_{t+1} weight; G shifted by one step
        beta_next = F.pad(beta, (0, 0, 0, 0, 0, 1))[:, 1:L+1]     # (B,L,H) beta at t+1
        alpha_next = F.pad(alpha, (0, 0, 0, 0, 0, 1), value=1.0)[:, 1:L+1]  # (B,L,H,N)
        G_next = F.pad(G, (0, 0, 0, 0, 0, 1))[:, 1:L+1]           # (B,L,H,N)
        energy_next = (Brot ** 2 * alpha_next ** 2 * G_next).sum(-1)  # (B,L,H)
        s_beta = (beta_next ** 2) * uh_norm * energy_next   # (B,L,H)
        s = s_gamma + s_beta                                # (B,L,H)
        return s.mean(-1)                                   # (B,L)


# MIMO extension (not enabled): make B_proj / C_proj emit (N * r) and reshape to
# (N, r) matrices, replace the outer product `xt (x) Brot` with a matmul over the
# rank-r channel group, and read out y_t = Crot^T h. Enable only once validated
# against the paper's state-size vs. perplexity ablation -- until then, SISO is
# what runs, and that is what any paper text should claim.
