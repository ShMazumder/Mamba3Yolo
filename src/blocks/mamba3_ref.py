"""
Faithful pure-PyTorch reference for the Mamba-3 selective SSM.

Implements the actual Mamba-3 recurrence (complex/rotational state + exponential-
trapezoidal discretization), NOT a gated-MLP stand-in.

Recurrence (SISO, per head h, per channel p):
    alpha_t = exp(dt_t * A)                          # scalar decay per head, A < 0
    beta_t  = (1 - lambda_t) * dt_t * alpha_t        # previous-input (trapezoidal) weight
    gamma_t = lambda_t * dt_t                        # current-input weight
    h_t = alpha_t * h_{t-1}
          + gamma_t * outer(x_t,     RoPE(B_t,     Phi_t))
          + beta_t  * outer(x_{t-1}, RoPE(B_{t-1}, Phi_{t-1}))
    y_t = < h_t , RoPE(C_t, Phi_t) >
Phi_t is the cumulative, data-dependent rotary angle; rotating B at write-time and C
at read-time is the real-valued realisation of the complex state transition
A_t = alpha_t * R(theta_t)  (scalar decay TIMES rotation -- the Mamba-2/3 SSD
structure).  lambda_t = 1  ->  exponential-Euler  ==  Mamba-2 (single input term):
that is the knob for the trapezoidal ablation.  use_rope=False -> real-valued
diagonal SSM: the control for the parity/state-tracking gate.

WHAT CHANGED vs. the first version of this file (and why the detector NaN'd)
--------------------------------------------------------------------------
1. The chunked scan used  intra = Pin * cumsum(U / Pin)  with
   Pin = cumprod(alpha).clamp(min=1e-6).  alpha = exp(dt*A) drops well below 1 after
   a few optimizer steps, so alpha**chunk underflows (in fp16 already for
   alpha < ~0.6), the clamp kicks in, and U / 1e-6 overflows fp16 -> inf -> NaN on
   every batch.  That is exactly what the run log shows: NaN from epoch 2 and a
   *frozen* model (identical val metrics every epoch = every AMP step skipped).
   It is now replaced by the segment-sum ("SSD") form:
       h_t = sum_{s<=t} exp(cs_t - cs_s) U_s ,   cs = cumsum(log alpha)
   Every exp() argument is <= 0, so every factor lives in [0, 1].  No division, no
   clamp, no overflow -- and it is *exactly* the same recurrence (see
   scripts/validate_core.py, which checks it against the O(L) loop).
2. A is now a scalar per head (shape (nheads,)) instead of (nheads, d_state).  That
   is the Mamba-2/3 structure (A_t = alpha_t * R_t: scalar decay, rotation carries
   the state-dependence) and it lets the readout be contracted analytically, so the
   (B, L, H, P, N) state trajectory is never materialised.  Memory drops ~30x and
   the block gets several times faster.
3. The SSM math runs with autocast disabled (fp32).  exp / cumsum / rotation over
   L = 4096..16384 tokens are not fp16-safe; the projections still run in fp16.
4. The rotation rate is bounded (rot_max * tanh) and Phi is wrapped mod 2*pi.
   Unbounded w_proj summed over 16k tokens produced |Phi| >> 1e4, which destroys
   cos/sin precision (and overflows fp16 outright).  pi rad/step is still reachable,
   so the parity gate still passes.
5. token_saliency: F.pad(beta, (0,0,0,0,0,1)) padded the BATCH dim of a 3-D tensor
   (beta is (B,L,H)), so beta_next came out length L-1 -> the
   "size of tensor a (16383) must match tensor b (16384)" crash in the XAI cell.
6. _reverse_energy was a Python loop over L (16384 steps x 4 directions x 8 blocks).
   It is now a vectorised reverse scan reusing the same bounded machinery.

MIMO extension (not enabled): make B_proj / C_proj emit (N * r) and reshape to
(N, r), replace the outer product with a matmul over the rank-r channel group.
Enable only once validated -- until then, SISO is what runs, and that is what any
paper text should claim.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_NEG_INF = float("-inf")


def _rope(v: Tensor, phi: Tensor) -> Tensor:
    """Rotate the state dimension of v in 2-D blocks by angles phi.

    v:   (..., N)      N even; consecutive pairs form the 2-D rotation blocks.
    phi: (..., N // 2) rotation angle per block.
    """
    v2 = v.unflatten(-1, (-1, 2))                      # (..., N/2, 2)
    cos, sin = torch.cos(phi), torch.sin(phi)
    x0, x1 = v2[..., 0], v2[..., 1]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)


def _causal_decay(cs: Tensor) -> Tensor:
    """D[b,c,t,s,h] = exp(cs_t - cs_s) for s <= t, else 0.   cs: (B, nc, chunk, H).

    cs is an inclusive cumsum of log(alpha) <= 0, so for s <= t the exponent is <= 0
    and D is in [0, 1]. The strictly-upper triangle would have a positive exponent,
    so it is masked to -inf BEFORE the exp (never after -- that would overflow).
    """
    c = cs.shape[2]
    dec = cs.unsqueeze(3) - cs.unsqueeze(2)            # (B, nc, t, s, H)
    causal = torch.ones(c, c, dtype=torch.bool, device=cs.device).tril()
    return torch.exp(dec.masked_fill(~causal[None, None, :, :, None], _NEG_INF))


class Mamba3RefSSM(nn.Module):
    """Selective SSM with exponential-trapezoidal discretization + complex (RoPE) states.

    Input/output are (B, L, D).
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
        chunk: int = 64,
        rot_max: float = 2.0 * math.pi,
    ):
        super().__init__()
        assert d_state % 2 == 0, "d_state must be even for 2-D rotary blocks"
        self.use_rope = use_rope          # False -> real-valued diagonal SSM (ablation control)
        self.trapezoidal = trapezoidal    # False -> lambda=1 = exponential-Euler = Mamba-2
        self.parallel = True              # chunked SSD scan; False -> O(L) reference loop
        self.chunk = int(chunk)
        self.rot_max = float(rot_max)     # max |rotation rate| per step (>= pi -> parity reachable)
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.headdim = min(headdim, self.d_inner)
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)   # x, z(gate)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dt_proj = nn.Linear(self.d_inner, self.nheads, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=True)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=True)
        self.lam_proj = nn.Linear(self.d_inner, self.nheads, bias=True)
        # Rotation RATE (imaginary part of complex A): data-dependent, decoupled from the
        # small real decay step dt, and bounded so cumulative angles stay well conditioned.
        # Must reach ~pi/step to represent parity-style rotational state tracking.
        self.w_proj = nn.Linear(self.d_inner, self.nheads, bias=True)

        # A < 0 via -exp(A_log); ONE scalar decay rate per head (Mamba-2/3 SSD structure).
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.norm = nn.LayerNorm(self.d_inner)

        theta = rope_base ** (-torch.arange(0, d_state, 2).float() / d_state)  # (N/2,)
        self.register_buffer("theta", theta, persistent=False)

        with torch.no_grad():
            dt = torch.empty(self.nheads).uniform_(math.log(dt_min), math.log(dt_max)).exp()
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    # ------------------------------------------------------------------ pieces
    def _rotate(self, w_proj_out: Tensor, Bt: Tensor, Ct: Tensor) -> Tuple[Tensor, Tensor]:
        B, L, _ = Bt.shape
        H, N = self.nheads, self.d_state
        Bh = Bt.float().unsqueeze(2).expand(B, L, H, N)
        Ch = Ct.float().unsqueeze(2).expand(B, L, H, N)
        if not self.use_rope:
            return Bh, Ch                              # real-valued control: no rotation
        w = self.rot_max * torch.tanh(w_proj_out.float())          # (B,L,H) in (-rot_max, rot_max)
        phi = torch.cumsum(w.unsqueeze(-1) * self.theta.float(), dim=1)  # (B,L,H,N/2)
        phi = torch.remainder(phi, 2.0 * math.pi)                # exact, keeps cos/sin accurate
        return _rope(Bh, phi), _rope(Ch, phi)

    def _coeffs(self, dt_proj_out: Tensor, lam_proj_out: Optional[Tensor]):
        """dt/lambda/A -> (log_alpha, gamma, beta). beta is None for the Euler ablation."""
        dt = F.softplus(dt_proj_out.float())                     # (B,L,H) > 0
        log_alpha = dt * (-torch.exp(self.A_log.float()))        # (B,L,H) <= 0
        if not self.trapezoidal:
            return log_alpha, dt, None                           # lambda = 1
        lam = torch.sigmoid(lam_proj_out.float())                # (B,L,H) in [0,1]
        gamma = lam * dt
        beta = (1.0 - lam) * dt * torch.exp(log_alpha)
        return log_alpha, gamma, beta

    # ------------------------------------------------------------------ forward
    def forward(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        H, P = self.nheads, self.headdim

        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)                     # (B, L, d_inner) each
        u = self.norm(u)

        # Projections run natively in whatever dtype model is in (e.g. float16 during val)
        dt_val = self.dt_proj(u)
        lam_val = self.lam_proj(u) if self.trapezoidal else None
        B_val = self.B_proj(u)
        C_val = self.C_proj(u)
        w_val = self.w_proj(u) if self.use_rope else None

        # The SSM core is not fp16-safe (exp / cumsum / rotation over thousands of
        # tokens). The sensitive math is forced to fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            log_alpha, gamma, beta = self._coeffs(dt_val, lam_val)
            Brot, Crot = self._rotate(w_val, B_val, C_val)
            uh = u.float().view(B, L, H, P)
            scan = self._ssd if self.parallel else self._ssm_reference
            y = scan(log_alpha, gamma, beta, uh, Brot, Crot)     # (B,L,H,P)
            y = y.reshape(B, L, self.d_inner)

        y = y.to(z.dtype) * F.silu(z)                  # gate
        return self.out_proj(y)

    # ------------------------------------------------------------------ scans
    @staticmethod
    def _shift(t: Tensor) -> Tensor:
        """x_{t-1} along dim 1, zero-filled at t=0. t: (B, L, ...) with 2 trailing dims."""
        L = t.shape[1]
        return F.pad(t, (0, 0, 0, 0, 1, 0))[:, :L]

    def _ssd(self, log_alpha: Tensor, gamma: Tensor, beta: Optional[Tensor],
             x: Tensor, Br: Tensor, Cr: Tensor) -> Tensor:
        """Chunked segment-sum scan. Mathematically identical to _ssm_reference.

        log_alpha, gamma, beta: (B,L,H);  x: (B,L,H,P);  Br, Cr: (B,L,H,N)  -> y: (B,L,H,P)

        Because A is scalar per head, the readout <h_t, C_t> can be contracted in closed
        form, so the (B,L,H,P,N) state trajectory is never built:
            y_t = sum_{s<=t} D[t,s] * (B_s . C_t) * w_s * x_s        (intra-chunk)
                + exp(cs_t) * <h_chunk_start, C_t>                   (inter-chunk)
        """
        B, L, H, P = x.shape
        N = Br.shape[-1]
        c = max(1, min(self.chunk, L))
        xp, Bp = self._shift(x), self._shift(Br)       # trapezoidal (previous-token) drive

        pad = (-L) % c
        if pad:
            log_alpha = F.pad(log_alpha, (0, 0, 0, pad))          # log alpha = 0 -> alpha = 1
            gamma = F.pad(gamma, (0, 0, 0, pad))                  # gamma = 0 -> no drive
            beta = None if beta is None else F.pad(beta, (0, 0, 0, pad))
            x, xp = F.pad(x, (0, 0, 0, 0, 0, pad)), F.pad(xp, (0, 0, 0, 0, 0, pad))
            Br, Bp = F.pad(Br, (0, 0, 0, 0, 0, pad)), F.pad(Bp, (0, 0, 0, 0, 0, pad))
            Cr = F.pad(Cr, (0, 0, 0, 0, 0, pad))
        Lp, nc = L + pad, (L + pad) // c

        la = log_alpha.view(B, nc, c, H)
        gc = gamma.view(B, nc, c, H)
        xc, xpc = x.view(B, nc, c, H, P), xp.view(B, nc, c, H, P)
        Brc, Bpc = Br.view(B, nc, c, H, N), Bp.view(B, nc, c, H, N)
        Crc = Cr.view(B, nc, c, H, N)
        bc = None if beta is None else beta.view(B, nc, c, H)

        cs = la.cumsum(2)                              # (B,nc,c,H) inclusive
        D = _causal_decay(cs)                          # (B,nc,t,s,H) in [0,1]

        # ---- intra-chunk readout
        att = torch.einsum("bcthn,bcshn->bctsh", Crc, Brc) * D * gc.unsqueeze(2)
        y = torch.einsum("bctsh,bcshp->bcthp", att, xc)
        if bc is not None:
            attp = torch.einsum("bcthn,bcshn->bctsh", Crc, Bpc) * D * bc.unsqueeze(2)
            y = y + torch.einsum("bctsh,bcshp->bcthp", attp, xpc)

        # ---- per-chunk end state, then the only sequential part: nc = L/chunk steps
        Sc = torch.exp(cs[:, :, -1:, :] - cs)          # (B,nc,c,H) in (0,1]: decay s -> chunk end
        h_local = torch.einsum("bcshp,bcshn->bchpn", (Sc * gc).unsqueeze(-1) * xc, Brc)
        if bc is not None:
            h_local = h_local + torch.einsum("bcshp,bcshn->bchpn",
                                             (Sc * bc).unsqueeze(-1) * xpc, Bpc)
        adv = torch.exp(cs[:, :, -1, :])               # (B,nc,H) full-chunk decay, in (0,1]
        states, h = [], x.new_zeros(B, H, P, N)
        for i in range(nc):
            states.append(h)                           # state entering chunk i
            h = adv[:, i, :, None, None] * h + h_local[:, i]
        states = torch.stack(states, 1)                # (B,nc,H,P,N)

        # ---- inter-chunk readout
        y = y + torch.einsum("bcthn,bchpn->bcthp", Crc, states) * torch.exp(cs).unsqueeze(-1)
        return y.reshape(B, Lp, H, P)[:, :L]

    def _ssm_reference(self, log_alpha: Tensor, gamma: Tensor, beta: Optional[Tensor],
                       x: Tensor, Br: Tensor, Cr: Tensor) -> Tensor:
        """Sequential O(L) ground truth. Correct and honest, not fast."""
        B, L, H, P = x.shape
        N = Br.shape[-1]
        alpha = torch.exp(log_alpha)
        xp, Bp = self._shift(x), self._shift(Br)
        h = x.new_zeros(B, H, P, N)
        out = []
        for t in range(L):
            U = gamma[:, t, :, None, None] * x[:, t, :, :, None] * Br[:, t, :, None, :]
            if beta is not None:
                U = U + beta[:, t, :, None, None] * xp[:, t, :, :, None] * Bp[:, t, :, None, :]
            h = alpha[:, t, :, None, None] * h + U
            out.append((h * Cr[:, t, :, None, :]).sum(-1))
        return torch.stack(out, dim=1)

    def _scan_scalar(self, log_a: Tensor, u: Tensor) -> Tensor:
        """h_j = exp(log_a_j) * h_{j-1} + u_j, vectorised. Both (B, L, H) -> (B, L, H)."""
        B, L, H = u.shape
        c = max(1, min(self.chunk, L))
        pad = (-L) % c
        if pad:
            log_a, u = F.pad(log_a, (0, 0, 0, pad)), F.pad(u, (0, 0, 0, pad))
        Lp, nc = L + pad, (L + pad) // c
        la, uc = log_a.view(B, nc, c, H), u.view(B, nc, c, H)
        cs = la.cumsum(2)
        y = torch.einsum("bctsh,bcsh->bcth", _causal_decay(cs), uc)
        Sc = torch.exp(cs[:, :, -1:, :] - cs)
        h_local = (Sc * uc).sum(2)                     # (B,nc,H)
        adv = torch.exp(cs[:, :, -1, :])
        states, h = [], u.new_zeros(B, H)
        for i in range(nc):
            states.append(h)
            h = adv[:, i] * h + h_local[:, i]
        y = y + torch.stack(states, 1).unsqueeze(2) * torch.exp(cs)
        return y.reshape(B, Lp, H)[:, :L]

    def _reverse_energy(self, log_alpha: Tensor) -> Tensor:
        """G_tau = sum_{t>=tau} (prod_{k=tau+1..t} alpha_k)^2, i.e. the stable reverse
        recurrence G_tau = 1 + alpha_{tau+1}^2 * G_{tau+1}, G_{L-1} = 1.  (B,L,H).

        Written as a forward scan on the reversed sequence so it costs O(L/chunk) steps
        instead of O(L) Python iterations (L is 4k-16k tokens per block here).
        """
        L = log_alpha.shape[1]
        af = torch.flip(2.0 * log_alpha, dims=[1])               # af[j] = 2*log alpha_{L-1-j}
        logc = F.pad(af, (0, 0, 1, 0))[:, :L]                    # c_j = alpha_{L-j}^2, c_0 free
        g = self._scan_scalar(logc, torch.ones_like(logc))       # g_j = c_j g_{j-1} + 1
        return torch.flip(g, dims=[1])

    # ------------------------------------------------------------------ XAI
    @torch.no_grad()
    def token_saliency(self, x: Tensor) -> Tensor:
        """Intrinsic controllability-energy saliency per input token. Returns (B, L).
        Closed-form from the SSM internals, one pass, no backprop.

        Accounts for BOTH Gramian contributions of each token:
          - gamma term: token t's direct drive at step t
          - beta  term: token t's carry-forward drive at step t+1 (trapezoidal)
        """
        B, L, _ = x.shape
        H, P = self.nheads, self.headdim
        u, _ = self.in_proj(x).chunk(2, dim=-1)
        u = self.norm(u)
        
        dt_val = self.dt_proj(u)
        lam_val = self.lam_proj(u) if self.trapezoidal else None
        B_val = self.B_proj(u)
        w_val = self.w_proj(u) if self.use_rope else None

        with torch.autocast(device_type=x.device.type, enabled=False):
            log_alpha, gamma, beta = self._coeffs(dt_val, lam_val)
            Brot, _ = self._rotate(w_val, B_val, B_val)
            uh = u.float().view(B, L, H, P)
            G = self._reverse_energy(log_alpha)                  # (B,L,H)
            bnorm = (Brot ** 2).sum(-1)                          # (B,L,H)
            xnorm = (uh ** 2).sum(-1)                            # (B,L,H)
            s = (gamma ** 2) * xnorm * bnorm * G                 # token t drives state at t
            if beta is not None:
                # token t also drives the state at t+1 with weight beta_{t+1}
                beta_next = F.pad(beta, (0, 0, 0, 1))[:, 1:L + 1]
                G_next = F.pad(G, (0, 0, 0, 1))[:, 1:L + 1]
                s = s + (beta_next ** 2) * xnorm * bnorm * G_next
        return s.mean(-1)                                        # (B, L)
