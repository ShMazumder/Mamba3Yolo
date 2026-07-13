# Research Brief — "X-LPM Mamba" proposal vs. reality

**Date:** 2026-07-13
**Sources:** Gemini Deep Research report (share.gemini.google/FuslUhUqHPRg); Mamba-3 paper (arXiv:2603.15569, verified); current Mamba3Yolo repo. NotebookLM notebook inaccessible (auth wall).

## What the report proposes
**X-LPM Mamba** (Explainable, Low-Precision, Multi-Dataset) — a unified real-time detector with **four** contributions:
1. **Spatial Mamba-3 neck** — Mamba-3 recurrence + Structure-Aware State Fusion (SASF: dilated depthwise conv on the complex latent states) to keep 2D geometry without 4-directional scan.
2. **Intrinsic control-theoretic XAI** — controllability Gramian of the diagonal SSM, closed-form in one forward pass, O(T), no backprop. Replaces Grad-CAM.
3. **Outlier-aware mixed-precision PTQ/QAT** — per-token static scaling + dynamic FP16 outlier routing + absmean scaling; targets ternary/INT4.
4. **GNN taxonomy alignment** — unified label graph across COCO / BDD100K / VisDrone with per-dataset projection matrices `M_i` to kill co-training gradient conflict.

## Grounding check — the literature is real and correctly used
- Mamba-3 equations in the report **match the actual paper** (α/β/γ trapezoidal, complex-state = data-dependent RoPE). Correct.
- MambaOut (CVPR 2025): correctly cited — SSM unhelpful for classification, helpful for **detection/long-seq**. This genuinely *supports* the framing.
- Spatial-Mamba (SASF), X-VMamba (Gramian XAI), PTQ4VM / OuroMamba / MambaQuant (SSM quant): all real, on-topic.
- Verdict: the *related work and motivation are solid and current*. The intellectual framing is publishable-grade.

## The two fatal problems
1. **Tables 3 & 4 are fabricated.** "X-LPM Mamba-3 (Ours): 45.4% COCO mAP, 6.2ms Jetson Orin, 285 FPS mixed-precision" — no such model has been trained. The repo runs a gated-MLP that isn't even an SSM, with zero mAP measured. Presenting invented numbers as results = desk-reject + research-integrity violation. Same failure class as the `official_mamba3_kernels: true` flag.
2. **Scope = 3–4 papers, not one.** SASF neck, Gramian XAI, mixed-precision quant, and GNN multi-dataset are each a paper. A single submission claiming all four, with no real results, reads as AI-generated vaporware to any reviewer. Over-scoping *lowers* acceptance odds.

## Feasibility ranking (what's real vs. hand-wavy)
| Contribution | Novelty | Feasibility now | Note |
|---|---|---|---|
| Mamba-3 core for detection | High | Medium | Have a verified reference SSM (`mamba3_ref.py`); needs real training |
| Gramian intrinsic XAI | High | **High** | Closed-form, cheap, genuinely novel pairing with detection. Best bet. |
| SASF on complex states | Medium | Medium | Spatial-Mamba exists; the "on complex states" twist is the delta |
| Mixed-precision quant | Medium | Low | Needs a working trained model first; downstream of everything |
| GNN taxonomy | Medium | Low | Heavy, orthogonal, its own project |

## Minimum Publishable Unit (recommendation)
**One paper:** "Mamba-3 selective SSM for real-time detection **+ intrinsic controllability-Gramian explainability**." Two contributions, tightly coupled (the complex diagonal SSM is exactly what makes the Gramian closed-form — the XAI *falls out of* the architecture). Defensible, novel, and cheap to validate. Bank quant + multi-dataset + SASF as follow-ups.

## Actionable roadmap (ordered)
1. Swap the verified `Mamba3RefSSM` into the ODSS block; validate on the paper's synthetic state-tracking (parity) task — honesty gate.
2. Train on ONE real dataset, report **true mAP** vs Mamba-YOLO + a modern YOLO at matched params.
3. Implement the Gramian influence score `I(t,τ)` (report's formula is correct in form); show it beats Grad-CAM++ on insertion/deletion + pointing-game.
4. Delete every fabricated table. Numbers or nothing.
5. Only then consider quant / multi-dataset as a v2.

## Risks / contrarian
- The complex-arithmetic-on-edge concern the report raises against itself is real: Jetson tensor cores are INT8/FP16; complex ops double memory traffic. The claimed 6.2ms latency is likely optimistic — measure before claiming.
- "Real-time + explainable + 4-bit + multi-dataset" is a lot of hedges; reviewers read breadth-without-depth as a red flag. Depth on two beats breadth on four.

---

## Competitive map (verified 2026-07-13)

Two distinct lanes — don't conflate them:

**Lane A — SSM backbones / scan design** (Mask/Cascade R-CNN heads, 3× schedule, NOT real-time):
| Paper | Verified fact | Occupies |
|---|---|---|
| Mamba2D (arXiv 2412.16146) | **52.2 box AP COCO** (3×), 84.0% IN-1K @27M, code out | Native 2D scan geometry |
| SF-Mamba (arXiv 2603.16423, Mar 2026) | Cascade Mask R-CNN 3×, +~1.0 AP^b over baseline, throughput-first | Bidirectional info under **unidirectional scan** |
| DAMamba / DefMamba (NeurIPS'25) | dynamic/deformable adaptive scan | Adaptive scan directions |

**Lane B — real-time Mamba detectors (YOLO-style)** — your actual lane:
| Paper | Status | Note |
|---|---|---|
| Mamba-YOLO (AAAI 2025) | published, real | the thing being "upgraded" |
| AKCMamba-YOLO (CVPR 2026) | **VERIFIED from PDF: 46.3 mAP / 63.1 AP50 / 9.1M / 14.9G on COCO val, V100.** S6-style (AKSS2D), YOLOv8 base, C2f→mamba. Ships Grad-CAM (their Fig 4). | direct SOTA competitor. Gemini's 42.1%/9.8M/7.1ms was FABRICATED (real=46.3/9.1M). |
| MambaNeXt-YOLO (arXiv 2506.03654) | real, hybrid SSM real-time | edge-focused |
| Mamba-YOLO-World (arXiv 2409.08513) | real | open-vocab |

**Whitespace after cross-referencing all of the above:**
- Every competitor is **Mamba-1/2/S6**. None use **Mamba-3** (complex-state, trapezoidal — only exists since Mar 2026). → "first Mamba-3 detector" is open but closing.
- **No competitor does explainability.** → controllability-Gramian XAI is uncontested, and only closed-form because Mamba-3 is complex-diagonal. The moat.
- Scan-geometry (proposed SASF) is contested by 3+ backbone papers. Abandon as a novelty claim.

**Baselines a real submission must beat:** Mamba-YOLO (same lane, matched params) is the mandatory head-to-head; Mamba2D's 52.2 AP is the backbone ceiling to acknowledge (different lane). AKCMamba numbers must be obtained from the actual CVPR PDF before any comparison — not from the fabricated report.
