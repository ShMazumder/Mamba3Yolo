# Mamba3Yolo Research Plan (Senior CV / AI Research)

## Precise Research Question
Does upgrading the selective SSM in Mamba-YOLO from Mamba-1/2-style (SISO + basic discretization) to Mamba-3 (exponential-trapezoidal discretization + complex-valued states via RoPE + MIMO) yield statistically significant gains in mAP, small-object AP, multi-domain generalization, and edge efficiency (latency/power after quantization) for real-time object detection?

## Why It Matters
- MambaOut (CVPR 2025) showed pure SSM underperforms on classification but retains value for long-sequence dense tasks (detection/segmentation).
- All current Mamba-YOLO / hybrids (AAAI 2025 – CVPR 2026) still use older SSM cores.
- Mamba-3 (Mar 2026) specifically improves state tracking (critical for precise localization) and inference efficiency (MIMO without state inflation).
- Edge + multi-domain + explainable detectors remain under-served for practical deployment (medical mobile, drones, LMIC settings).

## Closest Prior Work & Delta
- **Mamba-YOLO (Wang et al., AAAI 2025)**: ODSSBlock = LS + SS2D(S6) + RG. Strong baseline. **Delta**: replace SS2D/S6 with Mamba3SS2D.
- **AKCMamba-YOLO, MambaNeXt-YOLO, etc.**: Further architectural tweaks on old SSM. **Delta**: core SSM upgrade + systematic multi-domain + quant + XAI.
- **PTQ4VM / OuroMamba**: Quant for Vision Mamba backbones. **Delta**: full detector (backbone+neck+head) + detection-specific calibration.
- **DA-Mamba**: Domain adaptive. **Delta**: combine with Mamba-3 + multi-dataset joint training.

## Experimental Design (Controlled Ablations)
1. **Core ablations** (COCO val, 3 seeds):
   - Baseline: original Mamba-YOLO-T/M (reproduced).
   - +Exp-Trapezoidal only
   - +Complex (RoPE) only
   - +MIMO only
   - Full Mamba3 (all three)
   - Report mAP, AP_s, AP_m, AP_l, FPS (V100/A100 + Jetson Orin), params, FLOPs, peak mem.

2. **Multi-domain**:
   - Train on COCO + VisDrone + BDD100K (or medical: polyps + cells + DR lesions).
   - Evaluate zero-shot / few-shot transfer + domain gap reduction.
   - Optional source-free TTA.

3. **Quantization**:
   - PTQ (extend PTQ4VM) and QAT (W8A8, W4A8).
   - Measure accuracy drop, latency, power (if hardware available).

4. **Explainability**:
   - SSM-state derived saliency (selectivity parameters / hidden-state norms).
   - Grad-CAM++ on last Mamba3 / Conv layers.
   - Quantitative: Insertion/Deletion, Pointing Game, human preference (if possible).
   - Use XAI for hard-example mining or pseudo-label filtering in multi-domain.

5. **Edge**:
   - ONNX → TensorRT / TFLite.
   - Latency P50/P99, model size post-quant, energy if measurable.

## Metrics & Stats
- Primary: mAP@0.5:0.95, AP_s.
- Secondary: FPS, power, XAI faithfulness scores.
- Statistical: 3–5 seeds, mean ± std, paired t-test or Wilcoxon vs strongest baseline.
- Failure analysis: small / occluded / domain-shift cases + qualitative heatmaps.

## Success Criteria for Publication (CVPR / ICCV / ECCV / TPAMI / EdgeAI)
- Clear, significant gains from Mamba-3 components (ablation table).
- Competitive or better Pareto (accuracy vs latency/size) vs 2025–2026 SOTA real-time detectors.
- Open-source code + weights + multi-domain results.
- Honest limitations section (e.g., kernel dependency, training cost of MIMO, when complex helps most).

## Timeline Suggestion
- Week 1–2: Clean integration + shape/unit tests + single-GPU COCO baseline.
- Week 3–4: Full ablations + multi-seed.
- Week 5: Quant + XAI + multi-domain.
- Week 6: Paper draft (methods, related work, results) + humanization + architecture diagrams.
- Target: CVPR 2027 or concurrent workshop / journal.

## Code Reproducibility Checklist
- [ ] Config-driven (Hydra or YAML)
- [ ] Seed everything
- [ ] Official Mamba-3 kernels pinned
- [ ] Exact train/val splits published
- [ ] WandB / MLflow logging
- [ ] Export scripts for ONNX/TensorRT
- [ ] Grad-CAM++ visualization script

This plan is ready for execution. The provided `src/blocks/mamba3_odss.py` is the core drop-in upgrade.
