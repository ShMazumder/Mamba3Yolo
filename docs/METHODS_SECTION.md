# 3. Methods

## 3.1 Overview

We present Mamba3Yolo, a real-time object detector that upgrades the selective state-space core of Mamba-YOLO with the three methodological advances introduced in Mamba-3. The design keeps the proven local-spatial and residual-gated modules of the original architecture while replacing the older S6-based SS2D with a Mamba-3 mixer that uses exponential-trapezoidal discretization, complex-valued state transitions, and an optional multi-input multi-output (MIMO) formulation. The resulting hybrid remains linear-complexity, is easy to train from scratch, and provides natural hooks for post-training quantization and gradient-based explainability.

The detector follows the classic YOLO one-stage layout: a hierarchical backbone that produces multi-scale features, a path-aggregation neck that fuses them, and a decoupled detection head. All heavy feature mixing blocks (both backbone stages and neck) are instances of the new Mamba3ODSSBlock.

## 3.2 Mamba3ODSSBlock

The original ODSSBlock of Mamba-YOLO consists of three parts executed in sequence: a local spatial block (LSBlock), a 2-D selective scan (SS2D) that runs a Mamba-1/2 style S6 core on four directional sequences, and a residual gated block (RGBlock) that mixes channels. We keep LSBlock and RGBlock unchanged because pure state-space models still under-perform convolutional inductive biases on purely local patterns (see MambaOut). The only change is the middle mixer.

**Mamba3SS2D.**  
An input feature map \(X \in \mathbb{R}^{B \times C \times H \times W}\) is flattened into four sequences (left-to-right, right-to-left, top-to-bottom, bottom-to-top). Each sequence is processed by a shared Mamba-3 module whose recurrence is

\[
\mathbf{h}_t = \alpha_t R \mathbf{h}_{t-1} + \beta_t R B_t x_t + \gamma_t B_{t-1} x_{t-1},
\]

where the complex rotation \(R\) is realized by a RoPE-style transformation (the practical implementation of the complex-valued state) and \(\alpha_t, \beta_t, \gamma_t\) come from the exponential-trapezoidal discretization of the continuous SSM. When MIMO mode is enabled the input and output projections become low-rank multi-channel matrices of rank \(r\) (default \(r=4\)), increasing modeling capacity without enlarging the recurrent state that must be cached at inference time. The four directional outputs are reshaped back to 2-D and averaged. A residual connection and optional DropPath complete the block.

Because the official Mamba-3 CUDA kernels are used, the block inherits the same hardware-efficient scan and the improved state-tracking properties that Mamba-3 demonstrated on language benchmarks. In practice this translates to sharper localization of small or occluded objects, which we verify later with ablation tables.

## 3.3 Overall Architecture

**Backbone.**  
A lightweight stem of two strided convolutions produces a feature map at \(1/4\) resolution. Four stages then follow, each composed of one or more Mamba3ODSSBlocks interleaved with strided convolutions that double the channel count and halve spatial size. The final stage is followed by a Spatial Pyramid Pooling-Fast (SPPF) module. The stage depths and channel widths follow the standard YOLO scaling rules (T/M/L) so that the parameter counts remain comparable to the original Mamba-YOLO variants.

**Neck.**  
We retain a simplified Path Aggregation Network (PAFPN). Lateral connections and up-/down-sampling convolutions are ordinary \(1\times1\) or \(3\times3\) layers; the heavy mixing inside each fusion node is again performed by a Mamba3ODSSBlock. This keeps the long-range context modeling of Mamba-3 available at every scale that the detection head sees.

**Head.**  
A standard decoupled head predicts bounding-box distributions (Distribution Focal Loss style) and class scores from the three fused feature maps (P3/P4/P5). No architectural change is made to the head so that any future improvements in YOLO losses or assigners can be plugged in directly.

## 3.4 Medical Multi-Domain Training

For clinical deployment we also support joint training on several public medical detection collections (polyp, blood-cell, brain-tumor, and, when available, diabetic-retinopathy lesion boxes). All datasets are converted to a common YOLO-format folder structure and mapped onto a unified label space. Domain-balanced sampling (or simple concatenation followed by stratified epochs) is used so that no single imaging modality dominates the gradient. The same Mamba3Yolo weights can therefore be evaluated zero-shot or few-shot on any of the constituent domains, which is essential for low- and middle-income country settings where a single model must handle heterogeneous equipment.

## 3.5 Explainability

Two complementary explanations are generated for every prediction.  

1. **Grad-CAM++** is computed on the last Mamba3ODSSBlock of each scale (or on the neck outputs immediately before the head). The resulting heatmaps highlight the spatial regions that most influence the class score.  
2. **SSM-state saliency** uses the magnitude of the recurrent hidden state (or the selectivity parameters \(B,C\)) as a cheap proxy for token importance along each scan direction.  

Both maps are overlaid on the original image (fundus, endoscopic, or microscopy) and can be inspected by a clinician. Quantitative faithfulness is measured by insertion/deletion curves and pointing-game accuracy; these numbers appear in the experimental section.

## 3.6 Quantization

After training we apply post-training quantization (PTQ). Activation statistics are collected on a small calibration set (256 images drawn from the validation split or from a held-out medical subset). For the Mamba-3 layers we follow the recommendations of PTQ4VM and OuroMamba (per-token static scaling and outlier-aware channel handling). The remaining convolutional layers use standard min-max or SmoothQuant scaling. The full detector (backbone + neck + head) is then exported to ONNX and further optimized with TensorRT INT8 engines. We also provide a quantization-aware training (QAT) entry point for cases where the PTQ drop exceeds 1 mAP point.

## 3.7 Implementation Notes

The model is implemented in pure PyTorch and can be trained with the self-contained script supplied in the repository. For maximum compatibility with existing YOLO tooling the same Mamba3ODSSBlock can be registered inside the original Mamba-YOLO / Ultralytics code base with a five-line patch (see Integration documentation). All experiments use automatic mixed precision, cosine learning-rate schedules, and the standard Mosaic / MixUp / random affine augmentations of the YOLO family. Code, configuration files, and pretrained weights will be released upon publication.

---

*Word count of this section is suitable for a full CVPR/ICCV paper; the experimental section will report the corresponding ablations on COCO and the medical multi-domain suite.*
EOF