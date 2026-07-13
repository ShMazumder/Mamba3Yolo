# Graph Report - .  (2026-07-13)

## Corpus Check
- Corpus is ~15,961 words - fits in a single context window. You may not need a graph.

## Summary
- 226 nodes · 336 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.76)
- Token cost: 190,173 input · 0 output

## Community Hubs (Navigation)
- Mamba-3 Architecture Concepts
- ODSS Block Implementation
- Post-Training Quantization
- Training & Latency Harness
- Paper Submission & Supplementary
- Detector Model Assembly
- Grad-CAM++ XAI
- COCO mAP Evaluation
- Pure-PyTorch mAP Evaluator
- Architecture Diagram
- Detection Datasets
- YOLO Loss Function
- Training Curves Diagnostics
- XAI Figure Script
- Latency Benchmark

## God Nodes (most connected - your core abstractions)
1. `Mamba3ODSSBlock` - 11 edges
2. `Mamba3Yolo` - 11 edges
3. `Mamba3Yolo Supplementary Material` - 10 edges
4. `Mamba3ODSSBlock` - 9 edges
5. `YoloFolderDataset` - 8 edges
6. `build_mamba3yolo()` - 8 edges
7. `main()` - 7 edges
8. `Mamba3SS2D` - 7 edges
9. `COCOEvaluator` - 7 edges
10. `ConvBNAct` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Reproducibility Requirements` --semantically_similar_to--> `Reproducibility Checklist`  [INFERRED] [semantically similar]
  docs/CAMERA_READY_CHECKLIST.md → docs/latex/supplementary.pdf
- `YoloFolderDataset` --uses--> `Mamba3Yolo`  [INFERRED]
  scripts/train.py → src/models/mamba3yolo.py
- `SimpleDetectionLoss` --uses--> `Mamba3Yolo`  [INFERRED]
  scripts/train.py → src/models/mamba3yolo.py
- `main()` --calls--> `build_mamba3yolo()`  [EXTRACTED]
  scripts/generate_xai_figures.py → src/models/mamba3yolo.py
- `Mamba3Yolo-T Model Config` --implements--> `Mamba3Yolo`  [INFERRED]
  configs/models/Mamba3Yolo-T.yaml → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Three Mamba-3 Methodological Advances** — docs_methods_section_exp_trapezoidal_discretization, docs_methods_section_complex_valued_states, docs_methods_section_mimo [EXTRACTED 1.00]
- **Mamba3ODSSBlock Composition** — docs_methods_section_lsblock, docs_methods_section_mamba3ss2d, docs_methods_section_rgblock, configs_models_mamba3yolo_t_mamba3odssblock [EXTRACTED 1.00]
- **Medical Multi-Domain Training Set** — configs_datasets_medical_multi_polyp_domain, configs_datasets_medical_multi_blood_cell_domain, configs_datasets_medical_multi_brain_tumor_domain, configs_datasets_medical_multi_medical_multi_domain [EXTRACTED 1.00]
- **Mamba3ODSSBlock Internal Composition** — docs_latex_figures_architecture_mamba3odss_block, docs_latex_figures_architecture_lsblock, docs_latex_figures_architecture_mamba3ss2d, docs_latex_figures_architecture_rgblock [EXTRACTED 1.00]
- **Edge Quantization Pipeline** — docs_latex_supplementary_quantization_protocol, docs_latex_supplementary_ptq4vm, docs_latex_supplementary_smoothquant, docs_latex_supplementary_tensorrt [EXTRACTED 1.00]

## Communities (15 total, 2 thin omitted)

### Community 0 - "Mamba-3 Architecture Concepts"
Cohesion: 0.08
Nodes (35): Mamba3ODSSBlock, Mamba3Yolo-T Model Config, ODMamba3 Backbone, PAFPN Detection Head, SPPF Module, Mamba-YOLO Integration Patch, mamba3_odss.py Block Module, mamba-ssm (Official Mamba-3 kernels) (+27 more)

### Community 1 - "ODSS Block Implementation"
Cohesion: 0.14
Nodes (11): build_mamba3_odss(), DropPath, LSBlock, Mamba3ODSSBlock, Mamba3Reference, Mamba3SS2D, Module, Tensor (+3 more)

### Community 2 - "Post-Training Quantization"
Cohesion: 0.16
Nodes (17): DataLoader, CalibrationDataset, collect_activation_stats(), export_onnx_int8(), measure_drop(), prepare_qat(), Module, Tensor (+9 more)

### Community 3 - "Training & Latency Harness"
Cohesion: 0.15
Nodes (12): Dataset, main(), measure_onnx(), measure_pytorch(), Module, collate_fn(), main(), Simple folder dataset for research.     Expects:       root/         images/ (+4 more)

### Community 4 - "Paper Submission & Supplementary"
Cohesion: 0.12
Nodes (19): Camera-Ready Checklist, Final Submission Package, Formatting Requirements, Mamba-YOLO, MambaOut, Paper Content Checklist, Reproducibility Requirements, Broader Impact and Limitations (+11 more)

### Community 5 - "Detector Model Assembly"
Cohesion: 0.16
Nodes (9): ConvBNAct, Detect, Mamba3Yolo, Tensor, For XAI / Grad-CAM hooks., Spatial Pyramid Pooling - Fast (Ultralytics style)., Simplified decoupled detection head (box + cls)., Full Mamba3Yolo detector (Tiny scale by default).      Can be used stand-alone (+1 more)

### Community 6 - "Grad-CAM++ XAI"
Cohesion: 0.15
Nodes (13): load_image(), main(), ndarray, Tensor, find_target_layers(), GradCAMPlusPlus, overlay_heatmap(), Module (+5 more)

### Community 7 - "COCO mAP Evaluation"
Cohesion: 0.17
Nodes (9): COCOEvaluator, non_max_suppression(), Tensor, Real COCO-style mAP evaluator for Mamba3Yolo.  Uses torchmetrics (preferred on, Convert normalized YOLO (x_c,y_c,w,h) to absolute xyxy., Simplified NMS for a single image prediction tensor of shape (C, H, W)     or a, Lightweight COCO mAP evaluator compatible with YOLO outputs.          Predicti, preds: list of length B, each (N_i, 6) = [x1,y1,x2,y2,conf,cls] (absolute xyxy) (+1 more)

### Community 8 - "Pure-PyTorch mAP Evaluator"
Cohesion: 0.21
Nodes (13): box_iou(), compute_ap(), decode_preds(), map50(), ndarray, Tensor, Simple pure-PyTorch mAP@0.5 evaluator for Mamba3Yolo.  Does not require pycoco, normalized cxcywh → absolute xyxy (+5 more)

### Community 9 - "Architecture Diagram"
Cohesion: 0.27
Nodes (11): Decoupled Detect Head (Box + Cls), LSBlock (Local Spatial), Mamba3ODSSBlock, Mamba3SS2D (4-dir + Complex + MIMO), Mamba3Yolo Architecture Diagram, Path Aggregation Neck (PAFPN), RGBlock (Gated Residual), SPPF (+3 more)

### Community 10 - "Detection Datasets"
Cohesion: 0.22
Nodes (9): Blood Cell Detection Domain, Brain Tumor Detection Domain, Medical Multi-Domain Detection Config, Polyp Detection Domain, MS COCO 2017, Recommended Datasets for Paper, VisDrone 2019, Datasets for Paper (LaTeX copy) (+1 more)

### Community 11 - "YOLO Loss Function"
Cohesion: 0.28
Nodes (6): bbox_iou(), Tensor, Real YOLO-style detection loss for Mamba3Yolo.  Simplified but usable: - DFL, IoU / CIoU between box1 (N,4) and box2 (M,4) or broadcast., Practical YOLO loss for the multi-scale raw outputs of Mamba3Yolo.      preds:, YOLOLoss

### Community 12 - "Training Curves Diagnostics"
Cohesion: 0.36
Nodes (8): Rapid Convergence Insight, Cosine LR Decay Schedule, Epoch (Training Schedule, 50 epochs), Training Curves Figure (medical_0713_1230), Learning Rate (LR), Mamba3Yolo Medical Training Run (0713_1230), Train Loss, Val Proxy

## Knowledge Gaps
- **25 isolated node(s):** `train.py trainer`, `measure_latency.py`, `generate_xai_figures.py`, `coco_eval.py`, `gradcam.py (Grad-CAM++ + SSM saliency)` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Training & Latency Harness` to `Post-Training Quantization`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `build_mamba3yolo()` connect `Training & Latency Harness` to `ODSS Block Implementation`, `Detector Model Assembly`, `Grad-CAM++ XAI`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `Mamba3Yolo Supplementary Material` connect `Paper Submission & Supplementary` to `Architecture Diagram`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `Mamba3ODSSBlock` (e.g. with `ConvBNAct` and `Detect`) actually correct?**
  _`Mamba3ODSSBlock` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Mamba3Yolo` (e.g. with `SimpleDetectionLoss` and `YoloFolderDataset`) actually correct?**
  _`Mamba3Yolo` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Simple folder dataset for research.     Expects:       root/         images/`, `Mamba3ODSSBlock - NaN-safe + permanent fallback version`, `Real YOLO-style detection loss for Mamba3Yolo.  Simplified but usable: - DFL` to the rest of the system?**
  _59 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Mamba-3 Architecture Concepts` be split into smaller, more focused modules?**
  _Cohesion score 0.08403361344537816 - nodes in this community are weakly interconnected._