# Mamba3Yolo — Full Camera-Ready Checklist

Use this list before submitting to CVPR / ICCV / ECCV / MICCAI / TPAMI / WACV etc.

---

## 1. Paper Content

### Title & Abstract
- [ ] Title is concise and contains key terms (Mamba-3, real-time, object detection, edge / explainable)
- [ ] Abstract ≤ 150–200 words (or venue limit)
- [ ] Abstract states problem, method, key quantitative result, and implication
- [ ] No “we are the first” claims without careful verification
- [ ] GitHub / project URL is correct and public (or anonymized for double-blind)

### Introduction
- [ ] Motivation is clear in first paragraph
- [ ] Gap is stated explicitly (Mamba-YOLO still uses Mamba-1/2)
- [ ] Contributions are listed as bullet points (3–5 items)
- [ ] All claims later supported by experiments

### Related Work
- [ ] Groups papers by approach (not a laundry list)
- [ ] Closest prior work (Mamba-YOLO, Mamba-3, MambaOut, PTQ4VM, etc.) is discussed
- [ ] Differences / delta from closest work are clear
- [ ] Citations are recent (2024–2026) + foundational

### Methods
- [ ] Architecture is fully described (can be re-implemented from text + figure)
- [ ] Key equation(s) for Mamba-3 recurrence are present
- [ ] Why LSBlock + RGBlock are kept is justified (MambaOut)
- [ ] Training details (optimizer, LR, epochs, augmentations, resolution) are given
- [ ] Any medical multi-domain protocol is described

### Experiments
- [ ] Strong baselines (YOLOv8n, original Mamba-YOLO-T/M, recent hybrids)
- [ ] Ablation table isolates each Mamba-3 component
- [ ] Multi-seed or at least mean±std if claiming statistical significance
- [ ] Latency measured on the same hardware for all models
- [ ] Edge / Jetson numbers included if claiming edge suitability
- [ ] Medical results reported per domain + average
- [ ] Quantization drop is reported (FP32 → INT8 / W8A8)
- [ ] Failure cases or limitations are discussed honestly

### Figures & Tables
- [ ] Architecture diagram is clear, high-resolution, vector (TikZ / PDF)
- [ ] All tables use booktabs style, consistent significant digits
- [ ] Captions are self-contained
- [ ] Grad-CAM / XAI figures show real medical images + overlays
- [ ] Training curves (optional but nice) are clean
- [ ] Color is accessible (or patterns for B&W printing)

### Conclusion
- [ ] Restates main finding without hype
- [ ] Limitations are acknowledged
- [ ] Broader impact / clinical relevance mentioned if applicable

---

## 2. Formatting & Style (Venue Specific)

### CVPR / ICCV / ECCV
- [ ] Official style file used (`cvpr.sty` / author kit)
- [ ] 8 pages max (excluding references) for conference; check current year
- [ ] Two-column, 10 pt, Times
- [ ] Line numbers on for review version
- [ ] Blind submission: no author names, no acknowledgements, no self-identifying GitHub if required
- [ ] Supplementary material uploaded separately if needed

### General
- [ ] All fonts embedded
- [ ] PDF is PDF/A or at least PDF 1.5+
- [ ] No overfull boxes / warnings that affect layout
- [ ] Hyperlinks work (or disabled if required)
- [ ] Page numbers, headers/footers follow venue rules

---

## 3. Reproducibility

- [ ] Code released (or anonymized link for review)
- [ ] Config files (YAML) and training commands documented
- [ ] Random seeds fixed and reported
- [ ] Exact package versions in `requirements.txt` or environment.yml
- [ ] Pretrained weights will be released (or already on HF/GitHub)
- [ ] Dataset preprocessing scripts included (especially medical multi-domain mapping)
- [ ] Evaluation protocol matches official COCO (or stated differences)

---

## 4. Ethics & Broader Impact (if required)

- [ ] Medical data usage complies with licenses / IRB if applicable
- [ ] Potential dual-use or fairness issues briefly discussed
- [ ] No patient-identifying information in figures

---

## 5. Final Submission Package

- [ ] `main.pdf` (camera-ready)
- [ ] Source archive (`.zip` / `.tar.gz`) containing all `.tex`, `.bib`, figures
- [ ] Supplementary PDF (if any)
- [ ] Code / model links active
- [ ] Author response / rebuttal addressed (if post-review)
- [ ] Copyright form signed
- [ ] Final title / abstract match submission system

---

## 6. Pre-Submission Sanity Checks

```bash
# Compile cleanly
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Check for TODOs / placeholders
grep -n "TODO\|FIXME\|XX\|placeholder\|lorem" main.tex

# Page count
pdfinfo main.pdf | grep Pages

# Overfull boxes
grep -i overfull main.log
```

---

## Quick Status for Mamba3Yolo (current)

| Item                        | Status          | Notes |
|----------------------------|-----------------|-------|
| Architecture TikZ          | ✅ Ready        | `docs/latex/figures/architecture.tex` |
| Methods section            | ✅ Ready        | `docs/METHODS_SECTION.md` + main.tex |
| Related Work + Results     | ✅ Ready        | tables with placeholder numbers |
| Camera-ready main.tex      | ✅ Ready        | Overleaf-ready |
| Supplementary PDF          | ✅ Ready        | see below |
| Real COCO numbers          | ⏳ Pending      | replace placeholders after Kaggle runs |
| Medical real mAP           | ⏳ Pending      | after multi-domain training |
| XAI figures on real images | ⏳ Pending      | use `generate_xai_figures.py` |
| Jetson TensorRT numbers    | ⏳ Pending      | use `measure_latency.py` |
| Code public                | ✅ GitHub live  | https://github.com/ShMazumder/Mamba3Yolo |

Replace all placeholder numbers in Tables 1–4 with your experimental results before camera-ready submission.
