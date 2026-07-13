# Overleaf Instructions for Mamba3Yolo Paper

## Files to upload to a new Overleaf project

```
main.tex
references.bib
(optional) figures/   ← put your training_curves.png, xai overlays, architecture diagram here
```

## How to compile

1. Create a new Blank Project on Overleaf.
2. Upload `main.tex` and `references.bib`.
3. (Recommended) Also upload the official `cvpr.sty` / `cvpr.bst` if you want the exact CVPR look:
   - Download from https://github.com/cvpr-org/author-kit or the CVPR 2026 author kit.
   - Or simply change the documentclass line to a plain `article` + geometry (already commented in main.tex).
4. Set the compiler to **pdfLaTeX** (Menu → Compiler).
5. Click Recompile. It should produce a clean 2-column paper.

## Customizing

- Change title, authors, emails in the `\title` and `\author` blocks.
- Replace the placeholder numbers in Tables 1–4 with your real Kaggle/ experiment numbers.
- Add `\includegraphics` statements for architecture diagram and XAI figures (create a `figures/` folder).
- Update the GitHub URL and any camera-ready statements.

## Section map

| Section              | Source material                          |
|----------------------|------------------------------------------|
| Abstract + Intro     | Written in main.tex                      |
| Related Work         | docs/latex/related_work_and_results.tex  |
| Methods              | docs/METHODS_SECTION.md (ported)         |
| Experiments/Results  | docs/latex/related_work_and_results.tex  |
| Conclusion           | Written in main.tex                      |

You can also `\input{related_work_and_results.tex}` if you prefer to keep the files separate.
