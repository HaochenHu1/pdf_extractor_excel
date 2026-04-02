# Scanned Chinese PDF Pipeline Audit (Local-First)

## Why the previous pipeline underperformed

The prior flow (`img2table` OCR pass with light heuristics) failed on production-like Chinese scanned PDFs for structural reasons:

1. **No strong image preprocessing stage**
   - No explicit deskew before OCR.
   - Weak denoising/contrast handling for low-quality scans.
   - No adaptive thresholding tuned for uneven background illumination.
2. **Layout complexity not handled**
   - Full-page OCR/table detection on multi-block bills mixes header text, notes, and table body.
   - No region proposal stage to isolate likely table areas.
3. **OCR sensitivity**
   - Chinese scans with stamps/noise require stronger OCR setup and cleaned input images.
   - Character spacing artifacts (`计 量 点`) and fragmented columns degraded reconstruction.
4. **Table reconstruction bottlenecks**
   - Generic extraction can over/under split columns on noisy page-level inputs.
   - Lack of scanned-CN-specific postprocessing path reduced usable output quality.

## What was changed on this branch

New mode: `--mode scanned_cn_local`

Pipeline stages:

1. **Render PDF page to image** (`--scan-dpi`, default 300).
2. **Preprocess image locally (OpenCV)**
   - denoise (`fastNlMeansDenoising`)
   - contrast enhancement (CLAHE)
   - adaptive thresholding
   - deskew using minimum-area rectangle angle estimate
3. **Layout-aware table region proposal**
   - morphology-based horizontal/vertical line extraction
   - contour-based region filtering
   - crop candidate table regions before OCR
4. **OCR + table extraction**
   - Prefer local PaddleOCR if available, else local Tesseract.
   - Extract per region with implicit row/column inference.
5. **Postprocessing**
   - Chinese spacing cleanup and dataframe quality filtering
   - deduplication across regions/pages

This keeps the default implementation **fully local** and does not call external APIs.

## Evaluation setup

Use:

```bash
python tools/evaluate_scanned_cn_pipeline.py --dataset-dir eval_data/scanned_cn --out-dir eval_results
```

This benchmarks:
- baseline: `--mode img2table`
- improved: `--mode scanned_cn_local`

Outputs:
- `eval_results/baseline_summary.json`
- `eval_results/local_cn_summary.json`
- `eval_results/comparison.md`

## Baseline vs new results (current environment)

No scanned PDF dataset was available in this workspace at implementation time, so quantitative benchmark files cannot be generated yet.

**Next step:** place representative files under `eval_data/scanned_cn/*.pdf` and run the evaluation command above.

## Recommendation

For this document family, local preprocessing + region-aware OCR should be attempted first (implemented here).  
A specialized local model should only be considered if evaluation still shows unacceptable reconstruction after this pipeline, especially when OCR text quality is good but structure recovery remains the bottleneck.
