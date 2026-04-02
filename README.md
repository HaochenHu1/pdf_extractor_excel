# PDF Table Extractor (PDF → Excel)

`pdf_table_extractor.py` is a command-line utility that extracts tables from PDF documents and writes them to an Excel workbook (one sheet per table, plus a summary sheet).

---

## Features

- Multi-backend extraction:
  - **Camelot** for text-based PDFs
  - **pdfplumber** as a complementary text extractor
  - **img2table + OCR** for scanned PDFs
- Automatic PDF type detection (text vs scanned)
- Optional OCR auto-tuning for noisy Chinese scans
- Adaptive OCR fallback path when standard extraction fails
- Dedicated local pipeline mode for Chinese scanned PDFs (`scanned_cn_local`)
- Table cleanup, deduplication, and summary metadata export

---

## Repository Contents

- `pdf_table_extractor.py` — main extraction script
- `requirements.txt` — Python dependencies

---

## Installation

### 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2) OCR dependencies (for scanned PDFs)

`requirements.txt` includes Python OCR packages (`img2table`, `paddleocr`).

You must still install system OCR dependencies for Tesseract path:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim
```

```bash
# macOS
brew install tesseract tesseract-lang
```

Detailed setup instructions for both supported local OCR paths are in:

- `docs/local_ocr_setup.md`

---

## Usage

### Basic

```bash
python pdf_table_extractor.py input.pdf
```

Default output:

```text
input_tables.xlsx
```

### Common examples

```bash
# Custom output path
python pdf_table_extractor.py input.pdf -o output.xlsx

# Select pages
python pdf_table_extractor.py input.pdf --pages 1-3,5

# Force a backend
python pdf_table_extractor.py input.pdf --mode camelot
python pdf_table_extractor.py input.pdf --mode pdfplumber
python pdf_table_extractor.py input.pdf --mode img2table
python pdf_table_extractor.py input.pdf --mode scanned_cn_local

# Verbose logging
python pdf_table_extractor.py input.pdf --verbose
```

---

## OCR / Scanned PDF Guidance

For difficult scanned PDFs, especially Chinese invoices/statements, start with:

```bash
python pdf_table_extractor.py input.pdf \
  --mode img2table \
  --ocr-lang "chi_sim+eng" \
  --ocr-lang-auto \
  --img2table-min-confidence 30 \
  --verbose
```

For heavily degraded Chinese scans, use the dedicated local pipeline:

```bash
python pdf_table_extractor.py input.pdf \
  --mode scanned_cn_local \
  --ocr-lang "chi_sim+eng" \
  --ocr-lang-auto \
  --scan-dpi 300 \
  --img2table-min-confidence 30 \
  --verbose
```

### Notes

- `--ocr-lang-auto` applies OCR tuning heuristics for Chinese documents.
- Lower `--img2table-min-confidence` can improve recall on noisy scans.
- Use `--borderless` only when tables do not have clear ruling lines.

---

## Output Format

Generated workbook contains:

- `Table_001`, `Table_002`, ... — extracted tables
- `_summary` — extraction metadata:
  - source page
  - extraction engine
  - score
  - row/column counts
  - table title

---

## Troubleshooting

- If no tables are extracted from scanned PDFs:
  - verify OCR dependencies with:
    - `python tools/check_ocr_env.py`
  - ensure OCR dependencies are installed
  - retry with `--mode img2table --ocr-lang-auto --verbose`
- If output is fragmented:
  - lower OCR confidence threshold
  - ensure Paddle OCR dependencies are installed (`img2table[paddle]`)
  - constrain processing with `--pages` to isolate problematic sections

## Evaluation (baseline vs new local scanned-CN mode)

1. Put representative scanned Chinese PDFs in `eval_data/scanned_cn/`.
2. Run:

```bash
python tools/evaluate_scanned_cn_pipeline.py --dataset-dir eval_data/scanned_cn --out-dir eval_results
```

Generated artifacts:
- `eval_results/baseline_summary.json`
- `eval_results/local_cn_summary.json`
- `eval_results/comparison.md`
