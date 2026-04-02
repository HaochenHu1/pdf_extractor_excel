# Local OCR Setup for Scanned Chinese Bills

This project supports **two local OCR paths** for scanned Chinese PDF bills.

## Path A (recommended baseline): Tesseract + Chinese language pack

### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim
python -m pip install -r requirements.txt
```

### macOS (Homebrew)
```bash
brew install tesseract tesseract-lang
python -m pip install -r requirements.txt
```

## Path B: PaddleOCR (Python OCR backend)

```bash
python -m pip install -r requirements.txt
```

> `paddleocr` is listed in `requirements.txt` and is used as a Python-only OCR path when available.

---

## Verify environment before extraction

Run:
```bash
python tools/check_ocr_env.py
```

Expected checks:
- `img2table` importable
- `paddleocr` importable (Path B)
- `tesseract` binary available (Path A)
- Chinese OCR language support (`chi_sim`) for tesseract

---

## Rerun bill extraction locally

```bash
python tools/extract_electricity_bill_cn.py \
  training/training1.pdf \
  --target-json tools/manual_target_electricity_bill_cn.json \
  --out-dir bill_debug_outputs
```

If OCR dependencies are missing, the extractor now fails fast with a clear dependency message.
