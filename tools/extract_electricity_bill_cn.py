from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import fitz
import numpy as np


def normalize_ocr_text(text: str) -> str:
    t = text or ""
    t = t.replace("\u3000", " ")
    t = t.replace("：", ":").replace("，", ",").replace("。", ".")
    t = re.sub(r"(?<=\d)\s*[\.]\s*(?=\d)", ".", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d)", "", t)
    t = re.sub(r"(?<=\d)\s+(?=[\u4e00-\u9fff])", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def merge_region_text(*regions: str) -> str:
    return "\n".join([normalize_ocr_text(r) for r in regions if r])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Region-first extraction for scanned Chinese electricity bill PDFs")
    parser.add_argument("input_pdf", type=Path, help="Input PDF path")
    parser.add_argument("--page", type=int, default=1, help="1-based page index")
    parser.add_argument("--target-json", type=Path, required=True, help="Manual target JSON path")
    parser.add_argument("--out-dir", type=Path, default=Path("bill_debug_outputs"), help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI")
    return parser.parse_args()


def render_page(input_pdf: Path, page_num: int, dpi: int) -> np.ndarray:
    doc = fitz.open(str(input_pdf))
    page = doc[page_num - 1]
    zoom = max(1.0, dpi / 72.0)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def suppress_blue_stamp(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower1 = np.array([85, 40, 40], dtype=np.uint8)
    upper1 = np.array([145, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return cleaned


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoise = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    bw = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)
    return bw


def crop_rel(image: np.ndarray, rect: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = int(rect[0] * w)
    y0 = int(rect[1] * h)
    x1 = int(rect[2] * w)
    y1 = int(rect[3] * h)
    return image[y0:y1, x0:x1]


def region_layout(page_image: np.ndarray) -> Dict[str, np.ndarray]:
    left_main = crop_rel(page_image, (0.02, 0.03, 0.70, 0.95))
    regions = {
        "left_main_region": left_main,
        "top_summary_region": crop_rel(left_main, (0.0, 0.0, 1.0, 0.11)),
        "usage_detail_region": crop_rel(left_main, (0.0, 0.11, 1.0, 0.27)),
        "charge_detail_region": crop_rel(left_main, (0.0, 0.27, 1.0, 0.52)),
        "cost_breakdown_region": crop_rel(left_main, (0.48, 0.33, 0.99, 0.52)),
        "totals_region": crop_rel(left_main, (0.0, 0.52, 1.0, 0.59)),
        "bottom_metadata_region": crop_rel(left_main, (0.0, 0.81, 1.0, 0.95)),
    }
    return regions


def run_tesseract_cli(image_path: Path, lang: str = "chi_sim+eng") -> str:
    if shutil.which("tesseract") is None:
        return ""
    cmd = ["tesseract", str(image_path), "stdout", "-l", lang, "--psm", "6"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def run_paddleocr_python(image_path: Path) -> str:
    if importlib.util.find_spec("paddleocr") is None:
        return ""
    from paddleocr import PaddleOCR  # type: ignore

    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    results = ocr.ocr(str(image_path), cls=True)
    lines: List[str] = []
    for entry in results or []:
        for row in entry or []:
            txt = row[1][0] if len(row) > 1 and row[1] else ""
            if txt:
                lines.append(str(txt))
    return "\n".join(lines).strip()


def run_region_ocr(image_path: Path) -> Tuple[str, str]:
    if shutil.which("tesseract") is not None:
        return "tesseract_cli", run_tesseract_cli(image_path)
    if importlib.util.find_spec("paddleocr") is not None:
        return "paddleocr_python", run_paddleocr_python(image_path)
    return "none", ""


def tesseract_langs() -> List[str]:
    if shutil.which("tesseract") is None:
        return []
    proc = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("list of available languages"):
        lines = lines[1:]
    return lines


def require_ocr_backend() -> None:
    has_tesseract = shutil.which("tesseract") is not None
    has_paddle = importlib.util.find_spec("paddleocr") is not None
    has_img2table = importlib.util.find_spec("img2table") is not None
    langs = set(tesseract_langs())
    has_chi_sim = "chi_sim" in langs

    if has_paddle:
        return
    if has_tesseract and has_chi_sim:
        return

    missing: List[str] = []
    if not has_img2table:
        missing.append("python package: img2table")
    if not has_paddle:
        missing.append("python package: paddleocr (or install to enable Path B)")
    if not has_tesseract:
        missing.append("system binary: tesseract")
    elif not has_chi_sim:
        missing.append("tesseract language data: chi_sim")

    msg = (
        "OCR dependency check failed. Missing: "
        + ", ".join(missing)
        + ".\nSetup options:\n"
        + "  Path A: install tesseract + chi_sim and keep img2table installed.\n"
        + "  Path B: install paddleocr Python package.\n"
        + "Then run: python tools/check_ocr_env.py"
    )
    raise RuntimeError(msg)


def extract_number(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def extract_date(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text)
    if not m:
        return None
    y, mm, dd = m.groups()
    return f"{y}-{mm}-{dd}"


def reconstruct(ocr_by_region: Dict[str, str]) -> Dict[str, Any]:
    left = normalize_ocr_text(ocr_by_region.get("left_main_region", ""))
    top = normalize_ocr_text(ocr_by_region.get("top_summary_region", ""))
    totals = normalize_ocr_text(ocr_by_region.get("totals_region", ""))
    meta = normalize_ocr_text(ocr_by_region.get("bottom_metadata_region", ""))

    primary = merge_region_text(left, top, totals, meta)

    period_match = re.search(
        r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})\D{0,8}(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})",
        primary,
    )
    period = None
    if period_match:
        y1, m1, d1, y2, m2, d2 = period_match.groups()
        period = f"{y1}-{int(m1):02d}-{int(d1):02d} 至 {y2}-{int(m2):02d}-{int(d2):02d}"

    usage_rows: List[Dict[str, Any]] = []
    usage_candidates = re.findall(r"(\d{4,6})\s+(\d{4,6})\s+([12])\s+(\d{2,4})", primary)
    seen_usage: set[Tuple[str, str, str, str]] = set()
    for idx, row in enumerate(usage_candidates[:2]):
        if row in seen_usage:
            continue
        seen_usage.add(row)
        prev, curr, ratio, kwh = row
        item = "正向有功（平）" if idx == 0 else "正向有功（谷）"
        usage_rows.append(
            {
                "项目": item,
                "上期示数": float(prev),
                "本期示数": float(curr),
                "倍率": float(ratio),
                "计数电量": float(kwh),
            }
        )

    charge_rows: List[Dict[str, Any]] = []
    charge_candidates = re.findall(r"(\d{2,4})\s+(0\.\d{3,4})\s+(\d+\.\d{2})", primary)
    seen_charge: set[Tuple[str, str, str]] = set()
    for idx, row in enumerate(charge_candidates[:2]):
        if row in seen_charge:
            continue
        seen_charge.add(row)
        qty, rate, amount = row
        suffix = "平" if idx == 0 else "谷"
        charge_rows.append(
            {
                "项目": f"一般非工业（非经营性质），单一制不分时，电压220V（{suffix}）电度电费",
                "计费电量": float(qty),
                "计费标准": float(rate),
                "电费": float(amount),
            }
        )

    cost_rows: List[Dict[str, Any]] = []
    for label in ["代理购电电费", "上网环节线损费用", "输配电费", "系统运行费", "政府性基金及附加"]:
        m = re.search(rf"{re.escape(label)}[^0-9]{{0,8}}([0-9]+(?:\.[0-9]+)?)", primary)
        if m:
            cost_rows.append({"子项": label, "金额": float(m.group(1))})

    reconstructed: Dict[str, Any] = {
        "document_type": "electricity_bill_cn",
        "page_count": 1,
        "top_summary": {
            "本期电量": {"value": extract_number(primary, r"本期电量[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)"), "unit": "千瓦时"},
            "本期电费": {"value": extract_number(primary, r"本期电费[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)"), "unit": "元"},
            "账单打印日期": extract_date(primary, r"账单打印日期[^0-9]*(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})"),
            "交费截止日期": extract_date(primary, r"交费截止日期[^0-9]*(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})"),
        },
        "tables": {
            "本期用电明细": {"rows": usage_rows},
            "本期电费明细": {"rows": charge_rows},
            "费用构成": {"rows": cost_rows},
        },
        "summary_totals": {
            "小计": extract_number(primary, r"小计[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)"),
            "本月应付电费": extract_number(primary, r"本月应付电费[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)"),
            "本月实付电费": extract_number(primary, r"本月实付电费[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)"),
        },
        "bottom_metadata": {
            "账单周期": period,
            "户号": re.search(r"户号\s*([0-9]{8,})", primary).group(1) if re.search(r"户号\s*([0-9]{8,})", primary) else None,
            "用电类别": "非工业" if "非工业" in primary else None,
            "电压等级": "交流220V" if ("交流220V" in primary or "220V" in primary) else None,
            "供电服务单位": "浦东供电公司" if "浦东供电公司" in primary else None,
            "供账中心": "东方枢纽" if "东方枢纽" in primary else None,
        },
    }
    return reconstructed


def diff_json(expected: Any, actual: Any, path: str = "$") -> List[Dict[str, Any]]:
    diffs: List[Dict[str, Any]] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected.keys() - actual.keys():
            diffs.append({"path": f"{path}.{key}", "issue": "missing_in_actual", "expected": expected[key], "actual": None})
        for key in actual.keys() - expected.keys():
            diffs.append({"path": f"{path}.{key}", "issue": "extra_in_actual", "expected": None, "actual": actual[key]})
        for key in expected.keys() & actual.keys():
            diffs.extend(diff_json(expected[key], actual[key], f"{path}.{key}"))
        return diffs

    if isinstance(expected, list) and isinstance(actual, list):
        min_len = min(len(expected), len(actual))
        for i in range(min_len):
            diffs.extend(diff_json(expected[i], actual[i], f"{path}[{i}]"))
        if len(expected) > len(actual):
            for i in range(len(actual), len(expected)):
                diffs.append({"path": f"{path}[{i}]", "issue": "missing_list_item", "expected": expected[i], "actual": None})
        elif len(actual) > len(expected):
            for i in range(len(expected), len(actual)):
                diffs.append({"path": f"{path}[{i}]", "issue": "extra_list_item", "expected": None, "actual": actual[i]})
        return diffs

    if expected != actual:
        diffs.append({"path": path, "issue": "value_mismatch", "expected": expected, "actual": actual})
    return diffs


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        require_ocr_backend()
    except RuntimeError as exc:
        print(str(exc))
        return 2

    page_image = render_page(args.input_pdf, args.page, args.dpi)
    blue_suppressed = suppress_blue_stamp(page_image)
    regions = region_layout(blue_suppressed)

    ocr_texts: Dict[str, str] = {}
    ocr_backend = "none"
    for region_name, region_img in regions.items():
        region_dir = args.out_dir / "regions"
        region_dir.mkdir(parents=True, exist_ok=True)
        color_path = region_dir / f"{region_name}.png"
        bw_path = region_dir / f"{region_name}_bw.png"
        cv2.imwrite(str(color_path), region_img)
        bw = preprocess_for_ocr(region_img)
        cv2.imwrite(str(bw_path), bw)
        backend_name, ocr_text = run_region_ocr(bw_path)
        if ocr_backend == "none":
            ocr_backend = backend_name
        ocr_texts[region_name] = ocr_text

    reconstructed = reconstruct(ocr_texts)
    target = json.loads(args.target_json.read_text(encoding="utf-8"))
    mismatches = diff_json(target, reconstructed)

    (args.out_dir / "raw_ocr_by_region.json").write_text(
        json.dumps(ocr_texts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "reconstructed.json").write_text(
        json.dumps(reconstructed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "mismatch_vs_target.json").write_text(
        json.dumps(mismatches, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "input_pdf": str(args.input_pdf),
        "page": args.page,
        "ocr_backend_selected": ocr_backend,
        "tesseract_available": shutil.which("tesseract") is not None,
        "paddleocr_importable": importlib.util.find_spec("paddleocr") is not None,
        "img2table_importable": importlib.util.find_spec("img2table") is not None,
        "region_names": list(regions.keys()),
        "mismatch_count": len(mismatches),
        "diagnosis": (
            "OCR engine unavailable in runtime"
            if (shutil.which("tesseract") is None and importlib.util.find_spec("paddleocr") is None)
            else "OCR executed; inspect mismatch report for reconstruction gaps"
        ),
    }
    (args.out_dir / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
