from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd


def run_extractor(script: Path, pdf_path: Path, out_xlsx: Path, mode: str) -> Dict[str, object]:
    cmd = [
        "python",
        str(script),
        str(pdf_path),
        "--mode",
        mode,
        "--ocr-lang",
        "chi_sim+eng",
        "--ocr-lang-auto",
        "--img2table-min-confidence",
        "30",
        "--verbose",
        "-o",
        str(out_xlsx),
    ]
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "seconds": round(elapsed, 3),
        "stdout": proc.stdout[-1200:],
        "stderr": proc.stderr[-1200:],
    }


def read_summary_metrics(xlsx_path: Path) -> Dict[str, object]:
    if not xlsx_path.exists():
        return {"tables": 0, "avg_score": 0.0, "avg_rows": 0.0, "avg_cols": 0.0}
    summary = pd.read_excel(xlsx_path, sheet_name="_summary")
    if summary.empty:
        return {"tables": 0, "avg_score": 0.0, "avg_rows": 0.0, "avg_cols": 0.0}
    return {
        "tables": int(summary.shape[0]),
        "avg_score": float(summary["score"].mean()),
        "avg_rows": float(summary["rows"].mean()),
        "avg_cols": float(summary["cols"].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs scanned_cn_local mode.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("eval_data/scanned_cn"))
    parser.add_argument("--script", type=Path, default=Path("pdf_table_extractor.py"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results"))
    args = parser.parse_args()

    pdfs = sorted(args.dataset_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.dataset_dir}. Add test files and rerun.")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows: List[Dict[str, object]] = []
    improved_rows: List[Dict[str, object]] = []

    for pdf in pdfs:
        stem = pdf.stem
        base_out = args.out_dir / f"{stem}_baseline.xlsx"
        new_out = args.out_dir / f"{stem}_local_cn.xlsx"

        base_exec = run_extractor(args.script, pdf, base_out, mode="img2table")
        new_exec = run_extractor(args.script, pdf, new_out, mode="scanned_cn_local")
        base_metrics = read_summary_metrics(base_out)
        new_metrics = read_summary_metrics(new_out)

        baseline_rows.append({"pdf": pdf.name, **base_exec, **base_metrics})
        improved_rows.append({"pdf": pdf.name, **new_exec, **new_metrics})

    baseline = pd.DataFrame(baseline_rows)
    improved = pd.DataFrame(improved_rows)

    baseline_path = args.out_dir / "baseline_summary.json"
    improved_path = args.out_dir / "local_cn_summary.json"
    compare_path = args.out_dir / "comparison.md"

    baseline.to_json(baseline_path, orient="records", force_ascii=False, indent=2)
    improved.to_json(improved_path, orient="records", force_ascii=False, indent=2)

    merged = baseline.merge(improved, on="pdf", suffixes=("_baseline", "_local_cn"))
    with compare_path.open("w", encoding="utf-8") as f:
        f.write("# Baseline vs Local Chinese Scanned Pipeline\n\n")
        f.write("| PDF | tables (base) | tables (local) | avg_score (base) | avg_score (local) | sec (base) | sec (local) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, row in merged.iterrows():
            f.write(
                f"| {row['pdf']} | {int(row['tables_baseline'])} | {int(row['tables_local_cn'])} | "
                f"{row['avg_score_baseline']:.3f} | {row['avg_score_local_cn']:.3f} | "
                f"{row['seconds_baseline']:.2f} | {row['seconds_local_cn']:.2f} |\n"
            )

    print(f"Wrote: {baseline_path}")
    print(f"Wrote: {improved_path}")
    print(f"Wrote: {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
