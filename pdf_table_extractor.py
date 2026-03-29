from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import fitz  # PyMuPDF


@dataclass
class ExtractedTable:
    df: pd.DataFrame
    page: int
    engine: str
    score: float
    title: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract tables from a PDF and write them to an Excel workbook."
    )
    parser.add_argument("input_pdf", type=Path, help="Path to the input PDF")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output .xlsx file. Defaults to <input_stem>_tables.xlsx",
    )
    parser.add_argument(
        "--pages",
        default="all",
        help='Pages to process. Examples: "all", "1", "1,3,5", "2-6"',
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "camelot", "pdfplumber", "img2table"],
        default="auto",
        help="Extraction backend selection",
    )
    parser.add_argument(
        "--prefer",
        choices=["stream", "lattice", "both"],
        default="both",
        help="Camelot extraction style preference for text-based PDFs",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=2,
        help="Minimum number of rows a table must have after cleanup",
    )
    parser.add_argument(
        "--min-cols",
        type=int,
        default=2,
        help="Minimum number of columns a table must have after cleanup",
    )
    parser.add_argument(
        "--min-filled-ratio",
        type=float,
        default=0.15,
        help="Minimum non-empty cell ratio required to keep a table",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=50.0,
        help="Minimum Camelot accuracy for keeping a table when available",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="Tesseract OCR language for img2table fallback",
    )
    parser.add_argument(
        "--borderless",
        action="store_true",
        help="Enable borderless table extraction for img2table",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extraction progress",
    )
    return parser.parse_args()


def log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def expand_page_ranges(pages_spec: str, max_pages: int) -> List[int]:
    if pages_spec.lower() == "all":
        return list(range(1, max_pages + 1))

    pages: set[int] = set()
    chunks = [chunk.strip() for chunk in pages_spec.split(",") if chunk.strip()]
    pattern = re.compile(r"^(\d+)(?:-(\d+))?$")

    for chunk in chunks:
        match = pattern.match(chunk)
        if not match:
            raise ValueError(f"Invalid page spec: {chunk!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < 1 or start > end:
            raise ValueError(f"Invalid page range: {chunk!r}")
        if start > max_pages:
            continue
        end = min(end, max_pages)
        pages.update(range(start, end + 1))

    return sorted(pages)


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.map(normalize_cell)

    # Drop fully empty rows and columns.
    non_empty_row_mask = cleaned.apply(
        lambda row: any(normalize_cell(v) != "" for v in row), axis=1
    )
    cleaned = cleaned.loc[non_empty_row_mask]

    if cleaned.empty:
        return cleaned

    non_empty_col_mask = [
        any(normalize_cell(v) != "" for v in cleaned[col]) for col in cleaned.columns
    ]
    cleaned = cleaned.loc[:, non_empty_col_mask]

    cleaned = cleaned.reset_index(drop=True)
    cleaned.columns = [f"col_{i+1}" for i in range(cleaned.shape[1])]
    return cleaned


def dataframe_filled_ratio(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    total = df.shape[0] * df.shape[1]
    if total == 0:
        return 0.0
    filled = int((df != "").sum().sum())
    return filled / total


def looks_like_table(
    df: pd.DataFrame,
    min_rows: int,
    min_cols: int,
    min_filled_ratio: float,
) -> bool:
    if df.empty:
        return False
    if df.shape[0] < min_rows or df.shape[1] < min_cols:
        return False
    if dataframe_filled_ratio(df) < min_filled_ratio:
        return False
    return True


def dataframe_signature(df: pd.DataFrame) -> Tuple[int, int, Tuple[Tuple[str, ...], ...]]:
    rows: List[Tuple[str, ...]] = []
    for row in df.itertuples(index=False, name=None):
        rows.append(tuple(normalize_cell(v) for v in row))
    return df.shape[0], df.shape[1], tuple(rows)


def deduplicate_tables(tables: Sequence[ExtractedTable]) -> List[ExtractedTable]:
    best_by_signature: Dict[Tuple[int, int, Tuple[Tuple[str, ...], ...]], ExtractedTable] = {}
    for table in tables:
        sig = dataframe_signature(table.df)
        current = best_by_signature.get(sig)
        if current is None or table.score > current.score:
            best_by_signature[sig] = table
    deduped = list(best_by_signature.values())
    deduped.sort(key=lambda t: (t.page, t.engine, -t.score))
    return deduped


def detect_pdf_kind(input_pdf: Path, sample_pages: int = 3) -> str:
    doc = fitz.open(input_pdf)
    total_chars = 0
    checked = min(sample_pages, len(doc))
    for idx in range(checked):
        text = doc[idx].get_text("text") or ""
        total_chars += len(text.strip())
    doc.close()
    if total_chars >= 40:
        return "text"
    return "scanned"


def extract_with_camelot(
    input_pdf: Path,
    pages: List[int],
    prefer: str,
    accuracy_threshold: float,
    min_rows: int,
    min_cols: int,
    min_filled_ratio: float,
    verbose: bool,
) -> List[ExtractedTable]:
    try:
        import camelot
    except ImportError:
        return []

    page_spec = ",".join(str(p) for p in pages)
    flavors = ["stream", "lattice"] if prefer == "both" else [prefer]
    extracted: List[ExtractedTable] = []

    for flavor in flavors:
        try:
            log(f"Trying Camelot ({flavor}) on pages {page_spec}", verbose)
            kwargs = {
                "filepath": str(input_pdf),
                "pages": page_spec,
                "flavor": flavor,
                "suppress_stdout": True,
            }
            if flavor == "stream":
                kwargs.update({"row_tol": 10})
            tables = camelot.read_pdf(**kwargs)
        except Exception as exc:
            log(f"Camelot ({flavor}) failed: {exc}", verbose)
            continue

        for idx, table in enumerate(tables):
            try:
                df = clean_dataframe(table.df)
                report = getattr(table, "parsing_report", {}) or {}
                accuracy = float(report.get("accuracy", 100.0))
                page_num = int(report.get("page", pages[0] if pages else 1))
                score = accuracy / 100.0
                if accuracy < accuracy_threshold:
                    continue
                if not looks_like_table(df, min_rows, min_cols, min_filled_ratio):
                    continue
                extracted.append(
                    ExtractedTable(
                        df=df,
                        page=page_num,
                        engine=f"camelot_{flavor}",
                        score=score,
                        title=f"Camelot {flavor} table {idx + 1}",
                    )
                )
            except Exception as exc:
                log(f"Skipping Camelot table due to error: {exc}", verbose)
    return extracted


PDFPLUMBER_SETTINGS: List[Dict[str, object]] = [
    {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
    },
    {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
    },
]


def extract_with_pdfplumber(
    input_pdf: Path,
    pages: List[int],
    min_rows: int,
    min_cols: int,
    min_filled_ratio: float,
    verbose: bool,
) -> List[ExtractedTable]:
    try:
        import pdfplumber
    except ImportError:
        return []

    extracted: List[ExtractedTable] = []
    with pdfplumber.open(str(input_pdf)) as pdf:
        for page_num in pages:
            page = pdf.pages[page_num - 1]
            for setting_idx, table_settings in enumerate(PDFPLUMBER_SETTINGS, start=1):
                try:
                    raw_tables = page.extract_tables(table_settings=table_settings)
                except Exception as exc:
                    log(
                        f"pdfplumber failed on page {page_num} with setting {setting_idx}: {exc}",
                        verbose,
                    )
                    continue
                for idx, raw_table in enumerate(raw_tables):
                    try:
                        df = clean_dataframe(pd.DataFrame(raw_table))
                        if not looks_like_table(df, min_rows, min_cols, min_filled_ratio):
                            continue
                        score = dataframe_filled_ratio(df)
                        extracted.append(
                            ExtractedTable(
                                df=df,
                                page=page_num,
                                engine=f"pdfplumber_s{setting_idx}",
                                score=score,
                                title=f"pdfplumber setting {setting_idx} table {idx + 1}",
                            )
                        )
                    except Exception as exc:
                        log(f"Skipping pdfplumber table due to error: {exc}", verbose)
    return extracted


def extract_with_img2table(
    input_pdf: Path,
    pages: List[int],
    ocr_lang: str,
    borderless: bool,
    min_rows: int,
    min_cols: int,
    min_filled_ratio: float,
    verbose: bool,
) -> List[ExtractedTable]:
    try:
        from img2table.document import PDF as Img2TablePDF
    except ImportError:
        log("img2table is not installed. Skipping OCR fallback.", verbose)
        return []

    ocr = None
    try:
        from img2table.ocr import TesseractOCR

        ocr = TesseractOCR(n_threads=1, lang=ocr_lang)
    except Exception as exc:
        log(f"Tesseract OCR is unavailable: {exc}", verbose)

    try:
        pdf = Img2TablePDF(src=str(input_pdf), pages=[p - 1 for p in pages], pdf_text_extraction=True)
        tables_by_page = pdf.extract_tables(
            ocr=ocr,
            implicit_rows=False,
            implicit_columns=False,
            borderless_tables=borderless,
            min_confidence=50,
        )
    except Exception as exc:
        log(f"img2table failed: {exc}", verbose)
        return []

    extracted: List[ExtractedTable] = []
    for zero_based_page, tables in tables_by_page.items():
        page_num = zero_based_page + 1
        for idx, table in enumerate(tables):
            try:
                raw_df = getattr(table, "df", None)
                if raw_df is None:
                    continue
                df = clean_dataframe(pd.DataFrame(raw_df))
                if not looks_like_table(df, min_rows, min_cols, min_filled_ratio):
                    continue
                score = dataframe_filled_ratio(df)
                extracted.append(
                    ExtractedTable(
                        df=df,
                        page=page_num,
                        engine="img2table",
                        score=score,
                        title=f"img2table table {idx + 1}",
                    )
                )
            except Exception as exc:
                log(f"Skipping img2table table due to error: {exc}", verbose)
    return extracted


def write_excel(output_path: Path, tables: Sequence[ExtractedTable]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_rows = []
        for idx, table in enumerate(tables, start=1):
            sheet_name = f"Table_{idx:03d}"
            table.df.to_excel(writer, index=False, sheet_name=sheet_name)
            summary_rows.append(
                {
                    "sheet_name": sheet_name,
                    "page": table.page,
                    "engine": table.engine,
                    "score": round(table.score, 4),
                    "rows": table.df.shape[0],
                    "cols": table.df.shape[1],
                    "title": table.title or "",
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, index=False, sheet_name="_summary")

        workbook = writer.book
        for sheet in workbook.worksheets:
            for column_cells in sheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
                max_len = max((len(v) for v in values), default=10)
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 40)
            sheet.freeze_panes = "A2"


def main() -> int:
    args = parse_args()
    input_pdf: Path = args.input_pdf
    if not input_pdf.exists():
        print(f"Input PDF not found: {input_pdf}", file=sys.stderr)
        return 1

    if args.output is None:
        output_path = input_pdf.with_name(f"{input_pdf.stem}_tables.xlsx")
    else:
        output_path = args.output

    try:
        doc = fitz.open(str(input_pdf))
        max_pages = len(doc)
        doc.close()
        pages = expand_page_ranges(args.pages, max_pages)
    except Exception as exc:
        print(f"Failed to read PDF metadata: {exc}", file=sys.stderr)
        return 1

    if not pages:
        print("No valid pages selected.", file=sys.stderr)
        return 1

    pdf_kind = detect_pdf_kind(input_pdf)
    log(f"Detected PDF type: {pdf_kind}", args.verbose)

    extracted: List[ExtractedTable] = []

    if args.mode == "camelot":
        extracted.extend(
            extract_with_camelot(
                input_pdf,
                pages,
                args.prefer,
                args.accuracy_threshold,
                args.min_rows,
                args.min_cols,
                args.min_filled_ratio,
                args.verbose,
            )
        )
    elif args.mode == "pdfplumber":
        extracted.extend(
            extract_with_pdfplumber(
                input_pdf,
                pages,
                args.min_rows,
                args.min_cols,
                args.min_filled_ratio,
                args.verbose,
            )
        )
    elif args.mode == "img2table":
        extracted.extend(
            extract_with_img2table(
                input_pdf,
                pages,
                args.ocr_lang,
                args.borderless,
                args.min_rows,
                args.min_cols,
                args.min_filled_ratio,
                args.verbose,
            )
        )
    else:
        if pdf_kind == "text":
            extracted.extend(
                extract_with_camelot(
                    input_pdf,
                    pages,
                    args.prefer,
                    args.accuracy_threshold,
                    args.min_rows,
                    args.min_cols,
                    args.min_filled_ratio,
                    args.verbose,
                )
            )
            extracted.extend(
                extract_with_pdfplumber(
                    input_pdf,
                    pages,
                    args.min_rows,
                    args.min_cols,
                    args.min_filled_ratio,
                    args.verbose,
                )
            )
        else:
            extracted.extend(
                extract_with_img2table(
                    input_pdf,
                    pages,
                    args.ocr_lang,
                    args.borderless,
                    args.min_rows,
                    args.min_cols,
                    args.min_filled_ratio,
                    args.verbose,
                )
            )
            extracted.extend(
                extract_with_pdfplumber(
                    input_pdf,
                    pages,
                    args.min_rows,
                    args.min_cols,
                    args.min_filled_ratio,
                    args.verbose,
                )
            )

    extracted = deduplicate_tables(extracted)

    if not extracted:
        print(
            "No tables were extracted. If the PDF is scanned, install img2table and Tesseract OCR, then retry with --mode img2table.",
            file=sys.stderr,
        )
        return 2

    write_excel(output_path, extracted)
    print(f"Saved {len(extracted)} table(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
