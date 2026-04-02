from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from typing import Any, Dict


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def tesseract_langs() -> list[str]:
    if shutil.which("tesseract") is None:
        return []
    proc = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("list of available languages"):
        lines = lines[1:]
    return lines


def main() -> int:
    langs = tesseract_langs()
    result: Dict[str, Any] = {
        "img2table_importable": has_module("img2table"),
        "paddleocr_importable": has_module("paddleocr"),
        "tesseract_binary": shutil.which("tesseract") or "",
        "tesseract_languages": langs,
        "tesseract_has_chi_sim": "chi_sim" in set(langs),
    }
    result["ocr_ready"] = (
        (result["img2table_importable"] and result["tesseract_binary"] and result["tesseract_has_chi_sim"])
        or result["paddleocr_importable"]
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
