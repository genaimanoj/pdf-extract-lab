"""Low-level PDF primitives shared across extractors.

Uses pdfplumber (tables + raw text) and PyMuPDF/fitz (fast page geometry, images).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber


def get_page_count(pdf_path: str) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def get_page_geometry(pdf_path: str) -> List[Dict[str, float]]:
    out = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            r = page.rect
            out.append({"page": i + 1, "width": float(r.width), "height": float(r.height)})
    return out


def get_page_text(pdf_path: str, page: int) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        p = pdf.pages[page - 1]
        return p.extract_text() or ""


def get_page_text_with_fonts(pdf_path: str, page: int) -> List[Dict[str, Any]]:
    """Return per-line text with average font size + dominant font name.

    Useful for heading classification heuristics.
    """
    lines: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        p = pdf.pages[page - 1]
        words = p.extract_words(extra_attrs=["size", "fontname"])
        current: Dict[str, Any] = {}
        current_key: Optional[Tuple[int, str, float]] = None
        for w in words:
            top = int(w["top"])
            fname = w.get("fontname", "")
            fsize = float(w.get("size", 0.0))
            key = (top // 2, fname, round(fsize, 1))
            if key != current_key and current:
                lines.append(current)
                current = {}
            if not current:
                current = {
                    "text": w["text"],
                    "x0": w["x0"], "x1": w["x1"],
                    "top": w["top"], "bottom": w["bottom"],
                    "font": fname, "size": fsize,
                }
            else:
                current["text"] += " " + w["text"]
                current["x1"] = max(current["x1"], w["x1"])
                current["bottom"] = max(current["bottom"], w["bottom"])
            current_key = key
        if current:
            lines.append(current)
    return lines


def extract_tables(pdf_path: str, page: int) -> List[List[List[str]]]:
    """Return tables on a page as list of row-lists, using pdfplumber heuristics."""
    with pdfplumber.open(pdf_path) as pdf:
        p = pdf.pages[page - 1]
        tables = p.extract_tables() or []
        cleaned = []
        for t in tables:
            rows = [[(c or "").strip() for c in row] for row in t if any((c or "").strip() for c in row)]
            if rows:
                cleaned.append(rows)
        return cleaned


def extract_table_bbox(pdf_path: str, page: int, bbox: Tuple[float, float, float, float]) -> Optional[List[List[str]]]:
    with pdfplumber.open(pdf_path) as pdf:
        p = pdf.pages[page - 1]
        crop = p.within_bbox(bbox)
        t = crop.extract_table()
        if not t:
            return None
        return [[(c or "").strip() for c in row] for row in t]


def get_images_on_page(pdf_path: str, page: int) -> List[Dict[str, Any]]:
    out = []
    with fitz.open(pdf_path) as doc:
        p = doc[page - 1]
        for img in p.get_image_info(xrefs=True):
            bbox = img.get("bbox")
            if bbox:
                out.append({
                    "page": page,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "xref": img.get("xref"),
                    "width": img.get("width"),
                    "height": img.get("height"),
                })
    return out


@lru_cache(maxsize=1)
def pdf_meta(pdf_path: str) -> Dict[str, Any]:
    with fitz.open(pdf_path) as doc:
        m = doc.metadata or {}
        return {
            "title": m.get("title") or None,
            "author": m.get("author") or None,
            "page_count": doc.page_count,
        }
