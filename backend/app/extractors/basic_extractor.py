"""Dependency-light baseline: pdfplumber + font heuristics.

Used as a sanity-check engine when the heavy ML engines aren't available,
and as a source of raw primitives for the agentic engine.
"""
from __future__ import annotations

import time
import uuid
from collections import Counter
from typing import Any, Dict, List

from ..schema import ExtractionMetrics, ExtractionResult, PageMeta
from ..tools.markdown_render import blocks_to_markdown, compute_metrics
from ..tools.pdf_access import (
    extract_tables,
    get_images_on_page,
    get_page_geometry,
    get_page_text_with_fonts,
    pdf_meta,
)


def _classify_heading_level(size: float, body_size: float) -> int | None:
    """Return 1..6 if the size clearly indicates a heading; else None."""
    if size <= body_size * 1.05:
        return None
    ratio = size / max(body_size, 1.0)
    if ratio >= 1.8:
        return 1
    if ratio >= 1.45:
        return 2
    if ratio >= 1.25:
        return 3
    if ratio >= 1.12:
        return 4
    return None


class BasicExtractor:
    name = "basic"

    async def extract(self, pdf_path: str) -> ExtractionResult:
        t0 = time.time()
        meta = pdf_meta(pdf_path)
        pages_geom = get_page_geometry(pdf_path)

        # First pass: collect all line sizes so we can infer body text size.
        all_lines: List[Dict[str, Any]] = []
        for pg in pages_geom:
            lines = get_page_text_with_fonts(pdf_path, pg["page"])
            for ln in lines:
                ln["page"] = pg["page"]
            all_lines.extend(lines)

        # Drop sub-point "lines" — pdfplumber reports glyphs from rotated /
        # invisible / artifact text at sizes like 0.5–2pt, and they
        # outnumber the real body text on some PDFs. Anything below a few
        # points is never real prose, just noise that throws off both the
        # body-size estimator and the heading classifier downstream.
        TEXT_SIZE_FLOOR = 5.0
        all_lines = [l for l in all_lines if float(l.get("size", 0)) >= TEXT_SIZE_FLOOR]

        # Body size = mode of rounded line sizes. The median is too noisy
        # when most lines share the same size with sub-point float jitter,
        # so the heading-vs-body classifier was firing on every line whose
        # size was even slightly above the median. Mode picks the dominant
        # body-text size unambiguously.
        if all_lines:
            size_counter = Counter(round(l["size"], 1) for l in all_lines)
            body_size = size_counter.most_common(1)[0][0]
        else:
            body_size = 10.0

        blocks: List[Dict[str, Any]] = []
        para_buf: List[Dict[str, Any]] = []

        def flush_para():
            nonlocal para_buf
            if not para_buf:
                return
            text = " ".join(l["text"] for l in para_buf).strip()
            if not text:
                para_buf = []
                return
            x0 = min(l["x0"] for l in para_buf)
            x1 = max(l["x1"] for l in para_buf)
            top = min(l["top"] for l in para_buf)
            bot = max(l["bottom"] for l in para_buf)
            blocks.append({
                "id": f"p_{uuid.uuid4().hex[:8]}",
                "type": "paragraph",
                "page": para_buf[0]["page"],
                "bbox": [x0, top, x1, bot],
                "text": text,
            })
            para_buf = []

        for ln in all_lines:
            lvl = _classify_heading_level(ln["size"], body_size)
            text_trim = ln["text"].strip()
            # Headings need at least 2 chars and ≤ 20 words. Single-char
            # "lines" are typically rotated-text fragments masquerading as
            # large-font glyphs (arXiv banner, watermarks, etc.).
            if lvl is not None and len(text_trim) >= 2 and len(text_trim.split()) <= 20:
                flush_para()
                blocks.append({
                    "id": f"h_{uuid.uuid4().hex[:8]}",
                    "type": "heading",
                    "level": lvl,
                    "page": ln["page"],
                    "bbox": [ln["x0"], ln["top"], ln["x1"], ln["bottom"]],
                    "text": ln["text"].strip(),
                })
            else:
                if para_buf:
                    prev = para_buf[-1]
                    if prev["page"] != ln["page"]:
                        # New page → start a fresh paragraph.
                        flush_para()
                    else:
                        line_h = max(8.0, prev["bottom"] - prev["top"])
                        gap = ln["top"] - prev["bottom"]
                        # Visible blank line → paragraph break.
                        if gap > line_h * 1.2:
                            flush_para()
                para_buf.append(ln)
        flush_para()

        # Tables
        for pg in pages_geom:
            tables = extract_tables(pdf_path, pg["page"])
            for t in tables:
                rows = t
                blocks.append({
                    "id": f"t_{uuid.uuid4().hex[:8]}",
                    "type": "table",
                    "page": pg["page"],
                    "rows": rows,
                    "row_count": len(rows),
                    "col_count": max((len(r) for r in rows), default=0),
                    "col_headers": rows[0] if rows else [],
                })

        # Images as bbox-only placeholders
        for pg in pages_geom:
            for img in get_images_on_page(pdf_path, pg["page"]):
                blocks.append({
                    "id": f"i_{uuid.uuid4().hex[:8]}",
                    "type": "image",
                    "page": img["page"],
                    "bbox": img["bbox"],
                    "caption": None,
                    "description": f"Image region {img['width']}x{img['height']}",
                })

        blocks.sort(key=lambda b: (b.get("page", 0), (b.get("bbox") or [0, 0])[1]))

        duration_ms = int((time.time() - t0) * 1000)
        md = blocks_to_markdown(blocks, title=meta.get("title"))
        counts = compute_metrics(blocks)

        return ExtractionResult(
            engine=self.name,
            title=meta.get("title"),
            page_count=meta["page_count"],
            pages=[PageMeta(**pg) for pg in pages_geom],
            blocks=blocks,
            markdown=md,
            metrics=ExtractionMetrics(
                duration_ms=duration_ms,
                block_count=len(blocks),
                **counts,
            ),
        )
