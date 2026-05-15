"""Direct PyMuPDF (fitz) extraction.

Walks each page's text dictionary, using the largest span size per line as
the heading signal (ratio vs. the document-wide median of line sizes).
Produces paragraph / heading / image blocks in the same unified schema as
every other engine.

Overlaps with the `basic` engine in intent but reads primitives from a
different library (fitz vs. pdfplumber), which gives slightly different
coverage on some PDFs — useful for comparison.
"""
from __future__ import annotations

import time
import uuid
from collections import Counter
from typing import Any, Dict, List

import fitz  # PyMuPDF

from ..schema import ExtractionMetrics, ExtractionResult, PageMeta
from ..tools.markdown_render import blocks_to_markdown, compute_metrics
from ..tools.pdf_access import get_page_geometry, pdf_meta


def _heading_level(ratio: float) -> int | None:
    """Map font-size ratio over body to a 1..6 heading level, or None."""
    if ratio >= 1.8:
        return 1
    if ratio >= 1.45:
        return 2
    if ratio >= 1.25:
        return 3
    if ratio >= 1.12:
        return 4
    return None


class PyMuPDFExtractor:
    name = "pymupdf"

    async def extract(self, pdf_path: str) -> ExtractionResult:
        t0 = time.time()
        meta = pdf_meta(pdf_path)
        pages_geom = get_page_geometry(pdf_path)

        lines: List[Dict[str, Any]] = []
        image_blocks: List[Dict[str, Any]] = []

        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                pno = i + 1
                page_dict = page.get_text("dict")
                for blk in page_dict.get("blocks", []):
                    # fitz: 0=text, 1=image
                    if blk.get("type") != 0:
                        continue
                    for line in blk.get("lines", []):
                        spans = line.get("spans", []) or []
                        text = "".join(s.get("text", "") for s in spans).strip()
                        if not text:
                            continue
                        size = max((float(s.get("size", 0)) for s in spans), default=0.0)
                        x0s = [s.get("bbox", [0, 0, 0, 0])[0] for s in spans]
                        y0s = [s.get("bbox", [0, 0, 0, 0])[1] for s in spans]
                        x1s = [s.get("bbox", [0, 0, 0, 0])[2] for s in spans]
                        y1s = [s.get("bbox", [0, 0, 0, 0])[3] for s in spans]
                        lines.append({
                            "page": pno,
                            "text": text,
                            "size": size,
                            "bbox": [min(x0s), min(y0s), max(x1s), max(y1s)],
                        })

                for img in page.get_image_info(xrefs=True):
                    bbox = img.get("bbox")
                    if bbox:
                        image_blocks.append({
                            "id": f"mi_{uuid.uuid4().hex[:8]}",
                            "type": "image",
                            "page": pno,
                            "bbox": [float(b) for b in bbox],
                            "caption": None,
                            "description": (
                                f"Image region {img.get('width')}x{img.get('height')}"
                            ),
                        })

        # Body size = mode of rounded line sizes. Median is too noisy when
        # many lines have near-identical sizes with tiny float differences;
        # mode reliably locks onto the dominant body-text size.
        if lines:
            size_counter = Counter(round(ln["size"], 1) for ln in lines)
            body_size = size_counter.most_common(1)[0][0]
        else:
            body_size = 10.0

        blocks: List[Dict[str, Any]] = []
        para_buf: List[Dict[str, Any]] = []

        def flush():
            nonlocal para_buf
            if not para_buf:
                return
            text = " ".join(l["text"] for l in para_buf).strip()
            if text:
                blocks.append({
                    "id": f"mp_{uuid.uuid4().hex[:8]}",
                    "type": "paragraph",
                    "page": para_buf[0]["page"],
                    "bbox": [
                        min(l["bbox"][0] for l in para_buf),
                        min(l["bbox"][1] for l in para_buf),
                        max(l["bbox"][2] for l in para_buf),
                        max(l["bbox"][3] for l in para_buf),
                    ],
                    "text": text,
                })
            para_buf = []

        for ln in lines:
            ratio = ln["size"] / max(body_size, 1.0)
            lvl = _heading_level(ratio)
            if lvl is not None and len(ln["text"].split()) <= 20:
                flush()
                blocks.append({
                    "id": f"mh_{uuid.uuid4().hex[:8]}",
                    "type": "heading",
                    "level": lvl,
                    "page": ln["page"],
                    "bbox": ln["bbox"],
                    "text": ln["text"],
                })
            else:
                if para_buf:
                    prev = para_buf[-1]
                    # Break on page change
                    if prev["page"] != ln["page"]:
                        flush()
                    else:
                        line_h = max(8.0, prev["bbox"][3] - prev["bbox"][1])
                        gap = ln["bbox"][1] - prev["bbox"][3]
                        # Break on a visible gap (blank line between paragraphs)
                        if gap > line_h * 1.2:
                            flush()
                para_buf.append(ln)
        flush()

        blocks.extend(image_blocks)
        blocks.sort(key=lambda b: (
            b.get("page", 0),
            (b.get("bbox") or [0, 0, 0, 0])[1],
            (b.get("bbox") or [0, 0, 0, 0])[0],
        ))

        title = meta.get("title")
        if not title:
            for b in blocks:
                if b.get("type") == "heading" and b.get("level") == 1:
                    title = b.get("text")
                    break

        duration_ms = int((time.time() - t0) * 1000)
        md = blocks_to_markdown(blocks, title=title)
        counts = compute_metrics(blocks)

        return ExtractionResult(
            engine=self.name,
            title=title,
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
