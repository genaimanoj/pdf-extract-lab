"""LiteParse extraction (LlamaIndex).

LiteParse is the standalone, fully-local parsing core behind LlamaParse:
PDFium text extraction with optional Tesseract OCR, exposed to Python through
Rust bindings. For each page it returns flat *text items* — a string plus an
(x, y, width, height) box and font size — which is exactly the primitive this
lab needs to paint blocks back onto the page.

This engine turns those items into structure the same way the `pymupdf` engine
does: group items into lines, take the document's dominant font size as the
body baseline, and promote larger lines to headings by font-size ratio. Keeping
the heuristic identical means headings stay comparable across the two
text-layer engines.

Scope: LiteParse emits positioned text only — no table or figure model — so
this engine deliberately produces just headings and paragraphs. Use `docling`
or the ODL engines when table structure matters.

Coordinate note: LiteParse reports boxes top-left-origin in PDF points, which
matches the schema's convention, so `(x, y, w, h)` maps straight to
`[x, y, x + w, y + h]`.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from typing import Any, Dict, List

from ..config import settings
from ..schema import ExtractionMetrics, ExtractionResult, PageMeta
from ..tools.markdown_render import blocks_to_markdown, compute_metrics
from ..tools.pdf_access import get_page_geometry, pdf_meta

# Lines shorter than this many points are never real prose — they're OCR
# speckle or rotated-glyph artifacts that would skew the body-size estimate.
_TEXT_SIZE_FLOOR = 5.0

# A heading line is large *and* short. Anything longer is a styled paragraph.
_HEADING_MAX_WORDS = 20


def _heading_level(ratio: float) -> int | None:
    """Map a font-size ratio over body text to a 1..6 heading level, or None.

    Thresholds mirror the `pymupdf` engine so the two text-layer engines
    classify headings identically.
    """
    if ratio >= 1.8:
        return 1
    if ratio >= 1.45:
        return 2
    if ratio >= 1.25:
        return 3
    if ratio >= 1.12:
        return 4
    return None


class LiteParseExtractor:
    name = "liteparse"

    async def extract(self, pdf_path: str) -> ExtractionResult:
        t0 = time.time()
        meta = pdf_meta(pdf_path)
        pages_geom = get_page_geometry(pdf_path)

        # parse() is synchronous, CPU-bound native code — keep it off the event
        # loop so concurrent requests aren't blocked while a PDF is parsed.
        result, note = await asyncio.to_thread(self._parse, pdf_path)

        lines: List[Dict[str, Any]] = []
        for page in result.pages:
            lines.extend(self._lines_for_page(page))

        # Body size = the most common rounded line size. The mode is far steadier
        # than a mean/median when most lines share one size with float jitter.
        if lines:
            body_size = Counter(round(ln["size"], 1) for ln in lines).most_common(1)[0][0]
        else:
            body_size = 10.0

        blocks = self._build_blocks(lines, body_size)
        blocks.sort(key=lambda b: (
            b.get("page", 0),
            (b.get("bbox") or [0, 0, 0, 0])[1],
            (b.get("bbox") or [0, 0, 0, 0])[0],
        ))

        # Prefer the embedded title; otherwise fall back to the first real
        # heading on page 1. Require two-plus words so a stray marginal glyph
        # (rotated banners love the left edge) can't masquerade as the title.
        title = meta.get("title")
        if not title:
            for b in blocks:
                if b.get("page") != 1 or b.get("type") != "heading":
                    continue
                if len((b.get("text") or "").split()) >= 2:
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
            notes=note,
            metrics=ExtractionMetrics(
                duration_ms=duration_ms,
                block_count=len(blocks),
                **counts,
            ),
        )

    def _parse(self, pdf_path: str):
        """Parse with LiteParse, degrading gracefully if OCR can't start.

        OCR is a fallback LiteParse only reaches for on pages with no text
        layer, and it depends on a working Tesseract install (engine + language
        data). When that isn't present a scanned page would otherwise 500 the
        whole request, so we retry once with OCR off and return the text layer
        plus a note. Returns (result, note) where note is None on the happy path.
        """
        from liteparse import ParseError

        if not settings.liteparse_ocr_enabled:
            return self._run(pdf_path, ocr=False), None
        try:
            return self._run(pdf_path, ocr=True), None
        except ParseError as exc:
            if "ocr" not in str(exc).lower():
                raise
            note = (
                "OCR was unavailable (Tesseract failed to initialize), so only "
                "the embedded text layer was parsed — pages that are pure image "
                "may come back empty. Install Tesseract language data to enable OCR."
            )
            return self._run(pdf_path, ocr=False), note

    def _run(self, pdf_path: str, *, ocr: bool):
        from liteparse import LiteParse

        parser = LiteParse(
            ocr_enabled=ocr,
            ocr_language=settings.liteparse_ocr_language,
            dpi=float(settings.liteparse_dpi),
            quiet=True,
        )
        return parser.parse(pdf_path)

    def _lines_for_page(self, page) -> List[Dict[str, Any]]:
        """Cluster a page's text items into lines, top-to-bottom.

        Items arrive flat, so we sort by vertical then horizontal position and
        merge each item into the current line when their vertical centers are
        within half a line height. That tolerance absorbs the baseline jitter
        within a line without bridging the gap to the next one.

        Sub-point items are dropped first: PDFium hands back figure-internal
        micro-text (axis labels, tick numbers) at font sizes like 1pt, and left
        in place they both skew the body-size estimate and bleed into adjacent
        real lines during clustering.
        """
        items = [
            it for it in page.text_items
            if it.text and it.text.strip()
            and (it.font_size or it.height) >= _TEXT_SIZE_FLOOR
        ]
        if not items:
            return []

        rows: List[List[Any]] = []
        row_center = 0.0
        for it in sorted(items, key=lambda i: (i.y, i.x)):
            center = it.y + it.height / 2.0
            if rows and abs(center - row_center) <= max(2.0, it.height * 0.5):
                rows[-1].append(it)
            else:
                rows.append([it])
                row_center = center

        lines: List[Dict[str, Any]] = []
        for row in rows:
            row.sort(key=lambda i: i.x)
            # Split the row wherever a wide horizontal gap opens up — a column
            # gutter or the gap between a marginal note and the body. Clustering
            # on y alone would otherwise glue a left-margin glyph onto the line
            # that happens to share its baseline.
            segment: List[Any] = [row[0]]
            for prev, it in zip(row, row[1:]):
                gap = it.x - (prev.x + prev.width)
                if gap > max(18.0, it.height * 1.5):
                    lines.append(self._line(segment, page.page_num))
                    segment = [it]
                else:
                    segment.append(it)
            lines.append(self._line(segment, page.page_num))
        return [ln for ln in lines if ln]

    @staticmethod
    def _line(items: List[Any], page_num: int) -> Dict[str, Any] | None:
        """Fold one horizontal run of items into a single line dict."""
        text = " ".join(i.text.strip() for i in items).strip()
        if not text:
            return None
        # font_size is absent for OCR'd glyphs; box height is a fair proxy.
        size = max((i.font_size or i.height) for i in items)
        return {
            "page": page_num,
            "text": text,
            "size": float(size),
            "bbox": [
                min(i.x for i in items),
                min(i.y for i in items),
                max(i.x + i.width for i in items),
                max(i.y + i.height for i in items),
            ],
        }

    def _build_blocks(self, lines: List[Dict[str, Any]], body_size: float) -> List[Dict[str, Any]]:
        """Fold a flat line list into heading and paragraph blocks."""
        blocks: List[Dict[str, Any]] = []
        para: List[Dict[str, Any]] = []

        def flush():
            nonlocal para
            if not para:
                return
            text = " ".join(ln["text"] for ln in para).strip()
            if text:
                blocks.append({
                    "id": f"lp_{uuid.uuid4().hex[:8]}",
                    "type": "paragraph",
                    "page": para[0]["page"],
                    "bbox": [
                        min(ln["bbox"][0] for ln in para),
                        min(ln["bbox"][1] for ln in para),
                        max(ln["bbox"][2] for ln in para),
                        max(ln["bbox"][3] for ln in para),
                    ],
                    "text": text,
                })
            para = []

        for ln in lines:
            ratio = ln["size"] / max(body_size, 1.0)
            level = _heading_level(ratio)
            # A heading is large, short, and at least two characters — single
            # glyphs at heading size are almost always rotated-banner fragments.
            if (level is not None
                    and len(ln["text"]) >= 2
                    and len(ln["text"].split()) <= _HEADING_MAX_WORDS):
                flush()
                blocks.append({
                    "id": f"lp_{uuid.uuid4().hex[:8]}",
                    "type": "heading",
                    "level": level,
                    "page": ln["page"],
                    "bbox": ln["bbox"],
                    "text": ln["text"],
                })
                continue

            if para:
                prev = para[-1]
                if prev["page"] != ln["page"]:
                    flush()
                else:
                    line_h = max(8.0, prev["bbox"][3] - prev["bbox"][1])
                    gap = ln["bbox"][1] - prev["bbox"][3]
                    if gap > line_h * 1.2:  # blank line between paragraphs
                        flush()
            para.append(ln)
        flush()

        return blocks
