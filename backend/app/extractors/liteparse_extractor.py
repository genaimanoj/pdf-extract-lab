"""LiteParse extraction (LlamaIndex).

LiteParse is the standalone, fully-local parsing core behind LlamaParse:
PDFium text extraction with optional Tesseract OCR, exposed to Python through
Rust bindings. For each page it returns flat *text items* — a string plus an
(x, y, width, height) box and font size — which is exactly the primitive this
lab needs to paint blocks back onto the page.

For the structured **blocks** (what paints onto the page) this engine groups
items into lines, takes the document's dominant font size as the body baseline,
and promotes lines to headings using both font size *and* font weight — the
`font_name` field catches bold/medium headings the size test alone would miss.
OCR confidence rides along onto blocks whenever LiteParse reports it.

The Markdown view is fed from LiteParse's own grid-projected `result.text`
rather than re-rendered from our blocks. That layout-preserved text is the
library's signature output: multi-column flow and even table cells survive as
aligned monospace text, so the Markdown tab shows what LiteParse can actually
do — not a lossy reconstruction.

Scope: LiteParse has no table or figure *object* in its data model, so the
structured blocks are headings and paragraphs only (table content still shows
in the Markdown view as aligned text). Reach for `docling` or the ODL engines
when you need tables and figures as structured data.

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

# Substrings that mark a bold / medium PostScript font name (e.g. "Times-Bold",
# "NimbusRomNo9L-Medi", "Helvetica-BoldOblique"). Used to catch headings set in
# a heavier weight at (or near) body size, which the size ratio alone misses.
_WEIGHT_MARKERS = ("bold", "black", "heavy", "semibold", "demibold", "medi", "-bd", "-bf")

# An emphasized line only counts as a heading when it's this short. Real section
# headings are terse ("3.1 PDF backends"); a bold run-in lead-in drags a sentence
# along behind it ("Residual Representations. In image recognition, VLAD…"), so a
# tight word cap keeps those continuations in the paragraph where they belong.
_EMPHASIS_MAX_WORDS = 5


def _is_weight_emphasized(font_name: str | None) -> bool:
    if not font_name:
        return False
    name = font_name.lower()
    return any(marker in name for marker in _WEIGHT_MARKERS)


def _block_confidence(lines: List[Dict[str, Any]]) -> float | None:
    """Lowest reported confidence across a block's lines, or None.

    Text-layer items come back at 1.0, so we only surface a value when OCR
    actually had to guess — that's the number worth showing in the output.
    """
    vals = [ln["confidence"] for ln in lines if ln.get("confidence") is not None]
    if not vals:
        return None
    lo = min(vals)
    return round(lo, 3) if lo < 0.999 else None


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
        md = self._native_markdown(result, blocks, title)
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

    @staticmethod
    def _native_markdown(result, blocks: List[Dict[str, Any]], title: str | None) -> str:
        """Render the Markdown view from LiteParse's own layout-preserved text.

        `result.text` is grid-projected — columns and table cells stay aligned
        with spaces — so it goes inside a fenced block to keep the monospacing
        that alignment depends on. With no text layer (scanned page, OCR off)
        there's nothing to show, so we fall back to the block-derived markdown.
        """
        native = (getattr(result, "text", "") or "").strip()
        if not native:
            return blocks_to_markdown(blocks, title=title)
        return f"```\n{native}\n```\n"

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
        # Size = the size most of the line's characters are set at (font_size is
        # absent for OCR'd glyphs, so box height stands in). A plain max() would
        # let one oversized glyph — a rotated-banner digit that merged in from
        # the margin — masquerade the whole line as a heading.
        sizes: Counter = Counter()
        for i in items:
            sizes[round(i.font_size or i.height, 1)] += max(1, len((i.text or "").strip()))
        size = max(sizes.items(), key=lambda kv: kv[1])[0]
        # Emphasized only when *every* item is a heavy weight. A real heading is
        # uniformly bold; a bold run-in lead-in ("Method. We then…") is mixed,
        # so this keeps those lead-ins out of the heading path.
        emphasized = all(_is_weight_emphasized(i.font_name) for i in items)
        confs = [i.confidence for i in items if i.confidence is not None]
        return {
            "page": page_num,
            "text": text,
            "size": float(size),
            "emphasized": emphasized,
            "confidence": min(confs) if confs else None,
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
                block = {
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
                }
                conf = _block_confidence(para)
                if conf is not None:
                    block["confidence"] = conf
                blocks.append(block)
            para = []

        for ln in lines:
            ratio = ln["size"] / max(body_size, 1.0)
            words = ln["text"].split()
            level = _heading_level(ratio)
            # A bold/medium-weight short line at body size is a heading the size
            # ratio alone would miss (section headings often aren't enlarged).
            if (level is None
                    and ratio >= 0.98
                    and len(words) <= _EMPHASIS_MAX_WORDS
                    and ln.get("emphasized")):
                level = 4
            # A heading is short and at least two characters — single glyphs at
            # heading size are almost always rotated-banner fragments.
            if (level is not None
                    and len(ln["text"]) >= 2
                    and len(words) <= _HEADING_MAX_WORDS):
                flush()
                block = {
                    "id": f"lp_{uuid.uuid4().hex[:8]}",
                    "type": "heading",
                    "level": level,
                    "page": ln["page"],
                    "bbox": ln["bbox"],
                    "text": ln["text"],
                }
                conf = _block_confidence([ln])
                if conf is not None:
                    block["confidence"] = conf
                blocks.append(block)
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
