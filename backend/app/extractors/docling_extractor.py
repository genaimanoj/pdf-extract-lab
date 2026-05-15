"""Docling-powered extraction (ML: DocLayNet + TableFormer).

Docling's DoclingDocument preserves heading levels, table structure, and bbox
coordinates per element. We normalize it into our unified block schema.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

from ..schema import ExtractionMetrics, ExtractionResult, PageMeta
from ..tools.markdown_render import blocks_to_markdown, compute_metrics
from ..tools.pdf_access import get_page_geometry, pdf_meta


class DoclingExtractor:
    name = "docling"

    def __init__(self) -> None:
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter

    async def extract(self, pdf_path: str) -> ExtractionResult:
        t0 = time.time()
        meta = pdf_meta(pdf_path)
        pages_geom = get_page_geometry(pdf_path)

        converter = self._get_converter()
        result = converter.convert(pdf_path)
        doc = result.document

        blocks: List[Dict[str, Any]] = []

        # Cache page heights for coord conversion
        page_height_cache: Dict[int, float] = {pg["page"]: pg["height"] for pg in pages_geom}

        def _bbox_of(item) -> List[float] | None:
            prov = getattr(item, "prov", None)
            if not prov:
                return None
            p0 = prov[0]
            b = getattr(p0, "bbox", None)
            if b is None:
                return None
            try:
                if hasattr(b, "l"):
                    l = float(b.l); t_ = float(b.t); r = float(b.r); b_ = float(b.b)
                elif hasattr(b, "__iter__"):
                    vals = list(b)
                    l, t_, r, b_ = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                else:
                    return None
            except Exception:
                return None
            origin = getattr(b, "coord_origin", None)
            is_bottomleft = str(origin).upper().endswith("BOTTOMLEFT")
            page_no = int(getattr(p0, "page_no", 1) or 1)
            if is_bottomleft:
                H = page_height_cache.get(page_no, 792.0)
                y0 = H - max(t_, b_)
                y1 = H - min(t_, b_)
            else:
                y0, y1 = sorted([t_, b_])
            x0, x1 = sorted([l, r])
            return [x0, y0, x1, y1]

        def _page_of(item) -> int:
            prov = getattr(item, "prov", None)
            if prov:
                return int(getattr(prov[0], "page_no", 1))
            return 1

        # Headings
        for h in getattr(doc, "texts", []) or []:
            label = str(getattr(h, "label", "")).lower()
            text = (getattr(h, "text", "") or "").strip()
            if not text:
                continue
            if "section_header" in label or label == "title":
                level = int(getattr(h, "level", 1) or 1)
                if label == "title":
                    level = 1
                blocks.append({
                    "id": f"h_{uuid.uuid4().hex[:8]}",
                    "type": "heading",
                    "level": max(1, min(6, level)),
                    "page": _page_of(h),
                    "bbox": _bbox_of(h),
                    "text": text,
                })
            elif label == "caption":
                blocks.append({
                    "id": f"c_{uuid.uuid4().hex[:8]}",
                    "type": "caption",
                    "page": _page_of(h),
                    "bbox": _bbox_of(h),
                    "text": text,
                })
            elif label in ("footnote", "page_footer", "page_header"):
                if label == "footnote":
                    blocks.append({
                        "id": f"f_{uuid.uuid4().hex[:8]}",
                        "type": "footnote",
                        "page": _page_of(h),
                        "bbox": _bbox_of(h),
                        "text": text,
                    })
                # skip headers/footers
            elif label in ("formula",):
                blocks.append({
                    "id": f"fo_{uuid.uuid4().hex[:8]}",
                    "type": "formula",
                    "page": _page_of(h),
                    "bbox": _bbox_of(h),
                    "text": text,
                })
            elif label == "list_item":
                # accumulate as standalone list items — we'll group by page below
                blocks.append({
                    "id": f"li_{uuid.uuid4().hex[:8]}",
                    "type": "list",
                    "ordered": False,
                    "page": _page_of(h),
                    "bbox": _bbox_of(h),
                    "items": [text],
                })
            else:
                blocks.append({
                    "id": f"p_{uuid.uuid4().hex[:8]}",
                    "type": "paragraph",
                    "page": _page_of(h),
                    "bbox": _bbox_of(h),
                    "text": text,
                })

        # Tables
        for tbl in getattr(doc, "tables", []) or []:
            rows: List[List[str]] = []
            try:
                # Prefer the docling table rendering helpers if available.
                if hasattr(tbl, "export_to_dataframe"):
                    df = tbl.export_to_dataframe()
                    if df is not None:
                        rows = [list(df.columns.astype(str))] + df.astype(str).values.tolist()
            except Exception:
                rows = []
            if not rows:
                data = getattr(tbl, "data", None)
                grid = getattr(data, "grid", None) if data is not None else None
                if grid:
                    rows = [[getattr(cell, "text", "") or "" for cell in row] for row in grid]
            html = None
            try:
                if hasattr(tbl, "export_to_html"):
                    html = tbl.export_to_html()
            except Exception:
                html = None
            caption = None
            try:
                caps = getattr(tbl, "captions", None) or []
                if caps:
                    caption = getattr(caps[0], "text", None) or None
            except Exception:
                caption = None
            blocks.append({
                "id": f"t_{uuid.uuid4().hex[:8]}",
                "type": "table",
                "page": _page_of(tbl),
                "bbox": _bbox_of(tbl),
                "rows": rows,
                "row_count": len(rows),
                "col_count": max((len(r) for r in rows), default=0),
                "col_headers": rows[0] if rows else [],
                "html": html,
                "caption": caption,
            })

        # Pictures / Figures — we bbox them and flag as image.
        for pic in getattr(doc, "pictures", []) or []:
            caption = None
            try:
                caps = getattr(pic, "captions", None) or []
                if caps:
                    caption = getattr(caps[0], "text", None)
            except Exception:
                pass
            blocks.append({
                "id": f"i_{uuid.uuid4().hex[:8]}",
                "type": "image",
                "page": _page_of(pic),
                "bbox": _bbox_of(pic),
                "caption": caption,
                "description": "Figure / picture region (contents not OCR'd).",
            })

        # Stable order by (page, y-top, x-left)
        blocks.sort(key=lambda b: (
            b.get("page", 0),
            (b.get("bbox") or [0, 0, 0, 0])[1],
            (b.get("bbox") or [0, 0, 0, 0])[0],
        ))

        # Title heuristic: first heading with level 1
        title = meta.get("title")
        if not title:
            for b in blocks:
                if b.get("type") == "heading" and b.get("level") == 1:
                    title = b.get("text")
                    break

        duration_ms = int((time.time() - t0) * 1000)
        md = None
        try:
            md = doc.export_to_markdown()
        except Exception:
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
