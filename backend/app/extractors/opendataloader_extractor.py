"""OpenDataLoader PDF extraction (Apache-2.0, rule-based, Java core).

Wraps the official `opendataloader-pdf` Python package, which shells out to a
bundled Java CLI and emits structured JSON with type, bbox, heading level,
font info, and table/list structure.

Two adapter-side fixes that were missing from the v1 of this engine:

1. **Coordinate system.** ODL emits bboxes in PDF native bottom-left origin.
   Our viewer renders pages as PNGs at top-left origin (matching PyMuPDF and
   the rendered image). The adapter now flips y against the page height so
   bbox overlays line up with the rendered page.
2. **Degenerate "tables".** ODL occasionally classifies tiny chart fragments
   as nested tables with all-empty cells. We skip any table where every cell
   is empty so the result panel doesn't fill up with noise.

ODL has several extraction options that this adapter exposes via settings:
    table_method     default | cluster   (cluster picks up borderless tables)
    reading_order    xycut   | off       (xycut is the default reading order)
    use_struct_tree  bool                (use tagged-PDF structure tree)
    keep_line_breaks bool                (preserve original line breaks)
"""
from __future__ import annotations

import glob
import json
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

from ..config import settings
from ..schema import ExtractionMetrics, ExtractionResult, PageMeta
from ..tools.markdown_render import blocks_to_markdown, compute_metrics
from ..tools.pdf_access import get_page_geometry, pdf_meta


class OpenDataLoaderExtractor:
    name = "opendataloader"

    async def extract(self, pdf_path: str) -> ExtractionResult:
        t0 = time.time()
        meta = pdf_meta(pdf_path)
        pages_geom = get_page_geometry(pdf_path)
        page_heights: Dict[int, float] = {pg["page"]: pg["height"] for pg in pages_geom}

        with tempfile.TemporaryDirectory(prefix="odl_") as tmpd:
            import opendataloader_pdf  # type: ignore
            opendataloader_pdf.convert(
                input_path=[pdf_path],
                output_dir=tmpd,
                format="json",
                quiet=True,
                table_method=settings.odl_table_method,
                reading_order=settings.odl_reading_order,
                use_struct_tree=settings.odl_use_struct_tree,
                keep_line_breaks=settings.odl_keep_line_breaks,
            )
            json_files = glob.glob(os.path.join(tmpd, "**", "*.json"), recursive=True)
            if not json_files:
                raise RuntimeError("OpenDataLoader produced no JSON output")
            with open(json_files[0], "r", encoding="utf-8") as f:
                raw = json.load(f)

        blocks = self._to_blocks(raw, page_heights)

        # Sort top-down (now that bboxes are top-left origin, ascending y0 = top to bottom).
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

        md = blocks_to_markdown(blocks, title=title)
        counts = compute_metrics(blocks)

        duration_ms = int((time.time() - t0) * 1000)
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

    # --- bbox conversion ----------------------------------------------------
    @staticmethod
    def _bbox_topleft(bbox: Any, page: int, page_heights: Dict[int, float]) -> Optional[List[float]]:
        """Convert ODL's bottom-left bbox to top-left origin used by the viewer.

        Returns None if the bbox is missing or malformed.
        """
        if not (isinstance(bbox, list) and len(bbox) >= 4):
            return None
        try:
            x0, y0_bl, x1, y1_bl = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except (TypeError, ValueError):
            return None
        H = page_heights.get(page)
        if H is None:
            # No page height known; fall back to passing through (better than dropping).
            return [min(x0, x1), min(y0_bl, y1_bl), max(x0, x1), max(y0_bl, y1_bl)]
        # Flip vertically: y_top = H - y_bottom_bl
        y0_tl = H - max(y0_bl, y1_bl)
        y1_tl = H - min(y0_bl, y1_bl)
        return [min(x0, x1), y0_tl, max(x0, x1), y1_tl]

    # --- JSON normalization -------------------------------------------------
    def _to_blocks(self, raw: Any, page_heights: Dict[int, float]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []

        def walk(node: Any):
            if node is None:
                return
            if isinstance(node, list):
                for n in node:
                    walk(n)
                return
            if not isinstance(node, dict):
                return
            t = str(node.get("type") or node.get("Type") or "").lower()
            page = int(node.get("page number") or node.get("pageNumber") or node.get("page") or 0) or 1
            raw_bbox = node.get("bounding box") or node.get("boundingBox") or node.get("bbox")
            bbox = self._bbox_topleft(raw_bbox, page, page_heights)
            content = node.get("content") or node.get("text") or ""

            # "text block" is a container — just recurse into it.
            if t == "text block":
                for key in ("kids", "children"):
                    if key in node and isinstance(node[key], list):
                        for c in node[key]:
                            walk(c)
                return

            if t in ("heading", "title"):
                level = node.get("heading level") or node.get("headingLevel") or 1
                # ODL sometimes uses non-integer levels like "Doctitle"; coerce.
                try:
                    level_int = int(level)
                except (TypeError, ValueError):
                    level_int = 1
                blocks.append({
                    "id": f"h_{uuid.uuid4().hex[:8]}",
                    "type": "heading",
                    "level": max(1, min(6, level_int)),
                    "page": page, "bbox": bbox,
                    "text": str(content).strip(),
                })
            elif t == "paragraph":
                if str(content).strip():
                    blocks.append({
                        "id": f"p_{uuid.uuid4().hex[:8]}",
                        "type": "paragraph",
                        "page": page, "bbox": bbox,
                        "text": str(content).strip(),
                    })
            elif t in ("list", "list_item"):
                items = node.get("items") or []
                if not items and content:
                    items = [str(content)]
                if t == "list":
                    blocks.append({
                        "id": f"l_{uuid.uuid4().hex[:8]}",
                        "type": "list",
                        "ordered": bool(node.get("ordered", False)),
                        "page": page, "bbox": bbox,
                        "items": [str(i) for i in items],
                    })
                elif str(content).strip():
                    blocks.append({
                        "id": f"li_{uuid.uuid4().hex[:8]}",
                        "type": "list",
                        "ordered": False,
                        "page": page, "bbox": bbox,
                        "items": [str(content).strip()],
                    })
            elif t in ("table", "table row", "table cell"):
                if t != "table":
                    # rows/cells are parsed as part of the parent table node
                    return
                rows = self._table_rows(node)
                # Skip degenerate tables: ODL sometimes flags tiny chart
                # fragments as nested tables with every cell empty. They're
                # almost never useful; emitting them just clutters the
                # result panel.
                if not any(any((c or "").strip() for c in r) for r in rows):
                    return
                blocks.append({
                    "id": f"t_{uuid.uuid4().hex[:8]}",
                    "type": "table",
                    "page": page, "bbox": bbox,
                    "rows": rows,
                    "row_count": len(rows),
                    "col_count": max((len(r) for r in rows), default=0),
                    "col_headers": rows[0] if rows else [],
                    "caption": node.get("caption"),
                })
            elif t in ("image", "picture", "figure"):
                blocks.append({
                    "id": f"i_{uuid.uuid4().hex[:8]}",
                    "type": "image",
                    "page": page, "bbox": bbox,
                    "caption": node.get("caption"),
                    "description": node.get("description") or "Image region detected by OpenDataLoader.",
                })
            elif t == "caption":
                if str(content).strip():
                    blocks.append({
                        "id": f"c_{uuid.uuid4().hex[:8]}",
                        "type": "caption",
                        "page": page, "bbox": bbox,
                        "text": str(content).strip(),
                    })
            elif t in ("formula", "equation"):
                blocks.append({
                    "id": f"fo_{uuid.uuid4().hex[:8]}",
                    "type": "formula",
                    "page": page, "bbox": bbox,
                    "latex": node.get("latex"),
                    "text": str(content).strip() or None,
                })
            elif t == "footnote":
                blocks.append({
                    "id": f"fn_{uuid.uuid4().hex[:8]}",
                    "type": "footnote",
                    "page": page, "bbox": bbox,
                    "text": str(content).strip(),
                })
            elif t == "footer":
                # skip running footers
                return
            elif t == "list item":
                if str(content).strip():
                    blocks.append({
                        "id": f"li_{uuid.uuid4().hex[:8]}",
                        "type": "list",
                        "ordered": False,
                        "page": page, "bbox": bbox,
                        "items": [str(content).strip()],
                    })
            # Recurse into children (opendataloader uses "kids")
            for key in ("kids", "children", "elements", "content_elements", "contents"):
                if key in node and isinstance(node[key], (list, dict)):
                    walk(node[key])

        walk(raw)
        return blocks

    def _table_rows(self, node: Dict[str, Any]) -> List[List[str]]:
        """Parse table rows from OpenDataLoader JSON.

        ODL tables can use two structures:
          1. node["kids"] → [{"type":"table row", "kids": [cells...]}]   (old format)
          2. node["rows"] → [{"type":"table row", "cells": [cells...]}]  (v2 format)
        Cells may hold content in "content" key or nested "kids" with "content".
        """
        rows_out: List[List[str]] = []

        def _gather_text(n: Any) -> str:
            """Recursively gather text from a cell node."""
            if isinstance(n, str):
                return n
            if isinstance(n, dict):
                txt = n.get("content") or n.get("text") or ""
                if txt:
                    return str(txt)
                parts = []
                for key in ("kids", "children"):
                    for child in (n.get(key) or []):
                        t = _gather_text(child)
                        if t:
                            parts.append(t)
                return " ".join(parts)
            return ""

        # Try both "rows" (v2) and "kids" (legacy) as the row container
        row_containers = node.get("rows") or node.get("kids") or []
        for row_node in row_containers:
            if not isinstance(row_node, dict):
                continue
            if str(row_node.get("type", "")).lower() != "table row":
                continue
            row: List[str] = []
            # Cells can be in "cells" (v2) or "kids" (legacy)
            cells = row_node.get("cells") or row_node.get("kids") or []
            for cell in cells:
                if isinstance(cell, dict):
                    row.append(_gather_text(cell).strip())
            if row:
                rows_out.append(row)

        return rows_out
