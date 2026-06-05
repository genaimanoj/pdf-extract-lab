"""MinerU (OpenDataLab) extraction — ML layout + tables + formulas.

MinerU is a heavyweight document-parsing pipeline: layout detection, OCR,
TableFormer-class table recognition, and formula recognition. We invoke its
CLI into a temp directory and read back the flat ``*_content_list.json`` plus
the ``*.md`` it writes, then normalize into our unified block schema.

Why the CLI + output files rather than the in-process Python API: MinerU's
internal function signatures churn between 2.x releases, but the CLI and its
output-file format are the stable, documented contract. Running it as a
subprocess also keeps MinerU's heavy imports (torch, models) — and any crash —
out of the API process. The first run downloads layout/OCR models (hundreds of
MB) from HF Hub.

Coordinates: ``content_list.json`` bboxes are normalized to a 0..1000 box per
page with a top-left origin. We scale them back to PDF points using the page
geometry, so overlays line up with the rendered page (same top-left convention
as the Docling / OpenDataLoader engines).

Failures (missing CLI, model error, timeout) are surfaced as a ``notes`` string
with empty blocks rather than raising — the result panel then shows the reason
instead of a 500.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import subprocess
import time
import tempfile
import uuid
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

from ..schema import ExtractionMetrics, ExtractionResult, PageMeta
from ..tools.markdown_render import blocks_to_markdown, compute_metrics
from ..tools.pdf_access import get_page_geometry, pdf_meta
from .descriptor import _resolve_bin

# Model inference on CPU can be slow on large PDFs; cap it so a stuck run
# surfaces as a note instead of hanging the request forever.
_PARSE_TIMEOUT_S = 900


def _as_int(v: Any) -> Optional[int]:
    """Coerce a value to int, tolerating str like '1'; None on failure."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _join_caption(cap: Any) -> Optional[str]:
    """content_list captions are lists of strings; flatten to one or None."""
    if not cap:
        return None
    if isinstance(cap, list):
        joined = " ".join(str(c).strip() for c in cap if str(c).strip())
        return joined or None
    s = str(cap).strip()
    return s or None


class _TableHTMLParser(HTMLParser):
    """Minimal ``<table>`` → rows[][] parser (stdlib only)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(self._cell.strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell += data


def _html_table_to_rows(html: str) -> List[List[str]]:
    if not html:
        return []
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    return [r for r in parser.rows if any(c for c in r)]


class MinerUExtractor:
    name = "mineru"

    async def extract(self, pdf_path: str) -> ExtractionResult:
        t0 = time.time()
        meta = pdf_meta(pdf_path)
        pages_geom = get_page_geometry(pdf_path)
        page_dims: Dict[int, Tuple[float, float]] = {
            pg["page"]: (pg["width"], pg["height"]) for pg in pages_geom
        }

        note: Optional[str] = None
        items: List[Dict[str, Any]] = []
        markdown: Optional[str] = None
        try:
            # MinerU is CPU/GPU-bound model inference — keep it off the loop.
            items, markdown = await asyncio.to_thread(self._run, pdf_path)
        except Exception as e:  # noqa: BLE001 — surface as a note, don't 500
            note = f"MinerU failed: {e}"

        blocks = self._to_blocks(items, page_dims) if items else []

        # Top-down reading order (bboxes are now top-left origin in points).
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

        if not markdown:
            markdown = blocks_to_markdown(blocks, title=title)
        counts = compute_metrics(blocks)
        duration_ms = int((time.time() - t0) * 1000)

        return ExtractionResult(
            engine=self.name,
            title=title,
            page_count=meta["page_count"],
            pages=[PageMeta(**pg) for pg in pages_geom],
            blocks=blocks,
            markdown=markdown,
            metrics=ExtractionMetrics(
                duration_ms=duration_ms,
                block_count=len(blocks),
                **counts,
            ),
            notes=note,
        )

    # --- MinerU invocation --------------------------------------------------
    def _run(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Run the MinerU CLI into a temp dir; return (content_list, markdown)."""
        # Resolve via the same venv-aware lookup the descriptor uses, so that
        # "available" on /engines agrees with what extract() can actually find
        # (shutil.which alone misses .venv/bin when uvicorn isn't activated).
        mineru_bin = _resolve_bin("mineru")
        if not mineru_bin:
            raise RuntimeError(
                "the 'mineru' CLI is not installed (pip install 'mineru[pipeline]')"
            )
        with tempfile.TemporaryDirectory(prefix="mineru_") as tmpd:
            proc = subprocess.run(
                [mineru_bin, "-p", pdf_path, "-o", tmpd, "-b", "pipeline"],
                capture_output=True,
                text=True,
                timeout=_PARSE_TIMEOUT_S,
            )
            cl_files = glob.glob(
                os.path.join(tmpd, "**", "*_content_list.json"), recursive=True
            )
            if not cl_files:
                tail = (proc.stderr or proc.stdout or "").strip()[-500:]
                raise RuntimeError(
                    f"no content_list.json produced (exit {proc.returncode}). {tail}"
                )
            with open(cl_files[0], "r", encoding="utf-8") as f:
                items = json.load(f)
            markdown: Optional[str] = None
            md_files = glob.glob(os.path.join(tmpd, "**", "*.md"), recursive=True)
            if md_files:
                with open(md_files[0], "r", encoding="utf-8") as f:
                    markdown = f.read()
        return items, markdown

    # --- bbox conversion ----------------------------------------------------
    @staticmethod
    def _bbox_points(
        bbox: Any, page: int, page_dims: Dict[int, Tuple[float, float]]
    ) -> Optional[List[float]]:
        """Scale a content_list bbox (0..1000, top-left origin) to PDF points."""
        if not (isinstance(bbox, list) and len(bbox) >= 4):
            return None
        dims = page_dims.get(page)
        if not dims:
            return None
        w, h = dims
        try:
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            return None
        sx, sy = w / 1000.0, h / 1000.0
        return [min(x0, x1) * sx, min(y0, y1) * sy, max(x0, x1) * sx, max(y0, y1) * sy]

    # --- content_list → blocks ---------------------------------------------
    def _to_blocks(
        self, items: List[Dict[str, Any]], page_dims: Dict[int, Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            itype = str(it.get("type") or "").lower()
            page = int(it.get("page_idx", 0) or 0) + 1
            bbox = self._bbox_points(it.get("bbox"), page, page_dims)

            # Running page furniture — not document content. (In content_list,
            # "header"/"footer" are page running heads, NOT section headings;
            # real headings are type "text" with a text_level.)
            if itype in ("header", "footer", "page_header", "page_footer", "page_number"):
                continue

            if itype in ("text", "title"):
                text = (it.get("text") or "").strip()
                if not text:
                    continue
                level = _as_int(it.get("text_level"))  # int in 2.x/3.x, but be defensive
                is_heading = itype == "title" or (level is not None and level >= 1)
                if is_heading:
                    lvl = level if (level is not None and level >= 1) else 1
                    blocks.append({
                        "id": f"mu_h_{uuid.uuid4().hex[:8]}",
                        "type": "heading",
                        "level": max(1, min(6, lvl)),
                        "page": page, "bbox": bbox, "text": text,
                    })
                else:
                    blocks.append({
                        "id": f"mu_p_{uuid.uuid4().hex[:8]}",
                        "type": "paragraph",
                        "page": page, "bbox": bbox, "text": text,
                    })
            elif itype in ("page_footnote", "footnote"):
                text = (it.get("text") or "").strip()
                if text:
                    blocks.append({
                        "id": f"mu_fn_{uuid.uuid4().hex[:8]}",
                        "type": "footnote",
                        "page": page, "bbox": bbox, "text": text,
                    })
            elif itype == "table":
                html = it.get("table_body") or ""
                rows = _html_table_to_rows(html)
                blocks.append({
                    "id": f"mu_t_{uuid.uuid4().hex[:8]}",
                    "type": "table",
                    "page": page, "bbox": bbox,
                    "rows": rows,
                    "row_count": len(rows),
                    "col_count": max((len(r) for r in rows), default=0),
                    "col_headers": rows[0] if rows else [],
                    "html": html or None,
                    "caption": _join_caption(it.get("table_caption")),
                })
            elif itype in ("image", "chart"):
                blocks.append({
                    "id": f"mu_i_{uuid.uuid4().hex[:8]}",
                    "type": "image",
                    "page": page, "bbox": bbox,
                    "caption": _join_caption(it.get("image_caption")),
                    "description": f"{itype.capitalize()} region (MinerU).",
                })
            elif itype in ("equation", "interline_equation"):
                latex = (it.get("text") or "").strip()
                blocks.append({
                    "id": f"mu_fo_{uuid.uuid4().hex[:8]}",
                    "type": "formula",
                    "page": page, "bbox": bbox,
                    "latex": latex or None,
                    "text": latex or None,
                })
            elif itype == "list":
                list_items = it.get("list_items") or it.get("items")
                if not list_items:
                    txt = (it.get("text") or "").strip()
                    list_items = [ln for ln in txt.splitlines() if ln.strip()] if txt else []
                if list_items:
                    blocks.append({
                        "id": f"mu_l_{uuid.uuid4().hex[:8]}",
                        "type": "list",
                        "ordered": bool(it.get("ordered", False)),
                        "page": page, "bbox": bbox,
                        "items": [str(i).strip() for i in list_items],
                    })
            elif itype == "code":
                text = (it.get("text") or "").rstrip()
                if text:
                    blocks.append({
                        "id": f"mu_co_{uuid.uuid4().hex[:8]}",
                        "type": "code",
                        "page": page, "bbox": bbox, "text": text,
                    })
            else:
                text = (it.get("text") or "").strip()
                if text:
                    blocks.append({
                        "id": f"mu_p_{uuid.uuid4().hex[:8]}",
                        "type": "paragraph",
                        "page": page, "bbox": bbox, "text": text,
                    })
        return blocks
