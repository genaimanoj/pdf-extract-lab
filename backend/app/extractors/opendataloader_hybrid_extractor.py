"""OpenDataLoader PDF Hybrid extraction (Apache-2.0).

Hybrid mode routes complex pages (tables, scanned content, figures) to a local
Docling-Fast AI backend while keeping simple text pages on the fast Java engine.
This gives ~90% improvement on table accuracy (TEDS) vs Java-only.

Requires:
  pip install "opendataloader-pdf[hybrid]"

The Docling-Fast subprocess is auto-started on first request via
`_hybrid_backend.ensure_running()`. Set HYBRID_AUTOSTART=false to opt out and
run the backend as a managed service.

Inherits the base OpenDataLoader adapter so all the JSON normalization,
bbox bottom-left → top-left conversion, empty-table filtering, and ODL
option plumbing is shared. Only the convert() call is different.
"""
from __future__ import annotations

import glob
import json
import os
import tempfile
import time
from typing import Dict

from ..config import settings
from ..schema import ExtractionMetrics, ExtractionResult, PageMeta
from ..tools.markdown_render import blocks_to_markdown, compute_metrics
from ..tools.pdf_access import get_page_geometry, pdf_meta
from . import _hybrid_backend
from .opendataloader_extractor import OpenDataLoaderExtractor


class OpenDataLoaderHybridExtractor(OpenDataLoaderExtractor):
    name = "opendataloader_hybrid"

    async def extract(self, pdf_path: str) -> ExtractionResult:
        t0 = time.time()
        meta = pdf_meta(pdf_path)
        pages_geom = get_page_geometry(pdf_path)
        page_heights: Dict[int, float] = {pg["page"]: pg["height"] for pg in pages_geom}

        hybrid_url = getattr(settings, "hybrid_url", "http://localhost:5002")

        # Make sure the Docling-Fast backend is up before the Java CLI tries
        # to talk to it. No-op if it's already healthy.
        await _hybrid_backend.ensure_running(
            url=hybrid_url,
            autostart=settings.hybrid_autostart,
            startup_timeout=settings.hybrid_startup_timeout,
        )

        with tempfile.TemporaryDirectory(prefix="odl_hyb_") as tmpd:
            import opendataloader_pdf  # type: ignore

            opendataloader_pdf.convert(
                input_path=[pdf_path],
                output_dir=tmpd,
                format="json",
                quiet=True,
                hybrid="docling-fast",
                hybrid_url=hybrid_url,
                hybrid_timeout="120000",   # 2 min timeout per page
                hybrid_fallback=True,      # fall back to Java on backend error
                table_method=settings.odl_table_method,
                reading_order=settings.odl_reading_order,
                use_struct_tree=settings.odl_use_struct_tree,
                keep_line_breaks=settings.odl_keep_line_breaks,
            )
            json_files = glob.glob(os.path.join(tmpd, "**", "*.json"), recursive=True)
            if not json_files:
                raise RuntimeError("OpenDataLoader hybrid produced no JSON output")
            with open(json_files[0], "r", encoding="utf-8") as f:
                raw = json.load(f)

        # Reuse base adapter's JSON → block normalizer (with bbox y-flip).
        blocks = self._to_blocks(raw, page_heights)

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
