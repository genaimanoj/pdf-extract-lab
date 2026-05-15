"""Pure VLM-based PDF extraction.

Renders each page to a high-DPI PNG and sends it to a Vision-Language Model
via LiteLLM. VLM_MODEL accepts any LiteLLM "<provider>/<model>" string;
provider-specific model IDs are documented at https://docs.litellm.ai/docs/providers.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, List

import litellm

logger = logging.getLogger(__name__)

from ..config import settings
from ..schema import ExtractionMetrics, ExtractionResult, PageMeta
from ..tools.markdown_render import blocks_to_markdown, compute_metrics
from ..tools.pdf_access import get_page_geometry, pdf_meta

# Suppress litellm debug noise
litellm.suppress_debug_info = True

# ---- Prompts ----------------------------------------------------------------

PAGE_EXTRACTION_PROMPT = """\
You are a precise document structure extractor. Analyze this PDF page image and
extract ALL content into structured JSON blocks.

Return a JSON array of blocks. Each block must have these fields:
- "type": one of "heading", "paragraph", "list", "table", "image", "caption", "footnote", "formula"
- "text": the extracted text content (for heading, paragraph, caption, footnote, formula)
- "level": heading level 1-6 (only for type="heading")
- "items": array of strings (only for type="list")
- "ordered": boolean (only for type="list")
- "rows": 2D array of strings (only for type="table", first row = headers)
- "bbox": [x0, y0, x1, y1] approximate bounding box as fraction of page (0.0-1.0)
- "caption": optional caption text (for type="image" or type="table")

Rules:
1. Extract ALL text — do not skip anything visible on the page.
2. Detect heading hierarchy from font size/weight: largest = H1, next = H2, etc.
3. Tables must have complete row data. First row is headers.
4. List items should be individual strings in the "items" array.
5. Images/figures: set type="image" with a bbox and caption if visible.
6. Formulas: extract the text/LaTeX representation.
7. Return ONLY the JSON array, no markdown fencing, no explanation.
"""


class VlmExtractor:
    name = "vlm"

    async def extract(self, pdf_path: str) -> ExtractionResult:
        t0 = time.time()
        meta = pdf_meta(pdf_path)
        pages_geom = get_page_geometry(pdf_path)
        page_count = meta["page_count"]

        # Set API keys in environment for litellm
        self._setup_env()
        self._check_api_key()

        # Render all pages to PNG and process in parallel
        page_images = self._render_pages(pdf_path, page_count)

        semaphore = asyncio.Semaphore(settings.vlm_max_concurrent)
        tasks = [
            self._extract_page(page_images[i], i + 1, pages_geom[i], semaphore)
            for i in range(page_count)
        ]
        page_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge blocks from all pages
        all_blocks: List[Dict[str, Any]] = []
        for i, result in enumerate(page_results):
            if isinstance(result, Exception):
                logger.warning("vlm page %d failed: %s", i + 1, result)
                continue
            all_blocks.extend(result)

        # Sort by page, then y, then x
        all_blocks.sort(key=lambda b: (
            b.get("page", 0),
            (b.get("bbox") or [0, 0, 0, 0])[1],
            (b.get("bbox") or [0, 0, 0, 0])[0],
        ))

        title = meta.get("title")
        if not title:
            for b in all_blocks:
                if b.get("type") == "heading" and b.get("level") == 1:
                    title = b.get("text")
                    break

        md = blocks_to_markdown(all_blocks, title=title)
        counts = compute_metrics(all_blocks)
        duration_ms = int((time.time() - t0) * 1000)

        return ExtractionResult(
            engine=self.name,
            title=title,
            page_count=page_count,
            pages=[PageMeta(**pg) for pg in pages_geom],
            blocks=all_blocks,
            markdown=md,
            metrics=ExtractionMetrics(
                duration_ms=duration_ms,
                block_count=len(all_blocks),
                **counts,
            ),
        )

    # Map "<provider>/..." prefix → required env var name for that provider.
    # LiteLLM reads the key from os.environ; we just enforce presence here.
    _PROVIDER_ENV = {
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    def _setup_env(self):
        """Push API keys from settings into env for litellm."""
        import os
        if settings.gemini_api_key:
            os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
        if settings.anthropic_api_key:
            os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

    def _check_api_key(self):
        """Raise a clear error if the required API key is missing."""
        import os
        model = settings.vlm_model
        provider = model.split("/", 1)[0] if "/" in model else ""
        env_name = self._PROVIDER_ENV.get(provider)
        if env_name and not os.environ.get(env_name):
            raise RuntimeError(
                f"VLM model '{model}' requires {env_name}. "
                "Set it in backend/.env."
            )

    def _render_pages(self, pdf_path: str, page_count: int) -> List[bytes]:
        """Render all pages to PNG bytes at configured DPI."""
        import fitz
        dpi = settings.vlm_dpi
        images = []
        with fitz.open(pdf_path) as doc:
            for i in range(page_count):
                page = doc[i]
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                images.append(pix.tobytes("png"))
        return images

    async def _extract_page(
        self,
        png_bytes: bytes,
        page_num: int,
        page_geom: dict,
        semaphore: asyncio.Semaphore,
    ) -> List[Dict[str, Any]]:
        """Send a single page image to the VLM and parse the response."""
        async with semaphore:
            b64 = base64.b64encode(png_bytes).decode("utf-8")

            response = await litellm.acompletion(
                model=settings.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PAGE_EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=8192,
                temperature=0.0,
            )

            raw_text = response.choices[0].message.content or "[]"
            return self._parse_vlm_response(raw_text, page_num, page_geom)

    def _parse_vlm_response(
        self, raw: str, page_num: int, page_geom: dict
    ) -> List[Dict[str, Any]]:
        """Parse the VLM JSON response into normalized blocks."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON array from the text
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                try:
                    items = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    logger.warning("vlm: failed to parse page %d JSON", page_num)
                    return []
            else:
                return []

        if not isinstance(items, list):
            items = [items]

        pw = page_geom["width"]
        ph = page_geom["height"]

        blocks = []
        for item in items:
            if not isinstance(item, dict):
                continue
            typ = str(item.get("type", "paragraph")).lower()
            if typ not in ("heading", "paragraph", "list", "table", "image",
                           "caption", "footnote", "formula"):
                typ = "paragraph"

            # Convert fractional bbox (0-1) to PDF points
            raw_bbox = item.get("bbox")
            bbox = None
            if raw_bbox and isinstance(raw_bbox, list) and len(raw_bbox) >= 4:
                x0, y0, x1, y1 = [float(v) for v in raw_bbox[:4]]
                # If values are 0-1 (fractional), convert to points
                if all(0 <= v <= 1.01 for v in [x0, y0, x1, y1]):
                    bbox = [x0 * pw, y0 * ph, x1 * pw, y1 * ph]
                else:
                    bbox = [x0, y0, x1, y1]

            block: Dict[str, Any] = {
                "id": f"v_{uuid.uuid4().hex[:8]}",
                "type": typ,
                "page": page_num,
                "bbox": bbox,
            }

            if typ == "heading":
                block["text"] = str(item.get("text", "")).strip()
                block["level"] = max(1, min(6, int(item.get("level", 2))))
            elif typ in ("paragraph", "caption", "footnote"):
                block["text"] = str(item.get("text", "")).strip()
                if not block["text"]:
                    continue
            elif typ == "formula":
                block["text"] = str(item.get("text", "")).strip()
                block["latex"] = item.get("latex")
            elif typ == "list":
                block["items"] = [str(i) for i in (item.get("items") or [])]
                block["ordered"] = bool(item.get("ordered", False))
                if not block["items"]:
                    continue
            elif typ == "table":
                rows = item.get("rows") or []
                block["rows"] = rows
                block["row_count"] = len(rows)
                block["col_count"] = max((len(r) for r in rows), default=0)
                block["col_headers"] = rows[0] if rows else []
                block["caption"] = item.get("caption")
            elif typ == "image":
                block["caption"] = item.get("caption")
                block["description"] = item.get("description") or "Image detected by VLM."

            blocks.append(block)

        return blocks
