"""Agentic hybrid extraction with Claude Agent SDK.

Strategy (for "100% accuracy at any cost"):
 1. Run Docling (ML layout+tables) AND OpenDataLoader (rule-based) in parallel
    to get two candidate baselines plus pdfplumber raw text.
 2. Expose rich per-page primitives as Claude tools:
        - get_baselines()                    -> both baseline JSONs
        - get_page_text(page)                -> raw pdfplumber text
        - get_page_fonts(page)               -> line-level font + size
        - get_images_on_page(page)           -> image regions
        - extract_table(page, bbox)          -> re-extract a table from a region
        - cross_check(page)                  -> diff of baselines on that page
        - submit_blocks(blocks)              -> record the final answer
 3. Loop Claude over pages, picking the better baseline per block, fixing
    heading levels via font heuristics, stitching cross-page tables, tagging
    images. Every tool call is streamed into `trace` for the UI.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..config import settings
from ..schema import ExtractionMetrics, ExtractionResult, PageMeta, ToolCall
from ..tools.markdown_render import blocks_to_markdown, compute_metrics
from ..tools.pdf_access import (
    extract_table_bbox,
    get_images_on_page,
    get_page_geometry,
    get_page_text,
    get_page_text_with_fonts,
    pdf_meta,
)

SYSTEM_PROMPT = """You are a PDF structure extraction agent. Your single goal is \
to emit the most accurate, complete, hierarchically-ordered structured document \
representation possible from the provided PDF.

You have two candidate baseline extractions (docling + opendataloader) and raw \
per-page primitives. Follow this protocol:

1. Call get_baselines() ONCE to load both candidates.
2. For each page (in order), call cross_check(page) to see where the baselines \
   disagree, and get_page_fonts(page) to inform heading-level choices.
3. If a table looks malformed, call extract_table(page, bbox) to re-extract it.
4. Build the final block list in reading order across pages. Prefer opendataloader \
   for heading levels when it disagrees with docling — it uses explicit font signals. \
   Prefer docling for tables (TableFormer is strong). For paragraphs, favor whichever \
   text more closely matches get_page_text(page).
5. For images, emit type="image" blocks with bbox and (if evident from a nearby \
   caption) caption/description. Never OCR or invent image contents.
6. When done, call submit_blocks(blocks=...) exactly once. Block schema:
     {id, type, page, bbox, level?, text?, items?, ordered?, rows?, col_headers?,
      caption?, description?, html?, latex?}

Types: heading | paragraph | list | table | image | caption | footnote | formula.
Headings must have level in 1..6. Tables must have rows as List[List[str]].
Reading order: sort primarily by page, then by bbox y-top, then by x-left, but \
respect multi-column layouts by detecting column breaks from the fonts/geometry \
signal. DO NOT include page headers/footers as paragraphs.

Be thorough — missing blocks or wrong heading levels count as failure."""


class AgenticExtractor:
    name = "agentic"

    async def extract(self, pdf_path: str) -> ExtractionResult:
        t0 = time.time()
        meta = pdf_meta(pdf_path)
        pages_geom = get_page_geometry(pdf_path)

        # Run baselines in parallel so the agent starts with strong candidates.
        from .docling_extractor import DoclingExtractor
        from .opendataloader_extractor import OpenDataLoaderExtractor
        doc_task = asyncio.create_task(DoclingExtractor().extract(pdf_path))
        odl_task = asyncio.create_task(OpenDataLoaderExtractor().extract(pdf_path))

        baselines: Dict[str, Optional[ExtractionResult]] = {"docling": None, "opendataloader": None}
        try:
            baselines["docling"] = await doc_task
        except Exception as e:
            baselines["docling"] = None
            logger.warning("agentic: docling baseline failed: %s", e)
        try:
            baselines["opendataloader"] = await odl_task
        except Exception as e:
            baselines["opendataloader"] = None
            logger.warning("agentic: opendataloader baseline failed: %s", e)

        if not any(baselines.values()):
            raise RuntimeError("Both baselines failed; agentic cannot proceed.")

        trace: List[ToolCall] = []
        submitted_blocks: Dict[str, Any] = {"blocks": None}

        # --- Define tools for the agent -------------------------------------
        try:
            from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient, AssistantMessage, ToolUseBlock, TextBlock
        except ImportError as e:
            raise RuntimeError(
                "claude-agent-sdk not installed; run `pip install claude-agent-sdk`"
            ) from e

        def _t(name: str, inp: Dict[str, Any], preview: str, dur: int):
            trace.append(ToolCall(tool=name, input=inp, output_preview=preview[:400], duration_ms=dur))

        @tool("get_baselines", "Return the two baseline extractions as JSON (docling, opendataloader).", {})
        async def _get_baselines(args):
            ts = time.time()
            out = {
                "docling": baselines["docling"].model_dump() if baselines["docling"] else None,
                "opendataloader": baselines["opendataloader"].model_dump() if baselines["opendataloader"] else None,
            }
            body = json.dumps(out)[:180000]  # cap
            _t("get_baselines", {}, f"docling={'ok' if out['docling'] else 'None'} odl={'ok' if out['opendataloader'] else 'None'}", int((time.time()-ts)*1000))
            return {"content": [{"type": "text", "text": body}]}

        @tool("get_page_text", "Raw pdfplumber text for one page.", {"page": int})
        async def _get_page_text(args):
            ts = time.time()
            txt = get_page_text(pdf_path, int(args["page"]))
            _t("get_page_text", args, f"{len(txt)} chars", int((time.time()-ts)*1000))
            return {"content": [{"type": "text", "text": txt}]}

        @tool("get_page_fonts", "Per-line text with font name + size for one page.", {"page": int})
        async def _get_page_fonts(args):
            ts = time.time()
            lines = get_page_text_with_fonts(pdf_path, int(args["page"]))
            body = json.dumps(lines)[:80000]
            _t("get_page_fonts", args, f"{len(lines)} lines", int((time.time()-ts)*1000))
            return {"content": [{"type": "text", "text": body}]}

        @tool("get_images_on_page", "Image regions (bbox) on one page.", {"page": int})
        async def _get_images(args):
            ts = time.time()
            imgs = get_images_on_page(pdf_path, int(args["page"]))
            _t("get_images_on_page", args, f"{len(imgs)} images", int((time.time()-ts)*1000))
            return {"content": [{"type": "text", "text": json.dumps(imgs)}]}

        @tool("extract_table", "Re-extract a table from a specific bbox on a page.",
              {"page": int, "x0": float, "y0": float, "x1": float, "y1": float})
        async def _extract_table(args):
            ts = time.time()
            rows = extract_table_bbox(pdf_path, int(args["page"]),
                                      (float(args["x0"]), float(args["y0"]),
                                       float(args["x1"]), float(args["y1"]))) or []
            _t("extract_table", args, f"{len(rows)} rows", int((time.time()-ts)*1000))
            return {"content": [{"type": "text", "text": json.dumps(rows)}]}

        @tool("cross_check", "Compare what each baseline extracted for this page.", {"page": int})
        async def _cross_check(args):
            ts = time.time()
            page = int(args["page"])
            d_blocks = [b for b in (baselines["docling"].blocks if baselines["docling"] else []) if b.get("page") == page]
            o_blocks = [b for b in (baselines["opendataloader"].blocks if baselines["opendataloader"] else []) if b.get("page") == page]
            raw = get_page_text(pdf_path, page)
            d_text = "\n".join(b.get("text","") or " ".join(b.get("items") or []) for b in d_blocks if b.get("type") in ("paragraph","heading","list","caption"))
            o_text = "\n".join(b.get("text","") or " ".join(b.get("items") or []) for b in o_blocks if b.get("type") in ("paragraph","heading","list","caption"))
            ratio = difflib.SequenceMatcher(None, d_text, o_text).ratio() if (d_text and o_text) else 0.0
            info = {
                "page": page,
                "docling_blocks": d_blocks,
                "opendataloader_blocks": o_blocks,
                "text_similarity": round(ratio, 3),
                "raw_text_length": len(raw),
            }
            body = json.dumps(info)[:120000]
            _t("cross_check", args, f"sim={ratio:.2f} d={len(d_blocks)} o={len(o_blocks)}", int((time.time()-ts)*1000))
            return {"content": [{"type": "text", "text": body}]}

        @tool("submit_blocks", "Submit the final, merged block list. Call exactly once at end.",
              {"blocks": list})
        async def _submit(args):
            ts = time.time()
            blocks = args["blocks"] or []
            # Ensure each block has an id
            for b in blocks:
                if not b.get("id"):
                    b["id"] = f"ag_{uuid.uuid4().hex[:8]}"
            submitted_blocks["blocks"] = blocks
            _t("submit_blocks", {"n": len(blocks)}, f"accepted {len(blocks)} blocks", int((time.time()-ts)*1000))
            return {"content": [{"type": "text", "text": f"submitted {len(blocks)} blocks"}]}

        server = create_sdk_mcp_server(
            name="pdfx",
            version="1.0.0",
            tools=[_get_baselines, _get_page_text, _get_page_fonts, _get_images,
                   _extract_table, _cross_check, _submit],
        )

        allowed = [
            "mcp__pdfx__get_baselines",
            "mcp__pdfx__get_page_text",
            "mcp__pdfx__get_page_fonts",
            "mcp__pdfx__get_images_on_page",
            "mcp__pdfx__extract_table",
            "mcp__pdfx__cross_check",
            "mcp__pdfx__submit_blocks",
        ]

        options = ClaudeAgentOptions(
            mcp_servers={"pdfx": server},
            allowed_tools=allowed,
            system_prompt=SYSTEM_PROMPT,
            max_turns=settings.agent_max_turns,
            model=settings.agent_model,
            permission_mode="bypassPermissions",
        )

        user_prompt = (
            f"PDF has {meta['page_count']} pages ({pages_geom[0]['width']:.0f}x"
            f"{pages_geom[0]['height']:.0f} pt). Please produce the complete structured "
            f"extraction. Start with get_baselines, then iterate page-by-page, then call "
            f"submit_blocks with the final merged block list in reading order."
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_prompt)
            async for message in client.receive_response():
                # trace logging happens inside tool handlers; here we just drain the stream
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            pass  # logged in handler
                        elif isinstance(block, TextBlock):
                            pass  # narration ignored for now

        blocks = submitted_blocks["blocks"]
        fallback_used = False
        if not blocks:
            # Agent didn't call submit — fall back to the better baseline.
            fallback_used = True
            best = baselines["docling"] or baselines["opendataloader"]
            blocks = list(best.blocks) if best else []

        # Stable ordering
        blocks.sort(key=lambda b: (
            b.get("page", 0),
            (b.get("bbox") or [0, 0, 0, 0])[1],
            (b.get("bbox") or [0, 0, 0, 0])[0],
        ))

        title = (baselines["docling"].title if baselines["docling"] else None) \
            or (baselines["opendataloader"].title if baselines["opendataloader"] else None) \
            or meta.get("title")

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
            trace=trace,
            notes=("agent fallback-to-baseline" if fallback_used else None),
            metrics=ExtractionMetrics(
                duration_ms=duration_ms,
                block_count=len(blocks),
                tool_calls=len(trace),
                **counts,
            ),
        )
