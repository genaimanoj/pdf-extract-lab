"""End-to-end smoke test for VLM (LiteLLM, provider-swappable) and agentic
(Anthropic-only via claude-agent-sdk) on a single PDF.

Overrides VLM_MODEL / AGENT_MODEL per-run by monkey-patching the settings
object so we don't need to rewrite .env between runs.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app import config as cfg  # noqa: E402
from app.extractors import get_extractor  # noqa: E402


async def run(label: str, engine: str, *, vlm_model: str | None = None,
              agent_model: str | None = None, sample_blocks: int = 3) -> None:
    if vlm_model:
        cfg.settings.vlm_model = vlm_model
    if agent_model:
        cfg.settings.agent_model = agent_model

    print(f"\n=== {label} (engine={engine}, vlm={cfg.settings.vlm_model}, "
          f"agent={cfg.settings.agent_model}) ===", flush=True)
    t0 = time.perf_counter()
    try:
        ex = get_extractor(engine)
        result = await ex.extract(str(ROOT / ".." / "samples" / "docling_paper.pdf"))
        wall = time.perf_counter() - t0
        print(f"OK  blocks={len(result.blocks)}  wall={wall:.1f}s  "
              f"title={(result.title or '')[:80]!r}  "
              f"trace_calls={len(result.trace)}  notes={result.notes!r}", flush=True)
        for b in result.blocks[:sample_blocks]:
            preview = (b.get("text") or b.get("caption") or
                       " ".join(b.get("items") or []) or
                       f"<{b.get('type')}>")
            print(f"  - page={b.get('page')} type={b.get('type')} "
                  f"text={preview[:80]!r}", flush=True)
    except Exception as exc:
        wall = time.perf_counter() - t0
        print(f"ERR wall={wall:.1f}s  {type(exc).__name__}: {exc}", flush=True)


async def main():
    await run("Agentic + Claude (dated)", "agentic",
              agent_model="claude-opus-4-5-20250929")


if __name__ == "__main__":
    asyncio.run(main())
