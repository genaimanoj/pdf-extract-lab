"""One-shot benchmark for the engines used in the README's table.

Runs each requested engine on a single PDF and prints
    engine  block_count  wall_seconds
to stdout in CSV form. Designed to be invoked twice so the first
"cold" run (model downloads, hybrid backend spawn) doesn't dominate.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

# Make app importable when run from backend/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env so engines that need API keys pick them up.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from app.extractors import get_extractor  # noqa: E402


async def run_one(engine_name: str, pdf_path: str) -> tuple[str, int, float, str]:
    t0 = time.perf_counter()
    try:
        extractor = get_extractor(engine_name)
        result = await extractor.extract(pdf_path)
        wall = time.perf_counter() - t0
        return engine_name, len(result.blocks), wall, ""
    except Exception as exc:  # noqa: BLE001
        wall = time.perf_counter() - t0
        return engine_name, 0, wall, f"ERROR: {type(exc).__name__}: {exc}"


async def main(pdf_path: str, engines: list[str]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(["engine", "blocks", "wall_seconds", "error"])
    sys.stdout.flush()
    for engine in engines:
        name, blocks, wall, err = await run_one(engine, pdf_path)
        writer.writerow([name, blocks, f"{wall:.2f}", err])
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument(
        "--engines",
        default="pymupdf,basic,opendataloader,docling,opendataloader_hybrid,vlm",
    )
    args = parser.parse_args()
    asyncio.run(main(args.pdf, args.engines.split(",")))
