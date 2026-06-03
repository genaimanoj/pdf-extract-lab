"""Extractor registry.

Each engine lives in its own file in this directory. The mapping is 1:1:

    basic                    → basic_extractor.py
    pymupdf                  → pymupdf_extractor.py
    liteparse                → liteparse_extractor.py
    docling                  → docling_extractor.py
    opendataloader           → opendataloader_extractor.py
    opendataloader_hybrid    → opendataloader_hybrid_extractor.py
    vlm                      → vlm_extractor.py
    agentic                  → agentic_extractor.py

Adding a new engine is a 3-step operation:

1. Write `my_engine_extractor.py` that exposes a class implementing the
   `Extractor` Protocol from `base.py` (just an async `extract(pdf_path)`
   method returning ExtractionResult and a `name` attribute).
2. Add one entry to the `_ENGINES` list below: an ExtractorDescriptor and
   a lazy loader that imports + returns the class.
3. That's it. /engines, /health, /extract and the frontend dropdown all
   pick it up automatically.

Loaders are intentionally lazy so missing heavy dependencies (docling,
opendataloader, litellm, claude-agent-sdk) don't break the process at
startup — only when the relevant engine is requested.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from .base import Extractor
from .descriptor import ExtractorDescriptor


# ---- Loaders -----------------------------------------------------------------
# Each returns the Extractor class. Imports inside the functions so heavy deps
# don't load until the engine is actually used.

def _load_basic():
    from .basic_extractor import BasicExtractor
    return BasicExtractor


def _load_pymupdf():
    from .pymupdf_extractor import PyMuPDFExtractor
    return PyMuPDFExtractor


def _load_liteparse():
    from .liteparse_extractor import LiteParseExtractor
    return LiteParseExtractor


def _load_docling():
    from .docling_extractor import DoclingExtractor
    return DoclingExtractor


def _load_opendataloader():
    from .opendataloader_extractor import OpenDataLoaderExtractor
    return OpenDataLoaderExtractor


def _load_opendataloader_hybrid():
    from .opendataloader_hybrid_extractor import OpenDataLoaderHybridExtractor
    return OpenDataLoaderHybridExtractor


def _load_vlm():
    from .vlm_extractor import VlmExtractor
    return VlmExtractor


def _load_agentic():
    from .agentic_extractor import AgenticExtractor
    return AgenticExtractor


# ---- Registry ---------------------------------------------------------------
# Declaration order is the order engines appear in the UI dropdown.

_ENGINES: List[Dict[str, Any]] = [
    {
        "descriptor": ExtractorDescriptor(
            name="basic",
            label="Basic (pdfplumber)",
            description="Heuristic baseline: pdfplumber + font-size heading classification.",
            license="MIT",
            homepage="https://github.com/jsvine/pdfplumber",
        ),
        "loader": _load_basic,
    },
    {
        "descriptor": ExtractorDescriptor(
            name="pymupdf",
            label="PyMuPDF (fitz)",
            description="Direct PyMuPDF text-dict walk with font-ratio heading detection.",
            license="AGPL-3.0 (PyMuPDF)",
            homepage="https://pymupdf.readthedocs.io/",
        ),
        "loader": _load_pymupdf,
    },
    {
        "descriptor": ExtractorDescriptor(
            name="liteparse",
            label="LiteParse (LlamaIndex)",
            description="Local PDFium text + bundled Tesseract OCR (the LlamaParse core). Fast spatial text with boxes; no tables.",
            license="Apache-2.0",
            homepage="https://github.com/run-llama/liteparse",
        ),
        "loader": _load_liteparse,
    },
    {
        "descriptor": ExtractorDescriptor(
            name="docling",
            label="Docling (ML)",
            description="IBM DocLayNet + TableFormer. Strong headings and tables.",
            license="MIT",
            homepage="https://github.com/docling-project/docling",
        ),
        "loader": _load_docling,
    },
    {
        "descriptor": ExtractorDescriptor(
            name="opendataloader",
            label="OpenDataLoader (Java)",
            description="Rule-based Java parser. Clean heading levels.",
            license="Apache-2.0",
            requires_bin=["java"],
            homepage="https://github.com/opendataloader-project/opendataloader-pdf",
        ),
        "loader": _load_opendataloader,
    },
    {
        "descriptor": ExtractorDescriptor(
            name="opendataloader_hybrid",
            label="ODL Hybrid (AI + Java)",
            description="ODL Java core + Docling-Fast AI backend for complex pages. Auto-spawns the backend on demand.",
            license="Apache-2.0",
            requires_bin=["java", "opendataloader-pdf-hybrid"],
            homepage="https://github.com/opendataloader-project/opendataloader-pdf",
        ),
        "loader": _load_opendataloader_hybrid,
    },
    {
        "descriptor": ExtractorDescriptor(
            name="vlm",
            label="VLM (Vision Model)",
            description="Renders pages to PNG and asks a vision LM for blocks.",
            license="depends on provider",
            requires_env=["GEMINI_API_KEY"],  # default VLM_MODEL is gemini/...
            homepage="https://aistudio.google.com/",
        ),
        "loader": _load_vlm,
    },
    {
        "descriptor": ExtractorDescriptor(
            name="agentic",
            label="Agentic (Claude)",
            description="Claude + custom tools, orchestrates Docling + ODL baselines.",
            license="depends on provider",
            requires_env=["ANTHROPIC_API_KEY"],
            homepage="https://github.com/anthropics/claude-agent-sdk-python",
        ),
        "loader": _load_agentic,
    },
]

# Public views — name-keyed for O(1) lookup while preserving declaration order.
DESCRIPTORS: Dict[str, ExtractorDescriptor] = {e["descriptor"].name: e["descriptor"] for e in _ENGINES}
_LOADERS: Dict[str, Callable[[], type]] = {e["descriptor"].name: e["loader"] for e in _ENGINES}

# Back-compat alias — old code used this name.
REGISTRY: Dict[str, Callable[[], type]] = _LOADERS


def list_descriptors() -> List[ExtractorDescriptor]:
    """Descriptors in declaration order (how the UI should render them)."""
    return [e["descriptor"] for e in _ENGINES]


def get_extractor(name: str) -> Extractor:
    loader = _LOADERS.get(name)
    if loader is None:
        raise KeyError(f"Unknown engine '{name}'. Known: {list(DESCRIPTORS)}")
    cls = loader()
    return cls()


__all__ = [
    "DESCRIPTORS",
    "REGISTRY",
    "list_descriptors",
    "get_extractor",
    "Extractor",
    "ExtractorDescriptor",
]
