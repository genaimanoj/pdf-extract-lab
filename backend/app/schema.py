"""Unified extraction schema emitted by every engine.

Blocks are stored as plain dicts so we can evolve types without breaking
clients, but every block conforms to a stable set of fields documented below.

Common block fields:
    id:         stable string id (engine-prefixed)
    type:       one of: heading | paragraph | list | table | image |
                caption | footnote | formula | code | kv
    page:       1-indexed page number
    bbox:       [x0, y0, x1, y1] in PDF points (origin top-left, Docling convention)
    confidence: optional float 0..1

Type-specific fields:
    heading:   level (1..6), text
    paragraph: text
    list:      ordered (bool), items (List[str])
    table:     rows (List[List[str]]), col_headers (optional), html (optional),
               caption (optional), row_count, col_count
    image:     caption (optional), description (optional), image_path (optional)
    caption:   text, target_id (optional — id of image/table it captions)
    footnote:  text, marker (optional)
    formula:   latex (optional), text (optional)
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PageMeta(BaseModel):
    page: int
    width: float
    height: float


class ToolCall(BaseModel):
    tool: str
    input: Dict[str, Any] = Field(default_factory=dict)
    output_preview: Optional[str] = None
    duration_ms: Optional[int] = None


class ExtractionMetrics(BaseModel):
    duration_ms: int
    block_count: int
    heading_count: int = 0
    paragraph_count: int = 0
    table_count: int = 0
    image_count: int = 0
    formula_count: int = 0
    tool_calls: int = 0


class ExtractionResult(BaseModel):
    engine: str
    title: Optional[str] = None
    page_count: int
    pages: List[PageMeta] = Field(default_factory=list)
    blocks: List[Dict[str, Any]] = Field(default_factory=list)
    markdown: Optional[str] = None
    trace: List[ToolCall] = Field(default_factory=list)
    metrics: Optional[ExtractionMetrics] = None
    notes: Optional[str] = None
