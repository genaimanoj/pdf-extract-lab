"""Render a list of extraction blocks to clean GitHub-flavored Markdown."""
from typing import Any, Dict, Iterable, List


def _escape(s: str) -> str:
    return (s or "").replace("|", "\\|").strip()


def _table_md(rows: List[List[str]], col_headers: List[str] | None = None) -> str:
    if not rows:
        return ""
    if col_headers:
        header = col_headers
        body = rows
    else:
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
    width = max(len(header), *(len(r) for r in body), 1)
    header = header + [""] * (width - len(header))
    out = ["| " + " | ".join(_escape(c) for c in header) + " |"]
    out.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for r in body:
        r = list(r) + [""] * (width - len(r))
        out.append("| " + " | ".join(_escape(c) for c in r) + " |")
    return "\n".join(out)


def blocks_to_markdown(blocks: Iterable[Dict[str, Any]], title: str | None = None) -> str:
    lines: List[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    for b in blocks:
        t = b.get("type")
        if t == "heading":
            lvl = max(1, min(6, int(b.get("level", 2))))
            lines.append(f"{'#' * lvl} {b.get('text','').strip()}")
            lines.append("")
        elif t == "paragraph":
            lines.append((b.get("text") or "").strip())
            lines.append("")
        elif t == "list":
            ordered = bool(b.get("ordered"))
            for i, item in enumerate(b.get("items", []), start=1):
                prefix = f"{i}. " if ordered else "- "
                lines.append(prefix + str(item).strip())
            lines.append("")
        elif t == "table":
            if b.get("caption"):
                lines.append(f"**{b['caption']}**")
                lines.append("")
            lines.append(_table_md(b.get("rows", []), b.get("col_headers")))
            lines.append("")
        elif t == "image":
            cap = b.get("caption") or b.get("description") or "image"
            page = b.get("page", "?")
            bbox = b.get("bbox") or []
            lines.append(f"_[image on page {page} @ bbox {bbox}]: {cap}_")
            lines.append("")
        elif t == "caption":
            lines.append(f"*{(b.get('text') or '').strip()}*")
            lines.append("")
        elif t == "footnote":
            marker = b.get("marker") or "†"
            lines.append(f"[^{marker}]: {(b.get('text') or '').strip()}")
            lines.append("")
        elif t == "formula":
            eq = b.get("latex") or b.get("text") or ""
            lines.append(f"$$\n{eq}\n$$")
            lines.append("")
        elif t == "code":
            lines.append("```")
            lines.append((b.get("text") or "").rstrip())
            lines.append("```")
            lines.append("")
        else:
            txt = b.get("text")
            if txt:
                lines.append(str(txt))
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def compute_metrics(blocks: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"heading_count": 0, "paragraph_count": 0, "table_count": 0,
              "image_count": 0, "formula_count": 0}
    for b in blocks:
        t = b.get("type")
        if t == "heading": counts["heading_count"] += 1
        elif t == "paragraph": counts["paragraph_count"] += 1
        elif t == "table": counts["table_count"] += 1
        elif t == "image": counts["image_count"] += 1
        elif t == "formula": counts["formula_count"] += 1
    return counts
