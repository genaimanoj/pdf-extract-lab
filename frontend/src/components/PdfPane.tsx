import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store";
import { API_BASE } from "../api";
import type { Block } from "../types";

const BLOCK_COLORS: Record<string, string> = {
  heading: "rgba(34,139,230,0.25)",
  paragraph: "rgba(150,150,150,0.10)",
  list: "rgba(120,200,120,0.15)",
  table: "rgba(255,193,7,0.22)",
  image: "rgba(233,84,255,0.22)",
  caption: "rgba(100,200,200,0.18)",
  footnote: "rgba(180,120,0,0.15)",
  formula: "rgba(100,100,255,0.18)",
};
const BLOCK_BORDERS: Record<string, string> = {
  heading: "#228be6",
  paragraph: "#888",
  list: "#2f9e44",
  table: "#f59f00",
  image: "#c52d9b",
  caption: "#0ca678",
  footnote: "#b37400",
  formula: "#5c7cfa",
};

interface PageInfo {
  page_count: number;
  pages: { page: number; width: number; height: number }[];
}

export function PdfPane() {
  const {
    fileId,
    result,
    selectedBlockId,
    hoveredBlockId,
    setSelectedBlockId,
    setHoveredBlockId,
    setCurrentPage,
  } = useStore();
  const [info, setInfo] = useState<PageInfo | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const [pageWidth, setPageWidth] = useState(820);

  // Fetch page geometry whenever file changes
  useEffect(() => {
    setInfo(null);
    if (!fileId) return;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/pages/${fileId}/info`);
        if (!res.ok) throw new Error(String(res.status));
        const data = (await res.json()) as PageInfo;
        setInfo(data);
      } catch (e) {
        console.error("[pdf] info error", e);
      }
    })();
  }, [fileId]);

  // Resize observer for width
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setPageWidth(Math.max(320, Math.floor(el.clientWidth - 24)));
    });
    ro.observe(el);
    setPageWidth(Math.max(320, el.clientWidth - 24));
    return () => ro.disconnect();
  }, []);

  // Scroll to selected block
  useEffect(() => {
    if (!selectedBlockId || !result) return;
    const b = result.blocks.find((bl) => bl.id === selectedBlockId);
    if (!b) return;
    setCurrentPage(b.page);
    const node = pageRefs.current[b.page];
    if (node) node.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [selectedBlockId, result, setCurrentPage]);

  const blocksByPage = useMemo(() => {
    const m: Record<number, Block[]> = {};
    for (const b of result?.blocks ?? []) (m[b.page] ||= []).push(b);
    return m;
  }, [result]);

  if (!fileId) {
    return <div className="pane-empty">Upload a PDF to begin.</div>;
  }
  if (!info) {
    return (
      <div ref={containerRef} className="pdf-pane">
        <div className="pane-loading">Loading page info…</div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="pdf-pane">
      {info.pages.map((pg) => {
        const scale = pageWidth / pg.width;
        const renderedH = pg.height * scale;
        const blocks = blocksByPage[pg.page] ?? [];
        return (
          <div
            key={pg.page}
            ref={(el) => {
              pageRefs.current[pg.page] = el;
            }}
            className="pdf-page-wrap"
            data-page={pg.page}
            style={{ marginBottom: 12 }}
          >
            <div className="page-label">Page {pg.page}</div>
            <div
              style={{
                position: "relative",
                width: pageWidth,
                height: renderedH,
                background: "white",
              }}
            >
              <img
                src={`${API_BASE}/pages/${fileId}/${pg.page}.png?dpi=120`}
                alt={`Page ${pg.page}`}
                width={pageWidth}
                height={renderedH}
                style={{ display: "block", userSelect: "none" }}
                draggable={false}
              />
              <div
                className="overlay-layer"
                style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
              >
                {blocks.map((b) => (
                  <BlockOverlay
                    key={b.id}
                    block={b}
                    pageWidth={pg.width}
                    pageHeight={pg.height}
                    scale={scale}
                    selected={selectedBlockId === b.id}
                    hovered={hoveredBlockId === b.id}
                    onClick={(id) =>
                      setSelectedBlockId(id === selectedBlockId ? null : id)
                    }
                    onEnter={(id) => setHoveredBlockId(id)}
                    onLeave={() => setHoveredBlockId(null)}
                  />
                ))}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function BlockOverlay({
  block,
  scale,
  selected,
  hovered,
  onClick,
  onEnter,
  onLeave,
}: {
  block: Block;
  pageWidth: number;
  pageHeight: number;
  scale: number;
  selected: boolean;
  hovered: boolean;
  onClick: (id: string) => void;
  onEnter: (id: string) => void;
  onLeave: () => void;
}) {
  if (!block.bbox) return null;
  const [x0, y0, x1, y1] = block.bbox;
  const left = x0 * scale;
  const top = y0 * scale;
  const w = Math.max(1, (x1 - x0) * scale);
  const h = Math.max(1, (y1 - y0) * scale);
  const color = BLOCK_COLORS[block.type] ?? "rgba(100,100,100,0.1)";
  const border = BLOCK_BORDERS[block.type] ?? "#777";
  const isHi = selected || hovered;
  const label =
    block.type === "heading" ? `H${block.level}` : block.type.slice(0, 4);
  // Place the label inside the box when the block is near the top of the page
  // so it doesn't get clipped by the page wrapper.
  const labelInside = top < 20;
  return (
    <div
      className="bbox-overlay"
      data-block-id={block.id}
      onClick={() => onClick(block.id)}
      onMouseEnter={() => onEnter(block.id)}
      onMouseLeave={onLeave}
      style={{
        position: "absolute",
        left,
        top,
        width: w,
        height: h,
        background: isHi ? color.replace(/0\.1\d/, "0.35").replace("0.22", "0.45").replace("0.25", "0.45") : color,
        border: `${isHi ? 2 : 1}px solid ${border}`,
        cursor: "pointer",
        boxSizing: "border-box",
        pointerEvents: "auto",
      }}
      title={`${block.type}${
        block.type === "heading" ? ` L${block.level}` : ""
      } — ${(
        block.text ??
        block.caption ??
        block.items?.[0] ??
        ""
      ).slice(0, 120)}`}
    >
      {isHi && (
        <span
          style={{
            position: "absolute",
            top: labelInside ? 2 : -18,
            left: labelInside ? 2 : 0,
            fontSize: 10,
            padding: "1px 5px",
            background: border,
            color: "white",
            borderRadius: 3,
            whiteSpace: "nowrap",
            pointerEvents: "none",
          }}
        >
          {label}
        </span>
      )}
    </div>
  );
}
