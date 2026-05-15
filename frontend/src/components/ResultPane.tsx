import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { JsonView, darkStyles } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";
import { useStore } from "../store";
import type { Block } from "../types";

type Tab = "blocks" | "markdown" | "json" | "trace";

export function ResultPane() {
  const { result, selectedBlockId, setSelectedBlockId, hoveredBlockId, setHoveredBlockId } = useStore();
  const [tab, setTab] = useState<Tab>("blocks");

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const b of result?.blocks ?? []) c[b.type] = (c[b.type] || 0) + 1;
    return c;
  }, [result]);

  const tabCounts: Record<Tab, number | null> = {
    blocks: result?.blocks.length ?? 0,
    markdown: null,
    json: null,
    trace: result?.trace?.length ?? 0,
  };

  if (!result) {
    return (
      <div className="pane-empty">
        Pick an engine and press <b>Extract</b> to see the structured output here.
      </div>
    );
  }

  return (
    <div className="result-pane">
      <div className="result-header">
        <div className="tabs">
          {(["blocks", "markdown", "json", "trace"] as Tab[]).map((t) => {
            const n = tabCounts[t];
            return (
              <button
                key={t}
                className={`tab ${tab === t ? "active" : ""}`}
                onClick={() => setTab(t)}
              >
                {n != null ? `${t} (${n})` : t}
              </button>
            );
          })}
        </div>
        <div className="type-counts">
          {Object.entries(counts).map(([k, v]) => (
            <span key={k} className={`chip chip-${k}`}>{k}: {v}</span>
          ))}
        </div>
      </div>
      <div className="result-body">
        {tab === "blocks" && (
          <BlockList
            blocks={result.blocks}
            selectedId={selectedBlockId}
            hoveredId={hoveredBlockId}
            onSelect={setSelectedBlockId}
            onHover={setHoveredBlockId}
          />
        )}
        {tab === "markdown" && (
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {result.markdown ?? ""}
            </ReactMarkdown>
          </div>
        )}
        {tab === "json" && (
          <div className="json-body">
            <JsonView
              data={result as unknown as Record<string, unknown>}
              shouldExpandNode={(level) => level < 3}
              style={darkStyles}
            />
          </div>
        )}
        {tab === "trace" && <TraceList trace={result.trace ?? []} />}
      </div>
    </div>
  );
}

function BlockList({
  blocks,
  selectedId,
  hoveredId,
  onSelect,
  onHover,
}: {
  blocks: Block[];
  selectedId: string | null;
  hoveredId: string | null;
  onSelect: (id: string | null) => void;
  onHover: (id: string | null) => void;
}) {
  return (
    <div className="block-list">
      {blocks.map((b) => (
        <BlockRow
          key={b.id}
          block={b}
          selected={selectedId === b.id}
          hovered={hoveredId === b.id}
          onClick={() => onSelect(b.id === selectedId ? null : b.id)}
          onEnter={() => onHover(b.id)}
          onLeave={() => onHover(null)}
        />
      ))}
    </div>
  );
}

function BlockRow({
  block,
  selected,
  hovered,
  onClick,
  onEnter,
  onLeave,
}: {
  block: Block;
  selected: boolean;
  hovered: boolean;
  onClick: () => void;
  onEnter: () => void;
  onLeave: () => void;
}) {
  return (
    <div
      className={`block-row block-${block.type} ${selected ? "selected" : ""} ${hovered ? "hovered" : ""}`}
      onClick={onClick}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
    >
      <div className="row-header">
        <span className={`pill pill-${block.type}`}>
          {block.type}
          {block.type === "heading" ? ` · L${block.level}` : ""}
        </span>
        <span className="page-tag">p{block.page}</span>
        <span className="id-tag">{block.id}</span>
      </div>
      <div className="row-content">
        <BlockContent block={block} />
      </div>
    </div>
  );
}

function BlockContent({ block }: { block: Block }) {
  if (block.type === "heading") return <h4>{block.text}</h4>;
  if (block.type === "paragraph") return <p>{block.text}</p>;
  if (block.type === "list") {
    const Tag = block.ordered ? "ol" : "ul";
    return (
      <Tag>
        {(block.items ?? []).map((it, i) => <li key={i}>{it}</li>)}
      </Tag>
    );
  }
  if (block.type === "table") {
    const rows = block.rows ?? [];
    return (
      <div className="table-wrap">
        {block.caption && <div className="caption">{block.caption}</div>}
        <table>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {r.map((c, j) => (i === 0 ? <th key={j}>{c}</th> : <td key={j}>{c}</td>))}
              </tr>
            ))}
          </tbody>
        </table>
        <small>{block.row_count} rows × {block.col_count} cols</small>
      </div>
    );
  }
  if (block.type === "image") {
    return (
      <div className="image-card">
        <div className="img-placeholder">🖼 image region</div>
        {block.caption && <div className="caption">{block.caption}</div>}
        {block.description && <div className="desc">{block.description}</div>}
        <div className="bbox">bbox: {(block.bbox ?? []).map((v) => v.toFixed(0)).join(", ")}</div>
      </div>
    );
  }
  if (block.type === "formula") return <pre className="formula">{block.latex ?? block.text}</pre>;
  if (block.type === "caption") return <em>{block.text}</em>;
  if (block.type === "footnote") return <small>{block.text}</small>;
  return <pre>{JSON.stringify(block, null, 2)}</pre>;
}

function TraceList({ trace }: { trace: { tool: string; input: Record<string, unknown>; output_preview?: string | null; duration_ms?: number | null }[] }) {
  if (!trace.length) {
    return <div className="pane-empty">No tool trace. Run the <b>agentic</b> engine to see tool calls here.</div>;
  }
  return (
    <div className="trace-list">
      {trace.map((t, i) => (
        <div key={i} className="trace-row">
          <div className="trace-head">
            <span className="tool-name">{t.tool}</span>
            <span className="dur">{t.duration_ms ?? 0} ms</span>
          </div>
          <pre className="trace-input">{JSON.stringify(t.input, null, 2)}</pre>
          {t.output_preview && <pre className="trace-out">→ {t.output_preview}</pre>}
        </div>
      ))}
    </div>
  );
}
