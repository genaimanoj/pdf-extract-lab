export type BBox = [number, number, number, number];

export interface PageMeta {
  page: number;
  width: number;
  height: number;
}

export interface Block {
  id: string;
  type:
    | "heading"
    | "paragraph"
    | "list"
    | "table"
    | "image"
    | "caption"
    | "footnote"
    | "formula"
    | "code"
    | (string & {});
  page: number;
  bbox?: BBox;
  level?: number;
  text?: string;
  items?: string[];
  ordered?: boolean;
  rows?: string[][];
  col_headers?: string[];
  col_count?: number;
  row_count?: number;
  caption?: string | null;
  description?: string | null;
  html?: string | null;
  latex?: string | null;
  confidence?: number | null;
}

export interface ToolCall {
  tool: string;
  input: Record<string, unknown>;
  output_preview?: string | null;
  duration_ms?: number | null;
}

export interface ExtractionMetrics {
  duration_ms: number;
  block_count: number;
  heading_count?: number;
  paragraph_count?: number;
  table_count?: number;
  image_count?: number;
  formula_count?: number;
  tool_calls?: number;
}

export interface ExtractionResult {
  engine: string;
  title?: string | null;
  page_count: number;
  pages: PageMeta[];
  blocks: Block[];
  markdown?: string | null;
  trace: ToolCall[];
  metrics?: ExtractionMetrics | null;
  notes?: string | null;
}

export interface UploadResponse {
  file_id: string;
  bytes: number;
  filename: string;
}
