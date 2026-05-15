import type { ExtractionResult, UploadResponse } from "./types";

// Default matches the backend's default uvicorn port (see README).
// Override with VITE_API_BASE in frontend/.env for non-default deployments.
export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8001";

async function parseError(res: Response, label: string): Promise<Error> {
  let detail = "";
  try {
    const body = await res.clone().json();
    detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body);
  } catch {
    detail = (await res.text().catch(() => "")) || res.statusText;
  }
  return new Error(`${label} failed (${res.status}): ${detail}`);
}

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
  if (!res.ok) throw await parseError(res, "Upload");
  return res.json();
}

export async function extract(fileId: string, engine: string): Promise<ExtractionResult> {
  const fd = new FormData();
  fd.append("file_id", fileId);
  fd.append("engine", engine);
  const res = await fetch(`${API_BASE}/extract`, { method: "POST", body: fd });
  if (!res.ok) throw await parseError(res, "Extract");
  return res.json();
}

export function pdfUrl(fileId: string): string {
  return `${API_BASE}/files/${fileId}`;
}

export interface EngineInfo {
  name: string;
  label: string;
  description: string;
  license: string;
  requires_env: string[];
  requires_bin: string[];
  homepage: string | null;
  available: boolean;
  unavailable_reason: string | null;
}

export async function engines(): Promise<{ engines: EngineInfo[]; default: string }> {
  const res = await fetch(`${API_BASE}/engines`);
  if (!res.ok) throw await parseError(res, "engines");
  return res.json();
}
