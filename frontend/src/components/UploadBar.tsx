import { useCallback, useEffect, useState } from "react";
import { useStore } from "../store";
import {
  engines as fetchEngines,
  extract,
  uploadPdf,
  type EngineInfo,
} from "../api";

// Matches the backend MAX_UPLOAD_MB default; the server is still the source of truth.
const MAX_UPLOAD_MB_CLIENT = 50;

export function UploadBar() {
  const {
    fileId,
    filename,
    engine,
    loading,
    error,
    result,
    setFile,
    setEngine,
    setLoading,
    setError,
    setResult,
  } = useStore();

  const [engineList, setEngineList] = useState<EngineInfo[]>([]);
  const selected = engineList.find((e) => e.name === engine);

  useEffect(() => {
    let cancelled = false;
    fetchEngines()
      .then((r) => {
        if (cancelled) return;
        setEngineList(r.engines);
        // If the persisted selection isn't advertised by the backend, fall back
        // to the backend's default — otherwise extract will 400.
        const names = new Set(r.engines.map((e) => e.name));
        if (!names.has(engine)) setEngine(r.default);
      })
      .catch((e) => {
        if (!cancelled) setError(`Could not reach backend: ${String(e?.message ?? e)}`);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setError]);

  const onPick = useCallback(
    async (ev: React.ChangeEvent<HTMLInputElement>) => {
      const f = ev.target.files?.[0];
      // Always clear the input first so re-picking the same file triggers onChange.
      ev.target.value = "";
      if (!f) return;
      setError(null);

      const looksLikePdf =
        f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf");
      if (!looksLikePdf) {
        setError("Only PDF files are accepted.");
        return;
      }
      if (f.size > MAX_UPLOAD_MB_CLIENT * 1024 * 1024) {
        setError(`File too large (max ${MAX_UPLOAD_MB_CLIENT} MB).`);
        return;
      }
      if (f.size === 0) {
        setError("File is empty.");
        return;
      }

      setLoading(true);
      try {
        const up = await uploadPdf(f);
        setFile(up.file_id, up.filename);
      } catch (e: any) {
        setError(String(e?.message ?? e));
      } finally {
        setLoading(false);
      }
    },
    [setFile, setError, setLoading]
  );

  const onExtract = useCallback(async () => {
    if (!fileId) return;
    if (selected && !selected.available) {
      setError(`${selected.label} is not available: ${selected.unavailable_reason ?? ""}`);
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const res = await extract(fileId, engine);
      setResult(res);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setLoading(false);
    }
  }, [fileId, engine, selected, setError, setLoading, setResult]);

  const extractDisabled = !fileId || loading || (selected != null && !selected.available);

  return (
    <div className="upload-bar">
      <div className="brand">
        <span className="logo">📄</span>
        <span className="title">PDF Extractor Lab</span>
      </div>
      <label className="file-picker">
        <input type="file" accept="application/pdf,.pdf" onChange={onPick} />
        <span className="btn primary">Choose PDF…</span>
      </label>
      <span className="filename" title={filename ?? undefined}>{filename ?? "no file"}</span>
      <span className="spacer" />
      <label className="engine-select">
        <span>Engine</span>
        <select
          value={engine}
          onChange={(e) => setEngine(e.target.value)}
          title={selected?.description}
        >
          {engineList.map((e) => (
            <option
              key={e.name}
              value={e.name}
              disabled={!e.available}
              title={e.available ? e.description : e.unavailable_reason ?? undefined}
            >
              {e.label}
              {e.available ? "" : " — unavailable"}
            </option>
          ))}
        </select>
      </label>
      <button
        className="btn primary"
        onClick={onExtract}
        disabled={extractDisabled}
        title={selected && !selected.available ? (selected.unavailable_reason ?? "") : undefined}
      >
        {loading ? "Extracting…" : "Extract"}
      </button>
      {result?.metrics && (
        <span className="metrics-chip">
          {result.metrics.block_count} blocks · {result.metrics.duration_ms} ms
          {typeof result.metrics.tool_calls === "number" && result.metrics.tool_calls > 0
            ? ` · ${result.metrics.tool_calls} tools`
            : ""}
        </span>
      )}
      {error && <span className="err">⚠ {error}</span>}
    </div>
  );
}
