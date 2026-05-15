import { create } from "zustand";
import type { ExtractionResult } from "./types";

interface State {
  fileId: string | null;
  filename: string | null;
  engine: string;
  result: ExtractionResult | null;
  loading: boolean;
  error: string | null;
  selectedBlockId: string | null;
  hoveredBlockId: string | null;
  currentPage: number;

  setFile: (fileId: string, filename: string) => void;
  setEngine: (engine: string) => void;
  setResult: (result: ExtractionResult | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSelectedBlockId: (id: string | null) => void;
  setHoveredBlockId: (id: string | null) => void;
  setCurrentPage: (page: number) => void;
  reset: () => void;
}

export const useStore = create<State>((set) => ({
  fileId: null,
  filename: null,
  engine: "docling",
  result: null,
  loading: false,
  error: null,
  selectedBlockId: null,
  hoveredBlockId: null,
  currentPage: 1,

  setFile: (fileId, filename) =>
    set({ fileId, filename, result: null, selectedBlockId: null, currentPage: 1 }),
  setEngine: (engine) => set({ engine }),
  setResult: (result) => set({ result, selectedBlockId: null, currentPage: 1 }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setSelectedBlockId: (id) => set({ selectedBlockId: id }),
  setHoveredBlockId: (id) => set({ hoveredBlockId: id }),
  setCurrentPage: (page) => set({ currentPage: page }),
  reset: () =>
    set({
      fileId: null,
      filename: null,
      result: null,
      error: null,
      selectedBlockId: null,
      currentPage: 1,
    }),
}));
