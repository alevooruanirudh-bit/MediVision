"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type HeatMethod = "gradcam" | "gradcampp" | "occlusion";

interface ViewerState {
  heatmap: boolean;
  opacity: number;
  cmap: string;
  method: HeatMethod;
  mask: boolean;
  boxes: boolean;
  lines: boolean;
  invert: boolean;
  selected: string | null;
  tab: string;
  set: (p: Partial<ViewerState>) => void;
  toggle: (k: "heatmap" | "mask" | "boxes" | "lines" | "invert") => void;
}

export const useViewer = create<ViewerState>((set) => ({
  heatmap: true,
  opacity: 0.7,
  cmap: "inferno",
  method: "gradcam",
  mask: false,
  boxes: true,
  lines: true,
  invert: false,
  selected: null,
  tab: "findings",
  set: (p) => set(p),
  toggle: (k) => set((s) => ({ [k]: !s[k] }) as Partial<ViewerState>),
}));

interface SettingsState {
  reviewer: string;
  showProbabilities: boolean;
  chatDelay: number;
  set: (p: Partial<SettingsState>) => void;
}

export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({ reviewer: "Reviewing radiologist", showProbabilities: true, chatDelay: 18, set: (p) => set(p) }),
    { name: "medivision-settings" },
  ),
);
