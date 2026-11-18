import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const pct = (p: number | null | undefined, digits = 0) =>
  p == null || Number.isNaN(p) ? "—" : `${(p * 100).toFixed(digits)}%`;

export const fmt = (n: number | null | undefined, digits = 2) =>
  n == null || Number.isNaN(n) ? "—" : n.toFixed(digits);

export const ms = (v: number | null | undefined) =>
  v == null ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  const s = Math.round((Date.now() - d.getTime()) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export const modalityLabel: Record<string, string> = {
  chest_xray: "Chest X-ray",
  bone_xray: "Bone X-ray",
  brain_mri: "Brain MRI",
  head_ct: "Head CT",
  breast_us: "Breast US",
  fundus: "Fundus photo",
  chest_ct_volume: "Chest CT volume",
  chest_ct_scan: "Chest CT (whole scan)",
  mammography: "Mammogram",
  pet_ct: "PET-CT",
  other_medical: "Other medical",
  not_medical: "Not medical",
};

export const titleCase = (s: string) =>
  s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
