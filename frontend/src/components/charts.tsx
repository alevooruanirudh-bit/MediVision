"use client";

import * as React from "react";

/** Categorical series in fixed order (validated palette; light/dark via CSS vars). */
export const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];
export const STATUS: Record<string, string> = {
  CRITICAL: "var(--red)",
  URGENT: "var(--amber)",
  ROUTINE: "var(--cyan)",
  NORMAL: "var(--teal)",
  PENDING: "var(--subtle)",
};

export const axis = {
  stroke: "var(--subtle)",
  tick: { fill: "var(--muted)", fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: "var(--border)" },
} as const;

export const grid = { stroke: "var(--border)", strokeDasharray: "0", vertical: false } as const;

interface TipPayload {
  name?: string;
  value?: number | string;
  color?: string;
  dataKey?: string | number;
  payload?: Record<string, unknown>;
}

export function ChartTooltip({
  active,
  payload,
  label,
  format = (v) => String(v),
  labelFormat,
}: {
  active?: boolean;
  payload?: TipPayload[];
  label?: string | number;
  format?: (v: number | string, name?: string) => string;
  labelFormat?: (l: string | number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-border bg-elev/95 px-3 py-2 text-xs shadow-xl backdrop-blur">
      {label != null && label !== "" && (
        <div className="mb-1 font-medium text-fg">{labelFormat ? labelFormat(label) : label}</div>
      )}
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 text-muted">
          <span className="size-2 rounded-full" style={{ background: p.color }} />
          <span>{p.name}</span>
          <span className="num ml-auto pl-3 text-fg">{p.value != null ? format(p.value, p.name) : "—"}</span>
        </div>
      ))}
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-0.5 w-4 rounded"
            style={{ background: i.dashed ? `repeating-linear-gradient(90deg, ${i.color} 0 4px, transparent 4px 7px)` : i.color }}
          />
          {i.label}
        </span>
      ))}
    </div>
  );
}
