"use client";

import { useQuery } from "@tanstack/react-query";
import { TrendingDown, TrendingUp } from "lucide-react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { ChartTooltip, Legend, SERIES, axis, grid } from "@/components/charts";
import { Badge, Card, CardHeader, Skeleton } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { pct } from "@/lib/utils";

/** Disease progression over a patient's studies (M19): per-finding AI probability trajectories,
 * a linear trend per 30 days and a rule-based progression call. */
export function PatientTimeline({ patientId }: { patientId: string }) {
  const q = useQuery({ queryKey: ["timeline", patientId], queryFn: () => api.timeline(patientId) });
  if (q.isLoading) return <Skeleton className="h-72" />;
  const t = q.data;
  if (!t) return null;
  const top = t.trends.filter((x) => Math.max(...x.series) >= 0.1).slice(0, 4);
  const data = t.points.map((p, i) => ({
    date: p.date,
    ...Object.fromEntries(top.map((x) => [x.label, x.series[i]])),
  }));
  const tone = t.call.startsWith("possible progression") ? "red" : t.call.startsWith("possible improvement") ? "teal" : "neutral";
  return (
    <Card data-testid="patient-timeline">
      <CardHeader
        title="Progression timeline"
        subtitle={`${t.n_studies} studies over ${t.span_days} days · AI probability per finding`}
        action={<Badge tone={tone}>{t.call}</Badge>}
      />
      <div className="px-5">
        <Legend items={top.map((x, i) => ({ label: x.label, color: SERIES[i] }))} />
      </div>
      <div className="h-56 px-2 pb-2 pt-2">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 0 }}>
            <CartesianGrid {...grid} />
            <XAxis dataKey="date" {...axis} />
            <YAxis domain={[0, 1]} {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} width={40} />
            <ReferenceLine y={0.5} stroke="var(--border-strong)" strokeDasharray="4 4" />
            <Tooltip content={<ChartTooltip format={(v) => pct(Number(v))} />} />
            {top.map((x, i) => (
              <Line key={x.label} dataKey={x.label} name={x.label} stroke={SERIES[i]} strokeWidth={2} dot={{ r: 4, strokeWidth: 2, stroke: "var(--surface)" }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="grid gap-1 border-t border-border px-5 py-3 text-xs sm:grid-cols-2">
        {t.trends.slice(0, 6).map((x) => (
          <div key={x.label} className="flex items-center justify-between gap-2">
            <span className="text-muted">{x.label}</span>
            <span className="num inline-flex items-center gap-1 text-fg">
              {pct(x.first)} → {pct(x.last)}
              {x.trend === "worsening" ? <TrendingUp className="size-3.5 text-red" /> : x.trend === "improving" ? <TrendingDown className="size-3.5 text-teal" /> : null}
              <span className="text-subtle">({x.slope_per_30d > 0 ? "+" : ""}{(x.slope_per_30d * 100).toFixed(1)} pp/30 d)</span>
            </span>
          </div>
        ))}
      </div>
      {t.lesion_growth && (
        <div className="border-t border-border px-5 py-2 text-xs text-muted">
          Lesion size change: <span className="num text-fg">{t.lesion_growth.change_pct}%</span>
        </div>
      )}
      <p className="border-t border-border px-5 py-2 text-[11px] text-subtle">{t.note}</p>
    </Card>
  );
}
