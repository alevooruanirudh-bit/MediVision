"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, AlertOctagon, Clock, Handshake, ShieldAlert, Stethoscope, Upload } from "lucide-react";
import Link from "next/link";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { ChartTooltip, STATUS, axis, grid } from "@/components/charts";
import { EmptyState, PageHeader, StatCard } from "@/components/common";
import { Button, Card, CardHeader, Skeleton } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { ms, pct, timeAgo, titleCase } from "@/lib/utils";

const LEVELS = ["CRITICAL", "URGENT", "ROUTINE", "NORMAL", "PENDING"];

export default function DashboardPage() {
  const q = useQuery({ queryKey: ["dashboard"], queryFn: api.dashboard, refetchInterval: 20_000 });
  const d = q.data;
  const donut = d ? LEVELS.map((l) => ({ name: l, value: d.levels[l] ?? 0 })).filter((x) => x.value > 0) : [];
  const times = (d?.analysis_times ?? []).map((t, i) => ({ i: i + 1, ms: t.ms, modality: t.modality }));
  const freq = (d?.findings_frequency ?? []).map(([label, n]) => ({ label, n }));

  return (
    <div>
      <PageHeader
        eyebrow="Overview"
        title="Dashboard"
        subtitle="Live view of the AI-assisted reading room. Every number below comes from our own models and your reviews."
        actions={
          <Button asChild variant="primary">
            <Link href="/worklist"><Upload /> Upload studies</Link>
          </Button>
        }
      />
      {q.isLoading ? (
        <div className="grid gap-4 md:grid-cols-3 xl:grid-cols-6">
          {Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-28" />)}
        </div>
      ) : !d ? (
        <Card><EmptyState icon={<Activity />} title="Backend not reachable" body="Start the API with `make dev`." /></Card>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-6">
            <StatCard label="Studies" value={d.total_studies} hint={`${d.this_week} this week`} icon={<Stethoscope />} delay={0} />
            <StatCard label="Critical" value={d.critical} hint="needs immediate review" icon={<AlertOctagon />} tone="red" delay={0.04} />
            <StatCard label="Urgent" value={d.urgent} hint="same-day review" icon={<ShieldAlert />} tone="amber" delay={0.08} />
            <StatCard label="Mean analysis time" value={ms(d.mean_analysis_ms)} hint="CPU · ONNX runtime" icon={<Clock />} tone="violet" delay={0.12} />
            <StatCard label="AI–radiologist agreement" value={pct(d.agreement_rate)} hint={`${d.reviews} reviewed findings`} icon={<Handshake />} tone="teal" delay={0.16} />
            <StatCard label="OOD inputs caught" value={d.ood_count} hint="router + energy guard" icon={<ShieldAlert />} tone="cyan" delay={0.2} />
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Card>
              <CardHeader title="Worklist by urgency" subtitle="Triage level of every study" />
              {donut.length ? (
                <div className="flex items-center gap-4 px-5 pb-5">
                  <div className="relative size-44 shrink-0">
                    <ResponsiveContainer>
                      <PieChart>
                        <Pie data={donut} dataKey="value" nameKey="name" innerRadius="66%" outerRadius="100%" paddingAngle={2} stroke="var(--bg-elev)" strokeWidth={2} cornerRadius={4}>
                          {donut.map((e) => <Cell key={e.name} fill={STATUS[e.name]} />)}
                        </Pie>
                        <Tooltip content={<ChartTooltip />} />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
                      <div>
                        <div className="num text-2xl font-semibold text-fg">{d.total_studies}</div>
                        <div className="text-[10px] uppercase tracking-wider text-subtle">studies</div>
                      </div>
                    </div>
                  </div>
                  <ul className="flex-1 space-y-2 text-sm">
                    {donut.map((e) => (
                      <li key={e.name} className="flex items-center gap-2">
                        <span className="size-2.5 rounded-full" style={{ background: STATUS[e.name] }} />
                        <span className="text-muted">{titleCase(e.name.toLowerCase())}</span>
                        <span className="num ml-auto text-fg">{e.value}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <EmptyState icon={<Activity />} title="No studies yet" body="Upload images or load the demo data." />
              )}
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader title="Findings frequency" subtitle="Positive findings (calibrated probability above the per-label threshold)" />
              <div className="h-56 px-2 pb-4">
                {freq.length ? (
                  <ResponsiveContainer>
                    <BarChart data={freq} margin={{ left: 0, right: 12, top: 4 }} barCategoryGap={6}>
                      <CartesianGrid {...grid} />
                      <XAxis dataKey="label" {...axis} interval={0} angle={-20} textAnchor="end" height={48} />
                      <YAxis {...axis} allowDecimals={false} width={28} />
                      <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip />} />
                      <Bar dataKey="n" name="Studies" fill="var(--s1)" radius={[4, 4, 0, 0]} maxBarSize={28} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyState icon={<Activity />} title="No positive findings yet" />
                )}
              </div>
            </Card>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <CardHeader title="Analysis time per study" subtitle="End-to-end pipeline on CPU (router → model → XAI → measurements → triage → report)" />
              <div className="h-56 px-2 pb-4">
                <ResponsiveContainer>
                  <AreaChart data={times} margin={{ left: 0, right: 12, top: 6 }}>
                    <defs>
                      <linearGradient id="t" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0" stopColor="var(--cyan)" stopOpacity={0.35} />
                        <stop offset="1" stopColor="var(--cyan)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid {...grid} />
                    <XAxis dataKey="i" {...axis} />
                    <YAxis {...axis} width={44} tickFormatter={(v) => `${(v / 1000).toFixed(1)}s`} />
                    <Tooltip content={<ChartTooltip format={(v) => ms(Number(v))} labelFormat={(l) => `Study #${l}`} />} cursor={{ stroke: "var(--border-strong)" }} />
                    <Area type="monotone" dataKey="ms" name="Total" stroke="var(--cyan)" strokeWidth={2} fill="url(#t)" dot={false} activeDot={{ r: 4 }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </Card>
            <Card>
              <CardHeader title="Recent activity" subtitle="From the audit log" action={<Link className="text-xs text-cyan hover:underline" href="/audit">View all</Link>} />
              <ul className="scrollbar-thin max-h-56 space-y-1 overflow-auto px-3 pb-4">
                {d.recent_activity.map((a) => (
                  <li key={a.id} className="flex items-center gap-3 rounded-lg px-2 py-1.5 text-xs hover:bg-surface-2">
                    <span className="num w-14 shrink-0 text-subtle">{timeAgo(a.ts)}</span>
                    <span className="truncate text-fg">{a.action}</span>
                    <span className="ml-auto truncate text-subtle">{a.actor}</span>
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
