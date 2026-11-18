"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowDownRight, ArrowRight, ArrowUpRight, GitCompareArrows, Link2, Minus } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import * as React from "react";

import { EmptyState, PageHeader, UrgencyPill } from "@/components/common";
import { Badge, Card, CardHeader, NativeSelect, Skeleton, Switch } from "@/components/ui/primitives";
import { PatientTimeline } from "@/components/timeline";
import { ImageViewer, type View } from "@/components/viewer/image-viewer";
import { api } from "@/lib/api";
import type { StudySummary } from "@/lib/types";
import { cn, fmt, pct } from "@/lib/utils";

const STATUS_TONE = { new: "red", resolved: "teal", persistent: "amber", absent: "neutral" } as const;

function Inner() {
  const params = useSearchParams();
  const router = useRouter();
  const studies = useQuery({ queryKey: ["studies"], queryFn: api.studies });
  const byPatient = React.useMemo(() => {
    const m = new Map<string, StudySummary[]>();
    (studies.data ?? []).filter((s) => s.status === "analyzed" && s.pipeline).forEach((s) => {
      const k = s.patient?.pseudonym ?? "?";
      m.set(k, [...(m.get(k) ?? []), s]);
    });
    return [...m.entries()].filter(([, v]) => v.length >= 2);
  }, [studies.data]);
  const a = params.get("a");
  const b = params.get("b");
  React.useEffect(() => {
    if ((!a || !b) && byPatient.length) {
      const [, list] = byPatient[0];
      const sorted = [...list].sort((x, y) => (x.study_date ?? x.created_at).localeCompare(y.study_date ?? y.created_at));
      router.replace(`/compare?a=${sorted[0].id}&b=${sorted[sorted.length - 1].id}`);
    }
  }, [a, b, byPatient, router]);
  const [finding, setFinding] = React.useState<string | undefined>();
  const cmp = useQuery({
    queryKey: ["compare", a, b, finding],
    queryFn: () => api.compare(a!, b!, finding),
    enabled: !!a && !!b,
  });
  const [view, setView] = React.useState<View>({ zoom: 1, x: 0, y: 0 });
  const [sync, setSync] = React.useState(true);
  const [diff, setDiff] = React.useState(true);
  const [viewB, setViewB] = React.useState<View>({ zoom: 1, x: 0, y: 0 });
  const d = cmp.data;

  return (
    <div>
      <PageHeader
        eyebrow="Personalised · III-G"
        title="Longitudinal comparison"
        subtitle="Two studies of the same patient side by side with synced zoom/pan, per-finding deltas, CTR change and a heatmap difference map."
        actions={
          byPatient.length > 0 && (
            <NativeSelect
              aria-label="Patient"
              value={d?.current_study.patient?.pseudonym ?? ""}
              onChange={(e) => {
                const list = byPatient.find(([p]) => p === e.target.value)?.[1] ?? [];
                const s = [...list].sort((x, y) => (x.study_date ?? x.created_at).localeCompare(y.study_date ?? y.created_at));
                if (s.length >= 2) router.push(`/compare?a=${s[0].id}&b=${s[s.length - 1].id}`);
              }}
            >
              {byPatient.map(([p, l]) => <option key={p} value={p}>{p} · {l.length} studies</option>)}
            </NativeSelect>
          )
        }
      />
      {!a || !b ? (
        <Card>
          {studies.isLoading ? <Skeleton className="h-64" /> : (
            <EmptyState icon={<GitCompareArrows />} title="No patient with two analysed studies" body="Load the demo data — it includes a follow-up pair — or upload two studies with the same patient reference." />
          )}
        </Card>
      ) : cmp.isLoading || !d ? (
        <Skeleton className="h-[60vh]" />
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-4 text-xs text-muted">
            <label className="flex items-center gap-2"><Switch checked={sync} onCheckedChange={setSync} aria-label="Sync viewers" /><Link2 className="size-3.5" /> Sync zoom & pan</label>
            <label className="flex items-center gap-2"><Switch checked={diff} onCheckedChange={setDiff} aria-label="Heatmap difference" /> Heatmap difference on current ({d.heatmap_diff?.finding})</label>
            <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-red" /> increased</span>
            <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-[#3b82f6]" /> decreased</span>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            {[d.prior_study, d.current_study].map((s, i) => (
              <Card key={s.id} className="overflow-hidden p-2">
                <div className="flex items-center justify-between px-2 pb-2 pt-1 text-xs">
                  <span className="font-medium text-fg">{i === 0 ? "Prior" : "Current"} · <span className="num text-muted">{s.study_date ?? s.created_at.slice(0, 10)}</span></span>
                  <UrgencyPill level={s.urgency_level} score={s.urgency_score} />
                </div>
                <div className="h-[48vh] min-h-[320px]">
                  <ImageViewer
                    src={s.image_url ?? ""}
                    width={s.width}
                    height={s.height}
                    view={sync ? view : i === 0 ? view : viewB}
                    onViewChange={sync ? setView : i === 0 ? setView : setViewB}
                    heatmap={i === 1 && diff ? d.heatmap_diff?.url : null}
                    heatOpacity={0.85}
                  />
                </div>
              </Card>
            ))}
          </div>
          <div className="grid gap-4 lg:grid-cols-[1fr_420px]">
            <Card className="overflow-hidden">
              <CardHeader title="Per-finding change" subtitle="Calibrated probabilities, prior → current. Click a row to map its heatmap difference." />
              <div className="overflow-x-auto">
                <table className="w-full min-w-[560px] text-sm">
                  <thead>
                    <tr className="border-y border-border text-left text-[11px] uppercase tracking-wider text-subtle">
                      <th className="px-4 py-2 font-medium">Finding</th>
                      <th className="px-4 py-2 font-medium">Prior</th>
                      <th className="px-4 py-2 font-medium" />
                      <th className="px-4 py-2 font-medium">Current</th>
                      <th className="px-4 py-2 font-medium">Δ</th>
                      <th className="px-4 py-2 font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.deltas.map((x) => (
                      <tr
                        key={x.label}
                        onClick={() => setFinding(x.key)}
                        className={cn("cursor-pointer border-b border-border/60 hover:bg-surface-2", d.heatmap_diff?.finding === x.key && "bg-cyan/[0.06]")}
                      >
                        <td className="px-4 py-2 text-fg">{x.label}</td>
                        <td className="num px-4 py-2 text-muted">{pct(x.prior, 1)}</td>
                        <td className="px-1 text-subtle"><ArrowRight className="size-3.5" /></td>
                        <td className="num px-4 py-2 text-fg">{pct(x.current, 1)}</td>
                        <td className={cn("num px-4 py-2", x.delta > 0.05 ? "text-red" : x.delta < -0.05 ? "text-teal" : "text-muted")}>
                          <span className="inline-flex items-center gap-1">
                            {x.delta > 0.05 ? <ArrowUpRight className="size-3.5" /> : x.delta < -0.05 ? <ArrowDownRight className="size-3.5" /> : <Minus className="size-3.5" />}
                            {x.delta > 0 ? "+" : ""}{fmt(x.delta * 100, 1)} pp
                          </span>
                        </td>
                        <td className="px-4 py-2"><Badge tone={STATUS_TONE[x.status as keyof typeof STATUS_TONE] ?? "neutral"}>{x.status}</Badge></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            <div className="space-y-4">
              <Card className="p-5">
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-cyan">Interval change (Copilot)</div>
                <p className="text-sm leading-relaxed text-fg">{d.summary}</p>
              </Card>
              {d.ctr && (
                <Card className="grid grid-cols-3 gap-2 p-5 text-center">
                  <div><div className="text-[11px] text-muted">CTR prior</div><div className="num text-xl font-semibold text-fg">{fmt(d.ctr.prior)}</div></div>
                  <div><div className="text-[11px] text-muted">CTR current</div><div className="num text-xl font-semibold text-fg">{fmt(d.ctr.current)}</div></div>
                  <div><div className="text-[11px] text-muted">Change</div><div className={cn("num text-xl font-semibold", Math.abs(d.ctr.delta) >= 0.03 ? "text-amber" : "text-teal")}>{d.ctr.delta > 0 ? "+" : ""}{fmt(d.ctr.delta)}</div></div>
                </Card>
              )}
              {d.current_study.patient_id && <PatientTimeline patientId={d.current_study.patient_id} />}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <React.Suspense fallback={<Skeleton className="h-[60vh]" />}>
      <Inner />
    </React.Suspense>
  );
}
