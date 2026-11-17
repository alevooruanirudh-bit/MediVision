"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  Bot,
  Boxes,
  FileText,
  Flame,
  GitCompareArrows,
  Layers,
  ListChecks,
  Loader2,
  RefreshCcw,
  Ruler,
  Timer,
} from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import * as React from "react";
import { toast } from "sonner";

import { ModalityTag, ModelBadge, OODBanner, UrgencyPill } from "@/components/common";
import { CopilotChat } from "@/components/copilot/chat";
import { Badge, Button, Card, NativeSelect, Skeleton, Slider, Switch, Tabs, TabsContent, TabsList, TabsTrigger, Tip } from "@/components/ui/primitives";
import { FindingsPanel, MeasurementsPanel } from "@/components/viewer/panels";
import { type Box, ImageViewer, type Line } from "@/components/viewer/image-viewer";
import { api } from "@/lib/api";
import { type HeatMethod, useViewer } from "@/lib/store";
import type { ChatAction } from "@/lib/types";
import { ms, pct } from "@/lib/utils";

export default function StudyPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const qc = useQueryClient();
  const v = useViewer();
  const q = useQuery({ queryKey: ["study", id], queryFn: () => api.study(id) });
  const study = q.data;
  const res = study?.results;
  const [reanalysing, setReanalysing] = React.useState(false);

  // default selection: top positive finding
  React.useEffect(() => {
    if (res?.findings?.length && (!v.selected || !res.findings.some((f) => f.key === v.selected))) {
      v.set({ selected: (res.findings.find((f) => f.positive) ?? res.findings[0]).key });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [res]);

  const cached = v.selected ? res?.overlays?.heatmaps?.[v.selected]?.[v.method] : undefined;
  const needFetch = !!v.selected && !!res?.pipeline && (v.method !== "gradcam" || v.cmap !== "inferno" || !cached);
  const heat = useQuery({
    queryKey: ["explain", id, v.selected, v.method, v.cmap],
    queryFn: () => api.explain(id, v.selected!, v.method, v.cmap),
    enabled: needFetch && v.heatmap,
    staleTime: Infinity,
  });
  const heatUrl = v.heatmap ? (needFetch ? heat.data?.url : cached) : null;

  // keyboard shortcuts for overlays / tabs
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || e.metaKey || e.ctrlKey) return;
      const map: Record<string, () => void> = {
        h: () => v.toggle("heatmap"),
        m: () => v.toggle("mask"),
        b: () => (v.toggle("boxes"), v.toggle("lines")),
        i: () => v.toggle("invert"),
        "1": () => v.set({ tab: "findings" }),
        "2": () => v.set({ tab: "measurements" }),
        "3": () => v.set({ tab: "report" }),
        "4": () => v.set({ tab: "copilot" }),
      };
      map[e.key]?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [v]);

  const onAction = (a: ChatAction) => {
    if (a.type === "show_heatmap") v.set({ selected: a.finding ?? v.selected, method: (a.method as HeatMethod) ?? "gradcam", heatmap: true });
    else if (a.type === "open_tab") v.set({ tab: a.tab ?? "findings" });
    else if (a.type === "toggle_overlay") v.toggle("mask");
    else if (a.type === "open_compare") router.push(`/compare?a=${a.a}&b=${a.b}`);
    else if (a.type === "navigate" && a.href) router.push(a.href);
    else if (a.type === "rewrite_report" && study?.report_id)
      api.rewriteReport(study.report_id, a.style ?? "concise", a.section).then(() => {
        toast.success(`Report ${a.section ?? ""} rewritten (${a.style})`);
        router.push(`/studies/${id}/report`);
      });
  };

  const reanalyse = async (force = false) => {
    setReanalysing(true);
    try {
      await api.analyze(id, force);
      await qc.invalidateQueries({ queryKey: ["study", id] });
      qc.removeQueries({ queryKey: ["explain", id] });
      toast.success("Re-analysed with the current models");
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setReanalysing(false);
    }
  };

  if (q.isLoading || !study || !res)
    return (
      <div className="grid gap-4 lg:grid-cols-[1fr_420px]">
        <Skeleton className="h-[70vh]" />
        <Skeleton className="h-[70vh]" />
      </div>
    );

  const sel = res.findings.find((f) => f.key === v.selected);
  const boxes: Box[] = [];
  if (v.boxes) {
    res.detections?.forEach((d) => boxes.push({ box: d.box, label: `${d.label} ${d.score.toFixed(2)}`, color: "#f59e0b", labelBottom: true }));
    if (sel?.heat_box && v.heatmap) boxes.push({ box: sel.heat_box, label: `${sel.label}${sel.zone ? ` · ${sel.zone}` : ""}`, dashed: true });
    if (res.pipeline === "brain" && res.measurements?.box) boxes.push({ box: res.measurements.box, label: "segmented lesion", color: "#a78bfa", labelBottom: true });
  }
  const lines: Line[] =
    v.lines && res.measurements?.lines
      ? [
          { points: res.measurements.lines.thorax, label: "thorax", color: "#22D3EE" },
          { points: res.measurements.lines.heart, label: `heart · CTR ${res.measurements.ctr?.toFixed(2)}`, color: "#f59e0b" },
        ]
      : [];
  const mask = v.mask ? (res.overlays.lung_mask ?? res.overlays.tumour_mask) : null;
  const t = res.timings_ms ?? {};

  return (
    <div className="space-y-4">
      {/* top bar */}
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="ghost" size="icon" asChild aria-label="Back to worklist">
          <Link href="/worklist"><ArrowLeft /></Link>
        </Button>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="num text-lg font-semibold text-fg">{study.patient?.pseudonym}</h1>
            <UrgencyPill level={study.urgency_level} score={study.urgency_score} />
            <ModelBadge profile={res.model_profile} />
            {study.format === "dcm" && <Badge tone="teal">DICOM de-identified · {study.deid?.removed_tags?.length ?? 0} PHI tags removed</Badge>}
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-3 text-xs text-muted">
            <ModalityTag modality={study.modality} />
            {res.modality.confidence != null && <span className="num">router {pct(res.modality.confidence)}</span>}
            {study.patient?.sex && <span>{study.patient.sex}{study.patient.age ? ` · ${study.patient.age}y` : ""}</span>}
            {study.indication && <span className="hidden md:inline">Indication: {study.indication}</span>}
          </div>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {study.other_studies.length > 0 && (
            <Button variant="secondary" size="sm" asChild>
              <Link href={`/compare?a=${study.other_studies[0].id}&b=${study.id}`}><GitCompareArrows /> Compare with prior</Link>
            </Button>
          )}
          <Button variant="secondary" size="sm" asChild>
            <Link href={`/studies/${id}/report`}><FileText /> Report editor</Link>
          </Button>
          <Tip content="Re-run the full pipeline">
            <Button variant="ghost" size="icon" onClick={() => reanalyse()} aria-label="Re-analyse">
              {reanalysing ? <Loader2 className="animate-spin" /> : <RefreshCcw />}
            </Button>
          </Tip>
        </div>
      </div>
      <OODBanner message={res.modality.warning} />
      {res.pipeline == null && (
        <Card className="flex flex-wrap items-center justify-between gap-3 p-4 text-sm text-muted">
          The modality router did not send this image to any pipeline, so no findings were produced.
          <Button variant="outline" size="sm" onClick={() => reanalyse(true)}>Analyse anyway (not recommended)</Button>
        </Card>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_440px]">
        {/* viewer */}
        <Card className="flex flex-col overflow-hidden p-2">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-2 pb-2 pt-1 text-xs text-muted">
            <label className="flex items-center gap-2"><Switch checked={v.heatmap} onCheckedChange={() => v.toggle("heatmap")} aria-label="Heatmap" /><Flame className="size-3.5" /> Heatmap</label>
            <div className="flex w-28 items-center gap-2">
              <Slider value={[v.opacity * 100]} onValueChange={([x]) => v.set({ opacity: x / 100 })} max={100} step={1} aria-label="Heatmap opacity" />
            </div>
            <NativeSelect value={v.method} onChange={(e) => v.set({ method: e.target.value as HeatMethod })} className="h-7 text-xs" aria-label="Explanation method">
              <option value="gradcam">Grad-CAM</option>
              <option value="gradcampp">Grad-CAM++</option>
              <option value="occlusion">Occlusion</option>
            </NativeSelect>
            <NativeSelect value={v.cmap} onChange={(e) => v.set({ cmap: e.target.value })} className="h-7 text-xs" aria-label="Colormap">
              {["inferno", "jet", "turbo", "viridis", "magma"].map((c) => <option key={c}>{c}</option>)}
            </NativeSelect>
            <label className="flex items-center gap-2"><Switch checked={v.mask} onCheckedChange={() => v.toggle("mask")} aria-label="Segmentation mask" /><Layers className="size-3.5" /> {res.pipeline === "brain" ? "Tumour mask" : "Lung masks"}</label>
            <label className="flex items-center gap-2"><Switch checked={v.boxes} onCheckedChange={() => v.toggle("boxes")} aria-label="Boxes" /><Boxes className="size-3.5" /> Boxes</label>
            {res.measurements?.lines && (
              <label className="flex items-center gap-2"><Switch checked={v.lines} onCheckedChange={() => v.toggle("lines")} aria-label="CTR lines" /><Ruler className="size-3.5" /> CTR lines</label>
            )}
            {heat.isFetching && <span className="flex items-center gap-1 text-cyan"><Loader2 className="size-3.5 animate-spin" /> computing {v.method}…</span>}
          </div>
          <div className="relative h-[62vh] min-h-[380px] xl:h-[calc(100dvh-260px)]">
            <ImageViewer
              src={study.image_url ?? ""}
              width={study.width}
              height={study.height}
              heatmap={res.pipeline ? heatUrl : null}
              heatOpacity={v.opacity}
              mask={mask}
              boxes={boxes}
              lines={lines}
              invert={v.invert}
              onToggleInvert={() => v.toggle("invert")}
              scanning={reanalysing || heat.isFetching}
              label={`${study.modality} image of ${study.patient?.pseudonym}`}
            />
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-2 pt-2 text-[11px] text-subtle">
            <Timer className="size-3.5" />
            {Object.entries(t)
              .filter(([k]) => k !== "total")
              .map(([k, val]) => (
                <span key={k} className="num">{k} {ms(val)}</span>
              ))}
            <span className="num ml-auto text-muted">total {ms(t.total)}</span>
          </div>
        </Card>

        {/* side panel */}
        <Card className="flex min-h-[560px] flex-col overflow-hidden xl:h-[calc(100dvh-160px)]">
          <Tabs value={v.tab} onValueChange={(tab) => v.set({ tab })} className="flex min-h-0 flex-1 flex-col">
            <div className="border-b border-border p-3">
              <TabsList className="w-full justify-between">
                <TabsTrigger value="findings" className="flex-1"><ListChecks /> Findings</TabsTrigger>
                <TabsTrigger value="measurements" className="flex-1"><Ruler /> Measure</TabsTrigger>
                <TabsTrigger value="report" className="flex-1"><FileText /> Report</TabsTrigger>
                <TabsTrigger value="copilot" className="flex-1" data-testid="tab-copilot"><Bot /> Copilot</TabsTrigger>
              </TabsList>
            </div>
            <TabsContent value="findings" className="scrollbar-thin min-h-0 flex-1 overflow-y-auto p-3">
              <FindingsPanel
                results={res}
                selected={v.selected}
                onSelect={(key) => v.set({ selected: key, heatmap: true })}
                onReviewed={() => qc.invalidateQueries({ queryKey: ["study", id] })}
              />
              <div className="mt-4 rounded-xl border border-border bg-surface-2 p-3 text-xs text-muted">
                <div className="mb-1 font-medium text-fg">Triage · {res.triage.level} ({Math.round(res.triage.score)}/100)</div>
                {res.triage.drivers.length ? res.triage.drivers.map((d) => (
                  <div key={d.label} className="num flex justify-between"><span>{d.label}</span><span>+{d.contribution}</span></div>
                )) : "No finding above threshold."}
                {res.extra?.chest && (
                  <div className="num mt-2 border-t border-border pt-2 text-[11px]">
                    OOD energy {res.extra.chest.energy.toFixed(2)} (cut-off {res.extra.chest.energy_threshold?.toFixed(2)}) · {res.extra.chest.members.length} ensemble members
                  </div>
                )}
              </div>
            </TabsContent>
            <TabsContent value="measurements" className="scrollbar-thin min-h-0 flex-1 overflow-y-auto p-3">
              <MeasurementsPanel results={res} />
            </TabsContent>
            <TabsContent value="report" className="scrollbar-thin min-h-0 flex-1 overflow-y-auto p-4">
              <ReportPreview studyId={id} />
            </TabsContent>
            <TabsContent value="copilot" className="flex min-h-0 flex-1 flex-col p-3">
              <CopilotChat studyId={id} onAction={onAction} />
            </TabsContent>
          </Tabs>
        </Card>
      </div>
    </div>
  );
}

function ReportPreview({ studyId }: { studyId: string }) {
  const q = useQuery({ queryKey: ["report", studyId], queryFn: () => api.reportByStudy(studyId) });
  if (q.isLoading) return <Skeleton className="h-60" />;
  if (!q.data) return <p className="text-sm text-muted">No report yet.</p>;
  const s = q.data.sections;
  return (
    <div className="space-y-4 text-[13px] leading-relaxed">
      <div className="flex items-center gap-2">
        <Badge tone={q.data.status === "signed" ? "teal" : "amber"}>{q.data.status}</Badge>
        <Badge tone={q.data.engine === "experimental" ? "violet" : "cyan"}>{q.data.engine === "experimental" ? "Experimental engine" : "Rule-based engine"}</Badge>
        <span className="num text-[11px] text-subtle">v{q.data.version}</span>
      </div>
      <div>
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-subtle">Findings</div>
        {Object.entries(s.findings ?? {}).map(([k, val]) => (
          <p key={k} className="text-fg"><span className="text-muted">{k}: </span>{val}</p>
        ))}
      </div>
      <div>
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-subtle">Impression</div>
        {s.impression?.map((x) => <p key={x} className="font-medium text-fg">{x}</p>)}
      </div>
      <Button asChild variant="primary" className="w-full">
        <Link href={`/studies/${studyId}/report`}><FileText /> Open report editor</Link>
      </Button>
    </div>
  );
}
