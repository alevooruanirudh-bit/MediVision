"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Braces, Download, FileSignature, FlaskConical, Loader2, Lock, RotateCcw, Save, Sparkles } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import * as React from "react";
import { toast } from "sonner";

import { DISCLAIMER, ModelBadge, PageHeader, UrgencyPill } from "@/components/common";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Badge, Button, Card, Input, Skeleton, Tabs, TabsList, TabsTrigger, Textarea, Tip } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { useSettings } from "@/lib/store";
import type { Report, ReportSections } from "@/lib/types";
import { cn } from "@/lib/utils";

const SECTIONS: { key: keyof ReportSections; title: string }[] = [
  { key: "indication", title: "Indication" },
  { key: "technique", title: "Technique" },
  { key: "findings", title: "Findings" },
  { key: "measurements", title: "Measurements" },
  { key: "impression", title: "Impression" },
  { key: "recommendations", title: "Recommendations" },
  { key: "ai_confidence", title: "AI confidence & limitations" },
];
const STYLES = [
  ["standard", "Standard"],
  ["concise", "Concise"],
  ["detailed", "Detailed"],
  ["patient", "Patient-friendly"],
] as const;

export default function ReportEditor() {
  const { id } = useParams<{ id: string }>();
  const qc = useQueryClient();
  const study = useQuery({ queryKey: ["study", id], queryFn: () => api.study(id) });
  const rq = useQuery({ queryKey: ["report", id], queryFn: () => api.reportByStudy(id) });
  const [draft, setDraft] = React.useState<ReportSections | null>(null);
  const [busy, setBusy] = React.useState<string | null>(null);
  const [signOpen, setSignOpen] = React.useState(false);
  const [fhir, setFhir] = React.useState<Record<string, unknown> | null>(null);
  const reviewer = useSettings((s) => s.reviewer);
  const [signer, setSigner] = React.useState(reviewer);
  const report = rq.data;
  React.useEffect(() => {
    if (report) setDraft(report.sections);
  }, [report]);

  const apply = (r: Report) => {
    qc.setQueryData(["report", id], r);
    setDraft(r.sections);
  };
  const run = async (label: string, fn: () => Promise<Report>, ok: string) => {
    setBusy(label);
    try {
      apply(await fn());
      toast.success(ok);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(null);
    }
  };
  if (rq.isLoading || !report || !draft) return <Skeleton className="h-[70vh]" />;
  const signed = report.status === "signed";
  const dirty = JSON.stringify(draft) !== JSON.stringify(report.sections);
  const setSec = (k: keyof ReportSections, v: unknown) => setDraft((d) => ({ ...(d ?? {}), [k]: v }));

  return (
    <div>
      <PageHeader
        eyebrow={
          <Link href={`/studies/${id}`} className="inline-flex items-center gap-1 hover:underline">
            <ArrowLeft className="size-3" /> Back to viewer
          </Link>
        }
        title="Report editor"
        subtitle="AI-drafted by the MediVision Copilot engine. Edit any section, regenerate it, change the style, then sign off."
        actions={
          <>
            <Button variant="secondary" disabled={!dirty || signed || !!busy} onClick={() => run("save", () => api.updateReport(report.id, draft), "Report saved")}>
              <Save /> Save
            </Button>
            <Button variant="secondary" asChild>
              <a href={api.pdfUrl(report.id)} target="_blank" rel="noreferrer" data-testid="pdf-link"><Download /> PDF</a>
            </Button>
            <Button variant="secondary" onClick={async () => setFhir(await api.fhir(report.id))}><Braces /> FHIR</Button>
            <Button variant="primary" disabled={signed} onClick={() => setSignOpen(true)}>
              {signed ? <Lock /> : <FileSignature />} {signed ? `Signed by ${report.signed_by}` : "Sign off"}
            </Button>
          </>
        }
      />
      <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
        <Card className="p-5 md:p-7">
          <div className="mb-5 flex flex-wrap items-center gap-3 border-b border-border pb-4">
            <span className="num text-sm text-fg">{study.data?.patient?.pseudonym}</span>
            <UrgencyPill level={study.data?.urgency_level ?? null} score={study.data?.urgency_score} />
            <ModelBadge profile={study.data?.model_profile} />
            <Badge tone={report.engine === "experimental" ? "violet" : "cyan"}>{report.engine === "experimental" ? "Experimental engine" : "Rule-based engine"}</Badge>
            <Badge tone={signed ? "teal" : "amber"}>{signed ? "signed" : "draft"} · v{report.version}</Badge>
          </div>
          {report.engine === "experimental" && (
            <div className="mb-5 rounded-xl border border-violet/30 bg-violet/10 p-3 text-xs text-violet">
              Experimental: free text from our small self-trained Transformer decoder. It may hallucinate — the rule-based report remains the default.
            </div>
          )}
          <div className="space-y-6">
            {SECTIONS.map(({ key, title }) => {
              const val = draft[key];
              if (val == null) return null;
              return (
                <section key={key} className="group">
                  <div className="mb-2 flex items-center justify-between">
                    <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-cyan">{title}</h2>
                    {!signed && report.engine === "rules" && (
                      <Tip content="Regenerate this section from the model outputs">
                        <button
                          className="flex items-center gap-1 text-[11px] text-subtle opacity-0 transition hover:text-fg group-hover:opacity-100"
                          onClick={() => run(key, () => api.rewriteReport(report.id, "standard", key), `${title} regenerated`)}
                        >
                          {busy === key ? <Loader2 className="size-3 animate-spin" /> : <RotateCcw className="size-3" />} regenerate
                        </button>
                      </Tip>
                    )}
                  </div>
                  {typeof val === "string" ? (
                    <Textarea disabled={signed} value={val} rows={2} onChange={(e) => setSec(key, e.target.value)} aria-label={title} />
                  ) : Array.isArray(val) ? (
                    <Textarea
                      disabled={signed}
                      value={val.join("\n")}
                      rows={Math.max(2, val.length + 1)}
                      onChange={(e) => setSec(key, e.target.value.split("\n"))}
                      aria-label={title}
                      className={cn(key === "impression" && "font-medium")}
                    />
                  ) : (
                    <div className="space-y-2">
                      {Object.entries(val as Record<string, string>).map(([sk, sv]) => (
                        <div key={sk} className="grid gap-1 md:grid-cols-[170px_1fr] md:gap-3">
                          <span className="pt-2 text-xs text-muted">{sk}</span>
                          <Textarea
                            disabled={signed}
                            value={sv}
                            rows={2}
                            onChange={(e) => setSec(key, { ...(val as Record<string, string>), [sk]: e.target.value })}
                            aria-label={`${title}: ${sk}`}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              );
            })}
            <p className="rounded-xl border border-amber/30 bg-amber/10 p-3 text-xs text-amber">{draft.disclaimer ?? DISCLAIMER}</p>
          </div>
        </Card>
        <div className="space-y-4">
          <Card className="p-4">
            <div className="mb-2 flex items-center gap-2 text-sm font-medium text-fg"><Sparkles className="size-4 text-cyan" /> Style</div>
            <Tabs value={report.style} onValueChange={(s) => run("style", () => api.rewriteReport(report.id, s), `Rewritten: ${s}`)}>
              <TabsList className="grid w-full grid-cols-2 gap-1">
                {STYLES.map(([k, l]) => <TabsTrigger key={k} value={k} disabled={signed}>{l}</TabsTrigger>)}
              </TabsList>
            </Tabs>
            <p className="mt-2 text-[11px] text-subtle">Rule-based rewriting with a lay-term glossary — deterministic, no LLM.</p>
          </Card>
          <Card className="p-4">
            <div className="mb-2 flex items-center gap-2 text-sm font-medium text-fg"><FlaskConical className="size-4 text-violet" /> Engine</div>
            <div className="grid grid-cols-2 gap-2">
              <Button variant={report.engine === "rules" ? "primary" : "secondary"} size="sm" disabled={signed || !!busy} onClick={() => run("engine", () => api.generateReport(id, "rules"), "Rule-based draft generated")}>
                Rules
              </Button>
              <Button
                variant={report.engine === "experimental" ? "primary" : "secondary"}
                size="sm"
                disabled={signed || !!busy || study.data?.pipeline !== "chest"}
                onClick={() => run("engine", () => api.generateReport(id, "experimental"), "Experimental draft generated")}
              >
                {busy === "engine" ? <Loader2 className="animate-spin" /> : null} Experimental
              </Button>
            </div>
            <p className="mt-2 text-[11px] text-subtle">Experimental = our CNN + Transformer decoder trained from scratch on IU X-ray reports (chest only).</p>
          </Card>
          <Card className="p-4 text-xs text-muted">
            <div className="mb-1 font-medium text-fg">Exports</div>
            PDF (reportlab) and an HL7 FHIR R4-style DiagnosticReport with one Observation per AI finding, tagged “ai-generated-draft”.
          </Card>
        </div>
      </div>
      <Dialog open={signOpen} onOpenChange={setSignOpen}>
        <DialogContent title="Sign off report" description="Signing makes the report read-only and records it in the audit log.">
          <Input value={signer} onChange={(e) => setSigner(e.target.value)} aria-label="Signer name" />
          <div className="mt-4 flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setSignOpen(false)}>Cancel</Button>
            <Button
              variant="primary"
              onClick={async () => {
                if (dirty) await api.updateReport(report.id, draft);
                await run("sign", () => api.signReport(report.id, signer), "Report signed");
                setSignOpen(false);
              }}
            >
              <FileSignature /> Sign
            </Button>
          </div>
        </DialogContent>
      </Dialog>
      <Dialog open={!!fhir} onOpenChange={(o) => !o && setFhir(null)}>
        <DialogContent title="FHIR DiagnosticReport" description="HL7 FHIR R4-style JSON" className="w-[min(94vw,760px)]">
          <pre className="num scrollbar-thin max-h-[60vh] overflow-auto rounded-xl border border-border bg-surface-2 p-3 text-[11px] text-fg">{JSON.stringify(fhir, null, 2)}</pre>
          <div className="mt-3 flex justify-end">
            <Button
              variant="secondary"
              onClick={() => {
                const blob = new Blob([JSON.stringify(fhir, null, 2)], { type: "application/fhir+json" });
                const a = document.createElement("a");
                a.href = URL.createObjectURL(blob);
                a.download = `diagnosticreport_${id}.json`;
                a.click();
              }}
            >
              <Download /> Download JSON
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
