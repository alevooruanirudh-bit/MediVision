"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, BookOpenText, Database, Scale, Target, Users } from "lucide-react";

import { PageHeader } from "@/components/common";
import { Badge, Card, CardHeader, Skeleton } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { fmt } from "@/lib/utils";

export default function ModelCardPage() {
  const card = useQuery({ queryKey: ["model-card"], queryFn: api.modelCard });
  const metrics = useQuery({ queryKey: ["metrics"], queryFn: api.metrics });
  const c = card.data;
  return (
    <div className="max-w-6xl">
      <PageHeader
        eyebrow="Transparency"
        title="Model card"
        subtitle="What the models are, what they were trained on, what they are for — and what they must not be used for."
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader icon={<Target />} title="Intended use" />
          <div className="space-y-2 px-5 pb-5 text-sm leading-relaxed text-muted">
            <p><span className="text-fg">For:</span> research, teaching and portfolio demonstration of AI-assisted imaging workflows — explainability, triage, reporting, human–AI review, federated learning.</p>
            <p><span className="text-fg">Not for:</span> diagnosis, treatment decisions, screening or any clinical use. Not a medical device; not validated prospectively; no regulatory clearance.</p>
            <p><span className="text-fg">Users:</span> developers, students and researchers familiar with the limits of retrospective public datasets.</p>
          </div>
        </Card>
        <Card>
          <CardHeader icon={<AlertTriangle />} title="Limitations" />
          <ul className="list-disc space-y-1.5 px-5 pb-5 pl-10 text-sm leading-relaxed text-muted">
            <li>NIH ChestX-ray14 labels were mined from reports by NLP and are noisy (estimated label accuracy &gt;90% but not perfect).</li>
            <li>Starter models use 96–192 px inputs and a few epochs on a laptop; full models use 320–640 px on a Kaggle GPU.</li>
            <li>Single 2-D images only — no prior history, no lateral view, no full MRI volume.</li>
            <li>The brain MRI classifier knows three tumour types; other pathology (stroke, bleed, MS) is out of scope.</li>
            <li>Heatmaps are approximate and can highlight shortcuts (markers, devices). The CTR is an estimate from lung masks.</li>
            <li>Domain shift: different scanners, populations or paediatric images can degrade performance without warning beyond the OOD guard.</li>
          </ul>
        </Card>
        <Card>
          <CardHeader icon={<Scale />} title="Bias statement" />
          <div className="space-y-2 px-5 pb-5 text-sm leading-relaxed text-muted">
            <p>Public datasets under-represent many populations and are dominated by single institutions (e.g. NIH Clinical Center). We report AUROC by sex, age band and view (PA/AP) on patient-disjoint data in the Lab.</p>
            {c?.headline?.chest_ensemble && (
              <p>
                Current chest ensemble ({c.headline.chest_ensemble.profile}): macro AUROC{" "}
                <span className="num text-fg">{fmt(c.headline.chest_ensemble.macro_auroc, 3)}</span>, sex gap{" "}
                <span className="num text-fg">{fmt(c.headline.chest_ensemble.fairness_sex_gap, 3)}</span>.
              </p>
            )}
            <p>Small subgroup gaps on a sample are not proof of fairness; prospective, site-specific evaluation would be required for any real use.</p>
          </div>
        </Card>
        <Card>
          <CardHeader icon={<Users />} title="Human oversight" />
          <div className="space-y-2 px-5 pb-5 text-sm leading-relaxed text-muted">
            <p>Every finding can be accepted, rejected or edited; every action is in the audit log. Disagreements export as a re-training CSV. Reports are drafts until a named reviewer signs them.</p>
            <p>The Copilot never states a definitive diagnosis and always recommends clinical correlation. It is a self-built rule/retrieval engine; an optional local LLM plug-in is <Badge>off</Badge> by default.</p>
          </div>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader icon={<BookOpenText />} title="Triage weights" subtitle="score = 100 · (1 − Π(1 − wᵢ · evidenceᵢ · confidenceᵢ)) over positive findings" />
        {card.isLoading ? (
          <Skeleton className="m-5 h-32" />
        ) : (
          <div className="grid gap-4 px-5 pb-5 md:grid-cols-[1fr_220px]">
            <div className="flex flex-wrap gap-2">
              {Object.entries(c?.triage.weights ?? {}).sort((a, b) => b[1] - a[1]).map(([k, w]) => (
                <span key={k} className="inline-flex items-center gap-2 rounded-lg border border-border bg-surface-2 px-2.5 py-1 text-xs">
                  <span className="text-fg">{k}</span>
                  <span className="num text-muted">{w.toFixed(2)}</span>
                </span>
              ))}
            </div>
            <div className="space-y-1 text-xs text-muted">
              {Object.entries(c?.triage.levels ?? {}).map(([k, v]) => (
                <div key={k} className="flex justify-between"><span>{k}</span><span className="num text-fg">≥ {v}</span></div>
              ))}
            </div>
          </div>
        )}
      </Card>

      <Card className="mt-4">
        <CardHeader icon={<Target />} title="Per-label operating thresholds (chest)" subtitle="Maximise Youden's J on the validation split, after temperature scaling" />
        <div className="grid gap-2 px-5 pb-5 text-xs sm:grid-cols-3 lg:grid-cols-5">
          {Object.entries(c?.thresholds ?? {}).map(([k, v]) => (
            <div key={k} className="flex justify-between rounded-lg border border-border bg-surface-2 px-2.5 py-1.5">
              <span className="text-muted">{k.replace("_", " ")}</span>
              <span className="num text-fg">{v.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card className="mt-4 overflow-hidden">
        <CardHeader icon={<Database />} title="Datasets & licenses" subtitle="All free. Every model was trained by this project; only generic ImageNet/COCO initialisation was used." />
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-sm">
            <thead>
              <tr className="border-y border-border text-left text-[11px] uppercase tracking-wider text-subtle">
                <th className="px-5 py-2 font-medium">Dataset</th>
                <th className="px-5 py-2 font-medium">License</th>
                <th className="px-5 py-2 font-medium">Citation</th>
                <th className="px-5 py-2 font-medium">Profiles</th>
              </tr>
            </thead>
            <tbody>
              {metrics.data?.datasets.map((d) => (
                <tr key={d.key} className="border-b border-border/60 align-top">
                  <td className="px-5 py-2.5"><a href={d.url} target="_blank" rel="noreferrer" className="text-fg hover:text-cyan">{d.title}</a></td>
                  <td className="px-5 py-2.5 text-muted">{d.license}</td>
                  <td className="px-5 py-2.5 text-xs text-muted">{d.citation}</td>
                  <td className="px-5 py-2.5">{d.profiles.length ? d.profiles.map((p) => <Badge key={p} tone={p === "full" ? "violet" : "cyan"} className="mr-1">{p}</Badge>) : <Badge>optional</Badge>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
