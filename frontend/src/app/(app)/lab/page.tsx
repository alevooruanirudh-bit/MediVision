"use client";

import { useQuery } from "@tanstack/react-query";
import { FlaskConical } from "lucide-react";
import * as React from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { ChartTooltip, Legend, SERIES, axis, grid } from "@/components/charts";
import { EmptyState, PageHeader, SectionTitle } from "@/components/common";
import { Badge, Card, CardHeader, NativeSelect, Skeleton, Tabs, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { type AnyJson, api } from "@/lib/api";
import { cn, fmt, pct, titleCase } from "@/lib/utils";

const NAMES: Record<string, string> = {
  second_reader: "★ Second-reader summary (all models)",
  chest_ensemble: "Chest X-ray ensemble (14 labels)",
  chest_densenet121: "· DenseNet-121 member",
  chest_efficientnet_b2: "· EfficientNet-B2 member",
  chest_convnext_tiny: "· ConvNeXt-Tiny member",
  lung_unet: "Lung segmentation U-Net",
  fracture_cls: "Fracture classifier",
  fracture_det: "Fracture detector (Faster R-CNN)",
  brain_cls: "Brain MRI classifier",
  brain_seg: "Brain tumour Attention U-Net",
  router: "Modality router",
  denoiser: "Low-dose denoiser (DnCNN)",
  report_gen: "Experimental report generator",
  copilot_intent: "Copilot intent model",
  pneumonia_det: "Pneumonia localiser (RSNA)",
  head_ct_cls: "Head-CT hemorrhage classifier",
  nodule3d: "Chest-CT nodule malignancy (3-D CNN)",
  breast_us: "Breast-ultrasound malignancy",
  retina_dr: "Diabetic-retinopathy grader (fundus)",
  sr_gan: "Super-resolution ×2 (SR-GAN)",
  radiogenomics: "Radiogenomics + seg. fairness (LGG/TCGA)",
};

const BINARY = ["fracture_cls", "head_ct_cls", "nodule3d", "breast_us"];

/** One row per model at its operating point: sensitivity / specificity / FPR / FNR (M7). */
function SecondReader({ models, profile }: { models: Record<string, Record<string, AnyJson>>; profile: string }) {
  const rows: { model: string; target: string; auroc?: number; se?: number; sp?: number; ppv?: number; npv?: number; n?: number }[] = [];
  const get = (n: string) => models[n]?.[profile] ?? models[n]?.starter;
  const ce = get("chest_ensemble");
  if (ce?.test?.per_label)
    Object.entries(ce.test.per_label as Record<string, AnyJson>).forEach(([lab, v]) =>
      rows.push({ model: "Chest ensemble", target: lab.replace("_", " "), auroc: v.auroc, se: v.sensitivity, sp: v.specificity, ppv: v.ppv, npv: v.npv, n: ce.n_test }),
    );
  BINARY.forEach((n) => {
    const t = get(n)?.test;
    if (t) rows.push({ model: NAMES[n], target: get(n)?.config?.label_name ?? "positive", auroc: t.auroc, se: t.sensitivity, sp: t.specificity, ppv: t.ppv, npv: t.npv, n: get(n)?.n_test });
  });
  const r = get("retina_dr")?.test;
  if (r?.referable) rows.push({ model: NAMES.retina_dr, target: "Referable DR (grade ≥ 2)", auroc: r.referable_auroc, se: r.referable.sensitivity, sp: r.referable.specificity, ppv: r.referable.ppv, npv: r.referable.npv, n: get("retina_dr")?.n_test });
  ["brain_cls", "router"].forEach((n) => {
    const t = get(n)?.test;
    Object.entries((t?.per_class_reader ?? {}) as Record<string, AnyJson>).forEach(([c, v]) =>
      rows.push({ model: NAMES[n], target: titleCase(c), auroc: t.per_class_auroc?.[c], se: v.sensitivity, sp: v.specificity, n: get(n)?.n_test }),
    );
  });
  return (
    <Card className="overflow-hidden" data-testid="second-reader">
      <CardHeader
        title="AI as a second reader — error rates at the operating point"
        subtitle="False-positive rate = 1 − specificity (extra work for the radiologist); false-negative rate = 1 − sensitivity (missed findings). Thresholds: Youden's J on validation."
      />
      <div className="scrollbar-thin max-h-[70vh] overflow-auto">
        <table className="w-full min-w-[820px] text-sm">
          <thead className="sticky top-0 bg-surface">
            <tr className="border-y border-border text-left text-[11px] uppercase tracking-wider text-subtle">
              {["Model", "Target", "AUROC", "Sensitivity", "Specificity", "FPR", "FNR", "PPV", "NPV", "n test"].map((h) => (
                <th key={h} className={cn("px-3 py-2 font-medium", h !== "Model" && h !== "Target" && "text-right")}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((x, i) => (
              <tr key={i} className="border-b border-border/50">
                <td className="px-3 py-1.5 text-muted">{x.model}</td>
                <td className="px-3 py-1.5 text-fg">{x.target}</td>
                <td className="num px-3 py-1.5 text-right text-fg">{fmt(x.auroc, 3)}</td>
                <td className="num px-3 py-1.5 text-right">{pct(x.se, 1)}</td>
                <td className="num px-3 py-1.5 text-right">{pct(x.sp, 1)}</td>
                <td className="num px-3 py-1.5 text-right text-amber">{x.sp != null ? pct(1 - x.sp, 1) : "—"}</td>
                <td className="num px-3 py-1.5 text-right text-red">{x.se != null ? pct(1 - x.se, 1) : "—"}</td>
                <td className="num px-3 py-1.5 text-right text-muted">{x.ppv != null ? pct(x.ppv, 1) : "—"}</td>
                <td className="num px-3 py-1.5 text-right text-muted">{x.npv != null ? pct(x.npv, 1) : "—"}</td>
                <td className="num px-3 py-1.5 text-right text-subtle">{x.n ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="px-5 py-3 text-xs text-subtle">
        Small test sets (e.g. head CT n = 40) give wide confidence intervals; these are starter-profile numbers on public data, not
        clinical performance.
      </p>
    </Card>
  );
}

function BinarySection({ m }: { m: AnyJson }) {
  const t = m.test ?? {};
  const roc = (t.roc ?? []).map(([x, y]: number[]) => ({ fpr: x, tpr: y }));
  return (
    <div className="space-y-4">
      <Tiles
        items={[
          ["AUROC", fmt(t.auroc, 3)],
          ["Sensitivity", pct(t.sensitivity, 1)],
          ["Specificity", pct(t.specificity, 1)],
          ["Threshold", fmt(t.threshold, 3)],
          ["False-positive rate", pct(t.fpr ?? 1 - t.specificity, 1)],
          ["False-negative rate", pct(t.fnr ?? 1 - t.sensitivity, 1)],
          ["PPV / NPV", t.ppv != null ? `${pct(t.ppv, 0)} / ${pct(t.npv, 0)}` : "—"],
          ["TP · FP · FN · TN", t.tp != null ? `${t.tp} · ${t.fp} · ${t.fn} · ${t.tn}` : `n=${m.n_test}`],
        ]}
      />
      {roc.length > 0 && (
        <Card className="p-4">
          <SectionTitle>ROC (test)</SectionTitle>
          <div className="h-64">
            <ResponsiveContainer>
              <LineChart data={roc} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid {...grid} />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} {...axis} tickFormatter={(v) => v.toFixed(1)} />
                <YAxis domain={[0, 1]} {...axis} tickFormatter={(v) => v.toFixed(1)} width={36} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--border-strong)" strokeDasharray="4 4" />
                <Tooltip content={<ChartTooltip format={(v) => Number(v).toFixed(3)} labelFormat={(l) => `FPR ${Number(l).toFixed(2)}`} />} />
                <Line dataKey="tpr" name="TPR" stroke={SERIES[0]} strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
      <Card className="p-4"><History hist={m.history} /></Card>
    </div>
  );
}

function RadiogenomicsSection({ m }: { m: AnyJson }) {
  const targets = Object.entries((m.test?.targets ?? {}) as Record<string, AnyJson>);
  const fair = m.fairness_brain_seg ?? {};
  return (
    <div className="space-y-4">
      <Card className="overflow-hidden">
        <CardHeader title="Imaging features ↔ TCGA molecular subtypes" subtitle={`${m.n_patients} LGG patients · ${m.method}`} />
        <table className="w-full text-sm">
          <thead>
            <tr className="border-y border-border text-left text-[11px] uppercase tracking-wider text-subtle">
              <th className="px-5 py-2 font-medium">Target</th><th className="px-3 py-2 text-right font-medium">n (pos)</th>
              <th className="px-3 py-2 text-right font-medium">CV AUROC</th><th className="px-3 py-2 text-right font-medium">Permutation p95</th>
              <th className="px-5 py-2 text-right font-medium">Above chance</th>
            </tr>
          </thead>
          <tbody>
            {targets.map(([k, v]) => (
              <tr key={k} className="border-b border-border/60">
                <td className="px-5 py-2 text-fg">{v.description}</td>
                <td className="num px-3 py-2 text-right text-muted">{v.n} ({v.n_positive})</td>
                <td className="num px-3 py-2 text-right text-fg">{fmt(v.cv_auroc, 3)} ± {fmt(v.cv_auroc_sd, 3)}</td>
                <td className="num px-3 py-2 text-right text-muted">{fmt(v.permutation_auroc_p95, 3)}</td>
                <td className="px-5 py-2 text-right">{v.above_chance ? <Badge tone="teal">yes</Badge> : <Badge>no</Badge>}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="px-5 py-3 text-xs text-subtle">{m.note}</p>
      </Card>
      {fair.groups && (
        <Card className="overflow-hidden">
          <CardHeader title="Fairness: tumour-segmentation Dice by subgroup" subtitle={`${fair.n_patients} test patients · overall Dice ${fmt(fair.overall_dice, 3)}`} />
          <div className="grid gap-2 px-5 pb-3 text-xs sm:grid-cols-3">
            {Object.entries(fair.groups as Record<string, AnyJson>).map(([g, v]) => (
              <div key={g} className="flex justify-between rounded-lg border border-border bg-surface-2 px-3 py-1.5">
                <span className="text-muted">{g.replace(":", " · ")}</span>
                <span className="num text-fg">{fmt(v.dice, 3)} <span className="text-subtle">n={v.n}</span></span>
              </div>
            ))}
          </div>
          <p className="px-5 pb-4 text-xs text-subtle">{fair.note}</p>
        </Card>
      )}
    </div>
  );
}

function headline(name: string, m: AnyJson): [string, string] {
  const t = m?.test ?? {};
  if (name === "chest_ensemble") return ["macro AUROC", fmt(t.macro_auroc, 3)];
  if (name.startsWith("chest_")) return ["macro AUROC", fmt(t.macro_auroc, 3)];
  if (name === "fracture_cls") return ["AUROC", fmt(t.auroc, 3)];
  if (name.endsWith("_det")) return ["mAP@0.5", fmt(t.map50, 3)];
  if (name === "lung_unet" || name === "brain_seg") return ["Dice", fmt(t.dice_positive_slices ?? t.dice, 3)];
  if (name === "denoiser") return ["PSNR gain", `${fmt((t.psnr_denoised ?? 0) - (t.psnr_noisy ?? 0), 1)} dB`];
  if (name === "report_gen") return ["BLEU-4", fmt(t.bleu4, 3)];
  if (name === "copilot_intent") return ["accuracy", pct(t.accuracy, 1)];
  if (BINARY.includes(name)) return ["AUROC", fmt(t.auroc, 3)];
  if (name === "retina_dr") return ["referable AUROC", fmt(t.referable_auroc, 3)];
  if (name === "sr_gan") return ["PSNR vs bicubic", `${fmt((t.psnr_ours ?? 0) - (t.psnr_bicubic ?? 0), 2)} dB`];
  if (name === "radiogenomics")
    return ["above chance", `${Object.values(t.targets ?? {}).filter((x: AnyJson) => x.above_chance).length}/${Object.keys(t.targets ?? {}).length}`];
  return ["macro-F1", fmt(t.macro_f1, 3)];
}

function ConfusionMatrix({ cm, classes }: { cm: number[][]; classes: string[] }) {
  const max = Math.max(...cm.flat());
  return (
    <div className="overflow-x-auto">
      <table className="mx-auto text-[11px]">
        <thead>
          <tr>
            <th className="p-1 text-right text-subtle">true ↓ / pred →</th>
            {classes.map((c) => <th key={c} className="p-1 font-medium text-muted">{titleCase(c)}</th>)}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i}>
              <td className="p-1 pr-2 text-right font-medium text-muted">{titleCase(classes[i])}</td>
              {row.map((v, j) => (
                <td key={j} className="p-0.5">
                  <div
                    className={cn("num grid h-10 min-w-14 place-items-center rounded-md", v / max > 0.5 ? "text-white" : "text-fg")}
                    style={{ background: `color-mix(in srgb, var(--s1) ${Math.round((v / max) * 90) + 4}%, var(--surface-2))` }}
                    title={`${classes[i]} → ${classes[j]}: ${v}`}
                  >
                    {v}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Tiles({ items }: { items: [string, React.ReactNode][] }) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      {items.map(([k, v]) => (
        <div key={k} className="rounded-xl border border-border bg-surface-2 p-3">
          <div className="text-[11px] text-muted">{k}</div>
          <div className="num mt-1 text-lg font-semibold text-fg">{v}</div>
        </div>
      ))}
    </div>
  );
}

function History({ hist, metric }: { hist: AnyJson[]; metric?: string }) {
  if (!hist?.length) return null;
  const key = metric ?? Object.keys(hist[0]).find((k) => k.startsWith("val_")) ?? "train_loss";
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {["train_loss", key].map((k) => (
        <div key={k}>
          <SectionTitle>{k.replace(/_/g, " ")}</SectionTitle>
          <div className="h-40">
            <ResponsiveContainer>
              <LineChart data={hist} margin={{ left: 0, right: 12 }}>
                <CartesianGrid {...grid} />
                <XAxis dataKey="epoch" {...axis} />
                <YAxis {...axis} width={44} domain={["auto", "auto"]} tickFormatter={(v) => Number(v).toFixed(2)} />
                <Tooltip content={<ChartTooltip format={(v) => Number(v).toFixed(4)} labelFormat={(l) => `Epoch ${l}`} />} />
                <Line dataKey={k} name={k} stroke={k === "train_loss" ? SERIES[1] : SERIES[0]} strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ))}
    </div>
  );
}

function ChestSection({ m }: { m: AnyJson }) {
  const t = m.test;
  const labels = Object.keys(t.per_label);
  const [label, setLabel] = React.useState(labels.find((l) => l === "Effusion") ?? labels[0]);
  const bars = labels.map((l) => ({ label: l.replace("_", " "), auroc: t.per_label[l].auroc ?? 0, sens: t.per_label[l].sensitivity, spec: t.per_label[l].specificity })).sort((a, b) => b.auroc - a.auroc);
  const roc = (t.per_label[label]?.roc ?? []).map(([x, y]: number[]) => ({ fpr: x, tpr: y }));
  const rel = (t.reliability_calibrated?.bins ?? []).map((b: AnyJson) => ({ conf: b.confidence, cal: b.accuracy }));
  const relU = (t.reliability_uncalibrated?.bins ?? []).map((b: AnyJson) => ({ conf: b.confidence, uncal: b.accuracy }));
  const groups = Object.entries(m.fairness?.groups ?? {}).map(([g, v]: [string, AnyJson]) => ({ group: g.replace(":", " "), auroc: v.macro_auroc, n: v.n }));
  const members = Object.entries(m.history ?? {}) as [string, AnyJson[]][];
  const maxEp = Math.max(0, ...members.map(([, h]) => h.length));
  const curves = Array.from({ length: maxEp }, (_, i) => ({ epoch: i + 1, ...Object.fromEntries(members.map(([k, h]) => [k, h[i]?.val_macro_auroc])) }));
  return (
    <div className="space-y-4">
      <Tiles
        items={[
          ["Macro AUROC (test)", fmt(t.macro_auroc, 3)],
          ["Test images", (m.n_test ?? 0).toLocaleString()],
          ["ECE calibrated", fmt(t.reliability_calibrated?.ece, 3)],
          ["ECE uncalibrated", fmt(t.reliability_uncalibrated?.ece, 3)],
          ["Mean ensemble std", fmt(t.mean_ensemble_std, 3)],
          ["OOD energy AUROC", fmt(t.ood_energy_auroc, 3)],
          ["Fairness sex gap", fmt(m.fairness?.sex_gap, 3)],
          ["Members", m.members?.length],
        ]}
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Per-label AUROC" subtitle={m.dataset} />
          <div className="h-80 px-2 pb-4">
            <ResponsiveContainer>
              <BarChart data={bars} layout="vertical" margin={{ left: 12, right: 24 }} barCategoryGap={3}>
                <CartesianGrid {...grid} horizontal={false} vertical />
                <XAxis type="number" domain={[0.4, 1]} {...axis} tickFormatter={(v) => v.toFixed(1)} />
                <YAxis type="category" dataKey="label" {...axis} width={112} interval={0} />
                <ReferenceLine x={0.5} stroke="var(--subtle)" strokeDasharray="3 3" />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => Number(v).toFixed(3)} />} />
                <Bar dataKey="auroc" name="AUROC" fill={SERIES[0]} radius={[0, 4, 4, 0]} maxBarSize={14} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card>
          <CardHeader
            title="ROC curve"
            subtitle={`AUROC ${fmt(t.per_label[label]?.auroc, 3)} · threshold ${fmt(t.per_label[label]?.threshold, 3)} · sens ${pct(t.per_label[label]?.sensitivity)} / spec ${pct(t.per_label[label]?.specificity)}`}
            action={
              <NativeSelect value={label} onChange={(e) => setLabel(e.target.value)} className="h-8 text-xs" aria-label="Label">
                {labels.map((l) => <option key={l}>{l}</option>)}
              </NativeSelect>
            }
          />
          <div className="h-80 px-2 pb-4">
            <ResponsiveContainer>
              <LineChart data={roc} margin={{ left: 0, right: 16 }}>
                <CartesianGrid {...grid} vertical />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} {...axis} tickFormatter={(v) => v.toFixed(1)} label={{ value: "False positive rate", position: "insideBottom", offset: -2, fill: "var(--muted)", fontSize: 11 }} height={36} />
                <YAxis domain={[0, 1]} {...axis} width={40} tickFormatter={(v) => v.toFixed(1)} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--subtle)" strokeDasharray="3 3" />
                <Tooltip content={<ChartTooltip format={(v) => Number(v).toFixed(3)} labelFormat={(l) => `FPR ${Number(l).toFixed(2)}`} />} />
                <Line dataKey="tpr" name="TPR" stroke={SERIES[0]} strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card>
          <CardHeader title="Calibration (reliability diagram)" subtitle="Temperature scaling per label and member, fitted on validation" action={<Legend items={[{ label: "Calibrated", color: SERIES[0] }, { label: "Uncalibrated", color: SERIES[1], dashed: true }]} />} />
          <div className="h-72 px-2 pb-4">
            <ResponsiveContainer>
              <LineChart margin={{ left: 0, right: 16 }}>
                <CartesianGrid {...grid} vertical />
                <XAxis dataKey="conf" type="number" domain={[0, 1]} {...axis} tickFormatter={(v) => v.toFixed(1)} allowDuplicatedCategory={false} />
                <YAxis domain={[0, 1]} {...axis} width={40} tickFormatter={(v) => v.toFixed(1)} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--subtle)" strokeDasharray="3 3" />
                <Tooltip content={<ChartTooltip format={(v) => Number(v).toFixed(3)} labelFormat={(l) => `Predicted ${Number(l).toFixed(2)}`} />} />
                <Line data={rel} dataKey="cal" name="Calibrated" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 3 }} />
                <Line data={relU} dataKey="uncal" name="Uncalibrated" stroke={SERIES[1]} strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card>
          <CardHeader title="Fairness: macro AUROC by subgroup" subtitle={m.fairness?.dataset ?? "No metadata set available"} />
          <div className="h-72 px-2 pb-4">
            {groups.length ? (
              <ResponsiveContainer>
                <BarChart data={groups} margin={{ left: 0, right: 12 }}>
                  <CartesianGrid {...grid} />
                  <XAxis dataKey="group" {...axis} interval={0} />
                  <YAxis domain={[0.4, 1]} {...axis} width={40} tickFormatter={(v) => v.toFixed(1)} />
                  <ReferenceLine y={m.fairness?.overall_macro_auroc} stroke="var(--muted)" strokeDasharray="4 4" label={{ value: "overall", fill: "var(--muted)", fontSize: 10, position: "right" }} />
                  <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v, n) => (n === "n" ? String(v) : Number(v).toFixed(3))} />} />
                  <Bar dataKey="auroc" name="Macro AUROC" fill={SERIES[0]} radius={[4, 4, 0, 0]} maxBarSize={36} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="p-6 text-sm text-muted">No subgroup metadata.</p>
            )}
          </div>
        </Card>
      </div>
      {curves.length > 0 && (
        <Card>
          <CardHeader title="Training curves — validation macro AUROC per ensemble member" action={<Legend items={members.map(([k], i) => ({ label: k.replace("chest_", ""), color: SERIES[i] }))} />} />
          <div className="h-64 px-2 pb-4">
            <ResponsiveContainer>
              <LineChart data={curves} margin={{ left: 0, right: 16 }}>
                <CartesianGrid {...grid} />
                <XAxis dataKey="epoch" {...axis} />
                <YAxis {...axis} width={44} domain={["auto", "auto"]} tickFormatter={(v) => Number(v).toFixed(2)} />
                <Tooltip content={<ChartTooltip format={(v) => Number(v).toFixed(3)} labelFormat={(l) => `Epoch ${l}`} />} />
                {members.map(([k], i) => <Line key={k} dataKey={k} name={k.replace("chest_", "")} stroke={SERIES[i]} strokeWidth={2} dot={{ r: 3 }} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
    </div>
  );
}

function GenericSection({ name, m }: { name: string; m: AnyJson }) {
  const t = m.test ?? {};
  if (BINARY.includes(name)) return <BinarySection m={m} />;
  if (name === "radiogenomics") return <RadiogenomicsSection m={m} />;
  if (name === "sr_gan")
    return (
      <div className="space-y-4">
        <Tiles items={[["PSNR bicubic", `${fmt(t.psnr_bicubic, 2)} dB`], ["PSNR SR-GAN", `${fmt(t.psnr_ours, 2)} dB`], ["SSIM bicubic", fmt(t.ssim_bicubic, 3)], ["SSIM SR-GAN", fmt(t.ssim_ours, 3)]]} />
        <p className="text-xs text-subtle">{m.note}</p>
      </div>
    );
  if (t.confusion_matrix)
    return (
      <div className="space-y-4">
        <Tiles items={[["Accuracy", pct(t.accuracy, 1)], ["Macro-F1", fmt(t.macro_f1, 3)], ["Temperature", fmt(t.temperature, 2)], ["Test images", m.n_test]]} />
        {t.referable_auroc != null && (
          <Tiles items={[["Referable-DR AUROC", fmt(t.referable_auroc, 3)], ["Quadratic κ", fmt(t.quadratic_kappa, 3)], ["Referable sensitivity", pct(t.referable?.sensitivity, 1)], ["Referable specificity", pct(t.referable?.specificity, 1)]]} />
        )}
        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="p-4"><SectionTitle>Confusion matrix (test)</SectionTitle><ConfusionMatrix cm={t.confusion_matrix} classes={t.classes} /></Card>
          <Card className="p-4">
            <SectionTitle>Per-class</SectionTitle>
            <table className="w-full text-sm">
              <thead><tr className="text-left text-[11px] text-subtle"><th className="py-1">Class</th><th>F1</th><th>AUROC</th></tr></thead>
              <tbody>
                {t.classes.map((c: string) => (
                  <tr key={c} className="border-t border-border/60"><td className="py-1.5 text-fg">{titleCase(c)}</td><td className="num">{fmt(t.per_class_f1[c], 3)}</td><td className="num">{fmt(t.per_class_auroc[c], 3)}</td></tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
        <Card className="p-4"><History hist={m.history} /></Card>
      </div>
    );
  if (name === "denoiser")
    return (
      <div className="space-y-4">
        <Card className="p-4">
          <SectionTitle>PSNR / SSIM by simulated dose (test, 200 images)</SectionTitle>
          <table className="w-full text-sm">
            <thead><tr className="text-left text-[11px] text-subtle"><th className="py-1">Dose</th><th>PSNR noisy</th><th>PSNR Gaussian blur</th><th>PSNR ours</th><th>SSIM noisy</th><th>SSIM ours</th></tr></thead>
            <tbody>
              {Object.entries(t.levels ?? {}).map(([k, v]: [string, AnyJson]) => (
                <tr key={k} className="num border-t border-border/60">
                  <td className="py-1.5 font-sans text-fg">{k}</td><td>{fmt(v.psnr_noisy, 1)}</td><td>{fmt(v.psnr_gaussian_blur, 1)}</td><td className="text-cyan">{fmt(v.psnr_denoised, 1)}</td><td>{fmt(v.ssim_noisy, 3)}</td><td className="text-cyan">{fmt(v.ssim_denoised, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card className="p-4"><History hist={m.history} /></Card>
      </div>
    );
  if (name === "report_gen")
    return (
      <div className="space-y-4">
        <Tiles items={[["BLEU-4", fmt(t.bleu4, 3)], ["ROUGE-L", fmt(t.rouge_l, 3)], ["Baseline BLEU-4", fmt(t.baseline_most_common_report?.bleu4, 3)], ["Consistency F1", fmt(t.clinical_consistency?.f1, 3)]]} />
        <Card className="p-4 text-sm">
          <SectionTitle>Examples (test)</SectionTitle>
          {(t.examples ?? []).slice(0, 3).map((e: AnyJson, i: number) => (
            <div key={i} className="mb-3 grid gap-2 rounded-xl border border-border bg-surface-2 p-3 md:grid-cols-2">
              <div><div className="text-[11px] text-subtle">Reference</div><p className="text-muted">{e.reference}</p></div>
              <div><div className="text-[11px] text-subtle">Generated</div><p className="text-fg">{e.generated}</p></div>
            </div>
          ))}
          <p className="text-xs text-subtle">Baseline = always emitting the most common training report; honest context for the n-gram scores.</p>
        </Card>
      </div>
    );
  if (name === "copilot_intent")
    return (
      <Tiles items={[["Held-out accuracy", pct(t.accuracy, 1)], ["Macro-F1", fmt(t.macro_f1, 3)], ["Template-CV accuracy", pct(t.template_cv_accuracy_mean, 1)], ["Intents", m.n_intents]]} />
    );
  const items: [string, React.ReactNode][] = Object.entries(t)
    .filter(([, v]) => typeof v === "number")
    .slice(0, 8)
    .map(([k, v]) => [k.replace(/_/g, " "), fmt(v as number, 3)]);
  return (
    <div className="space-y-4">
      <Tiles items={items} />
      <Card className="p-4"><History hist={m.history} /></Card>
    </div>
  );
}

export default function LabPage() {
  const q = useQuery({ queryKey: ["metrics"], queryFn: api.metrics });
  const [profile, setProfile] = React.useState("starter");
  const [model, setModel] = React.useState("chest_ensemble");
  const models = q.data?.models ?? {};
  const hasFull = Object.values(models).some((p) => p.full);
  const order = Object.keys(NAMES).filter((n) => models[n] || n === "second_reader");
  const m = model === "second_reader" ? { profile } : (models[model]?.[profile] ?? models[model]?.starter);

  return (
    <div>
      <PageHeader
        eyebrow="Model performance"
        title="The Lab"
        subtitle="Honest, held-out metrics for every model we trained — exactly as written by ml/evaluate.py. Starter models are laptop-trained on small images; full models come from Kaggle GPU runs."
        actions={
          <Tabs value={profile} onValueChange={setProfile}>
            <TabsList>
              <TabsTrigger value="starter">Starter</TabsTrigger>
              <TabsTrigger value="full" disabled={!hasFull}>Full {hasFull ? "" : "(not trained yet)"}</TabsTrigger>
            </TabsList>
          </Tabs>
        }
      />
      {q.isLoading ? (
        <Skeleton className="h-[60vh]" />
      ) : !order.length ? (
        <Card><EmptyState icon={<FlaskConical />} title="No metrics yet" body="Run `make train-starter` to train and evaluate the starter models." /></Card>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[300px_1fr]">
          <Card className="h-fit p-2">
            {order.map((n) => {
              const [k, v] = n === "second_reader" ? ["sens / spec / FPR / FNR", ""] : headline(n, models[n][profile] ?? models[n].starter);
              const ms = models[n]?.[profile]?.export?.cpu_ms_per_image ?? models[n]?.starter?.export?.cpu_ms_per_image;
              return (
                <button
                  key={n}
                  onClick={() => setModel(n)}
                  className={cn("flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left transition", model === n ? "bg-surface-2 ring-1 ring-border-strong" : "hover:bg-surface-2")}
                >
                  <div className="min-w-0 flex-1">
                    <div className={cn("truncate text-[13px]", model === n ? "text-fg" : "text-muted")}>{NAMES[n]}</div>
                    <div className="num text-[11px] text-subtle">{k} {v}{ms ? ` · ${ms} ms CPU` : ""}</div>
                  </div>
                  {models[n]?.full && <Badge tone="violet">full</Badge>}
                </button>
              );
            })}
          </Card>
          <div className="min-w-0">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-semibold text-fg">{NAMES[model]?.replace("· ", "")}</h2>
              <Badge tone={profile === "full" ? "violet" : "cyan"}>{m?.profile ?? profile}</Badge>
              {m?.generated_at && <span className="num text-[11px] text-subtle">evaluated {m.generated_at}</span>}
            </div>
            {!m ? (
              <Card><EmptyState icon={<FlaskConical />} title="No metrics for this profile" /></Card>
            ) : model === "second_reader" ? (
              <SecondReader models={models} profile={profile} />
            ) : model === "chest_ensemble" ? (
              <ChestSection m={m} />
            ) : (
              <GenericSection name={model} m={m} />
            )}
            <Card className="mt-4 p-4">
              <SectionTitle>Dataset splits (patient-level where IDs exist)</SectionTitle>
              <div className="grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-4">
                {Object.entries(q.data?.dataset_sizes ?? {}).map(([k, v]) => (
                  <div key={k} className="rounded-lg border border-border bg-surface-2 px-3 py-2">
                    <div className="font-medium text-fg">{k}</div>
                    <div className="num text-muted">{Object.entries(v).map(([s, n]) => `${s} ${n.toLocaleString()}`).join(" · ")}</div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}

