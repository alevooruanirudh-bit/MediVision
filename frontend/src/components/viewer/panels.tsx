"use client";

import { motion } from "framer-motion";
import { Check, Fingerprint, Lightbulb, Pencil, Ruler, UserCheck, X } from "lucide-react";
import * as React from "react";
import { toast } from "sonner";

import { SectionTitle } from "@/components/common";
import { Badge, Button, Input, Tip } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { Finding, Results } from "@/lib/types";
import { cn, fmt, pct } from "@/lib/utils";

const CERT_TONE = { Likely: "red", Possible: "amber", Unlikely: "neutral" } as const;

export function ProbabilityBar({ f, delay = 0 }: { f: Finding; delay?: number }) {
  const p = Math.min(1, f.probability);
  const lo = Math.max(0, p - (f.uncertainty || 0));
  const hi = Math.min(1, p + (f.uncertainty || 0));
  const color = f.positive ? (f.certainty === "Likely" ? "var(--red)" : "var(--amber)") : "var(--s1)";
  return (
    <div className="relative h-2.5 w-full rounded-full bg-surface-2" aria-hidden>
      <motion.div
        className="absolute inset-y-0 left-0 rounded-full"
        style={{ background: color, opacity: f.positive ? 0.95 : 0.55 }}
        initial={{ width: 0 }}
        animate={{ width: `${p * 100}%` }}
        transition={{ delay, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      />
      {f.uncertainty > 0.005 && (
        <div className="absolute top-1/2 h-[2px] -translate-y-1/2 bg-fg/70" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }}>
          <span className="absolute -top-[3px] left-0 h-2 w-[2px] bg-fg/70" />
          <span className="absolute -top-[3px] right-0 h-2 w-[2px] bg-fg/70" />
        </div>
      )}
      <Tip content={`Operating threshold ${pct(f.threshold)} (Youden's J on validation)`}>
        <span className="absolute -top-1 h-[18px] w-[2px] rounded bg-fg/80" style={{ left: `calc(${f.threshold * 100}% - 1px)` }} />
      </Tip>
    </div>
  );
}

export function FindingsPanel({
  results,
  selected,
  onSelect,
  onReviewed,
}: {
  results: Results;
  selected: string | null;
  onSelect: (key: string) => void;
  onReviewed: () => void;
}) {
  const [showAll, setShowAll] = React.useState(false);
  const [editing, setEditing] = React.useState<string | null>(null);
  const [editText, setEditText] = React.useState("");
  const pos = results.findings.filter((f) => f.positive);
  const rest = results.findings.filter((f) => !f.positive);
  const list = showAll ? results.findings : [...pos, ...rest.slice(0, Math.max(0, 6 - pos.length))];

  const review = async (f: Finding, action: "accept" | "reject" | "edit", label?: string) => {
    if (!f.id) return;
    try {
      await api.review(f.id, action, label);
      toast.success(`${f.label}: ${action === "edit" ? `edited → ${label}` : action + "ed"} (logged)`);
      onReviewed();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  if (!results.findings.length)
    return <p className="p-4 text-sm text-muted">No model outputs for this image{results.modality.warning ? ` — ${results.modality.warning}` : "."}</p>;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between px-1 pb-1 text-[11px] text-subtle">
        <span>
          {pos.length} above threshold · bar = calibrated p · whisker = uncertainty · tick = threshold
        </span>
      </div>
      {list.map((f, i) => {
        const st = f.review?.status;
        return (
          <motion.div
            key={f.key}
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.03 }}
            className={cn(
              "group cursor-pointer rounded-xl border px-3 py-2.5 transition-all",
              selected === f.key ? "border-cyan/50 bg-cyan/[0.06]" : "border-transparent hover:border-border hover:bg-surface-2",
            )}
            onClick={() => onSelect(f.key)}
            data-testid="finding-row"
          >
            <div className="mb-1.5 flex items-center gap-2">
              <span className={cn("text-[13px] font-medium", f.positive ? "text-fg" : "text-muted")}>
                {f.review?.edited_label ?? f.label}
              </span>
              <Badge tone={CERT_TONE[f.certainty]}>{f.certainty}</Badge>
              {st && st !== "pending" && (
                <Badge tone={st === "accepted" ? "teal" : st === "rejected" ? "red" : "violet"}>{st}</Badge>
              )}
              <span className="num ml-auto text-[13px] text-fg">{pct(f.probability, 1)}</span>
              <span className="num w-12 text-right text-[11px] text-subtle">±{fmt(f.uncertainty)}</span>
            </div>
            <ProbabilityBar f={f} delay={i * 0.04} />
            {(selected === f.key || f.positive) && f.rationale && (
              <div className="mt-2 rounded-lg border border-border/70 bg-bg/40 px-2.5 py-2 text-[11px] leading-relaxed text-muted" data-testid="finding-rationale">
                <div className="mb-0.5 flex items-center gap-1 font-medium text-fg">
                  <Lightbulb className="size-3 text-amber" /> {f.positive ? "Why the AI flagged this" : "Why it is not flagged"}
                </div>
                <ul className="list-disc space-y-0.5 pl-4">
                  {f.rationale.reasons.map((r) => (
                    <li key={r.text}>{r.text}</li>
                  ))}
                </ul>
              </div>
            )}
            {(selected === f.key || f.positive) && (f.provenance || f.review?.reviewed_by) && (
              <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-subtle">
                {f.provenance && (
                  <span className="inline-flex items-center gap-1">
                    <Fingerprint className="size-3" /> AI · {f.provenance.model} <span className="num">{f.provenance.model_version}</span>
                  </span>
                )}
                {f.review?.reviewed_by && (
                  <span className="inline-flex items-center gap-1 text-teal">
                    <UserCheck className="size-3" /> {f.review.status} by {f.review.reviewed_by} ({f.review.reviewed_role})
                  </span>
                )}
              </div>
            )}
            {(selected === f.key || f.positive) && (f.zone || f.member_probs) && (
              <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted">
                {f.zone && <span>📍 {f.zone}</span>}
                {f.member_probs &&
                  Object.entries(f.member_probs).map(([m, v]) => (
                    <span key={m} className="num">
                      {m.replace("chest_", "").replace("efficientnet_", "effnet-").replace("convnext_", "convnext-")} {pct(v)}
                    </span>
                  ))}
              </div>
            )}
            {f.id && (
              <div className={cn("mt-2 items-center gap-1.5", selected === f.key || editing === f.key ? "flex" : "hidden md:group-hover:flex")} onClick={(e) => e.stopPropagation()}>
                {editing === f.key ? (
                  <form
                    className="flex w-full gap-1.5"
                    onSubmit={(e) => {
                      e.preventDefault();
                      review(f, "edit", editText);
                      setEditing(null);
                    }}
                  >
                    <Input autoFocus value={editText} onChange={(e) => setEditText(e.target.value)} className="h-7 text-xs" placeholder="Corrected label" aria-label="Corrected label" />
                    <Button size="sm" variant="primary" type="submit">Save</Button>
                  </form>
                ) : (
                  <>
                    <Button size="sm" variant="success" onClick={() => review(f, "accept")} aria-label={`Accept ${f.label}`}><Check /> Accept</Button>
                    <Button size="sm" variant="danger" onClick={() => review(f, "reject")} aria-label={`Reject ${f.label}`}><X /> Reject</Button>
                    <Button size="sm" variant="ghost" onClick={() => { setEditing(f.key); setEditText(f.label); }} aria-label={`Edit ${f.label}`}><Pencil /> Edit</Button>
                  </>
                )}
              </div>
            )}
          </motion.div>
        );
      })}
      {results.findings.length > list.length || showAll ? (
        <button className="w-full py-2 text-xs text-cyan hover:underline" onClick={() => setShowAll((s) => !s)}>
          {showAll ? "Show fewer" : `Show all ${results.findings.length} outputs`}
        </button>
      ) : null}
    </div>
  );
}

function Gauge({ value, max = 1, flag = 0.5, label }: { value: number; max?: number; flag?: number; label: string }) {
  const r = 46;
  const c = Math.PI * r;
  const frac = Math.min(1, value / max);
  const over = value > flag;
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 120 70" className="w-40" aria-label={`${label} ${value}`}>
        <path d={`M14 60 A ${r} ${r} 0 0 1 106 60`} fill="none" stroke="var(--surface-2)" strokeWidth="10" strokeLinecap="round" />
        <motion.path
          d={`M14 60 A ${r} ${r} 0 0 1 106 60`}
          fill="none"
          stroke={over ? "var(--amber)" : "var(--teal)"}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - frac) }}
          transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
        />
        <line x1="60" y1="8" x2="60" y2="18" stroke="var(--muted)" strokeWidth="1.5" transform={`rotate(${(flag / max) * 180 - 90} 60 60)`} />
      </svg>
      <div className="num -mt-6 text-2xl font-semibold text-fg">{fmt(value)}</div>
      <div className="text-[11px] text-muted">{label}</div>
    </div>
  );
}

export function RadiomicsPanel({ results }: { results: Results }) {
  const r = results.radiomics;
  if (!r) return null;
  const groups: [string, Record<string, number> | undefined][] = [
    ["Shape", r.shape],
    ["First-order intensity", r.first_order],
    ["Texture (GLCM)", r.texture_glcm],
  ];
  return (
    <div className="space-y-3 p-1" data-testid="radiomics-panel">
      <SectionTitle>Imaging biomarkers · radiomics</SectionTitle>
      <p className="text-[11px] text-muted">
        Region: <span className="text-fg">{r.roi}</span>. IBSI-style features computed locally; descriptive, not validated biomarkers.
      </p>
      {groups.map(([title, vals]) =>
        vals && Object.keys(vals).length ? (
          <div key={title}>
            <div className="mb-1 text-[11px] font-medium text-fg">{title}</div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[11px]">
              {Object.entries(vals).map(([k, v]) => (
                <div key={k} className="flex justify-between border-b border-border/50 py-0.5">
                  <span className="text-muted">{k.replace(/_/g, " ")}</span>
                  <span className="num text-fg">{typeof v === "number" ? (Math.abs(v) >= 100 ? v.toLocaleString() : v) : String(v)}</span>
                </div>
              ))}
            </div>
          </div>
        ) : null,
      )}
    </div>
  );
}

export function MeasurementsPanel({ results }: { results: Results }) {
  const m = results.measurements;
  if (results.pipeline === "ct_volume" && m?.available)
    return (
      <div className="space-y-4 p-1">
        <SectionTitle>Nodule volumetrics (3-D)</SectionTitle>
        <div className="grid grid-cols-2 gap-3">
          <Metric label="Volume" value={`${(m.volume_mm3 ?? 0).toLocaleString()} mm³`} hint={m.spacing_assumed ? "1 mm voxel spacing assumed" : "from DICOM spacing"} />
          <Metric label="Equivalent diameter" value={`${m.equivalent_diameter_mm ?? "—"} mm`} hint="sphere of equal volume" />
          <Metric label="Slices with nodule" value={m.slices_with_nodule ?? "—"} />
          <Metric label="Voxels" value={(m.voxels ?? 0).toLocaleString()} />
        </div>
        <p className="text-[11px] text-subtle">Classical 3-D segmentation (Otsu + central component) — replaces manual calliper measurement; verify on the MPR views.</p>
        <RadiomicsPanel results={results} />
      </div>
    );
  if (results.pipeline === "ct_chest_scan") {
    const dets = results.extra.chest_scan?.detections ?? [];
    return (
      <div className="space-y-4 p-1">
        <SectionTitle>Whole-scan nodule detection</SectionTitle>
        <div className="grid grid-cols-2 gap-3">
          <Metric label="Detections ≥ threshold" value={m.detections_above_threshold ?? 0} hint={m.operating_point ?? undefined} />
          <Metric label="Candidates scored" value={m.candidates_scored ?? "—"} hint="LoG blob candidates in the lung mask" />
          <Metric label="Lung volume" value={m.lung_volume_ml != null ? `${m.lung_volume_ml.toLocaleString()} ml` : "—"} hint={m.spacing_assumed ? "1 mm spacing assumed" : "from supplied spacing"} />
        </div>
        {dets.length > 0 && (
          <table className="w-full text-[11px]" data-testid="nodule-detections">
            <thead className="text-left text-muted">
              <tr><th className="py-1">#</th><th>Position z/y/x (mm)</th><th>Score</th><th>Ø mm</th><th>Malignancy</th></tr>
            </thead>
            <tbody>
              {dets.map((d) => (
                <tr key={d.rank} className="num border-t border-border/50">
                  <td className="py-1">{d.rank}</td>
                  <td>{d.centre_mm_zyx.join(" / ")}</td>
                  <td className={d.positive ? "font-medium text-fg" : "text-muted"}>{pct(d.probability, 0)}</td>
                  <td title={d.size_note ?? undefined}>{d.diameter_mm ?? "indet."}</td>
                  <td>{d.malignancy_probability != null ? pct(d.malignancy_probability, 0) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="text-[11px] text-subtle">Our lung mask + blob candidates + 3-D CNN (LUNA16). Top 5 candidates shown; bold rows are above the operating threshold. Size and malignancy come from the 64 mm cube around each detection.</p>
        <RadiomicsPanel results={results} />
      </div>
    );
  }
  if (results.pipeline === "petct")
    return (
      <div className="space-y-4 p-1">
        <SectionTitle>PET-CT quantification</SectionTitle>
        <div className="grid grid-cols-2 gap-3">
          <Metric label="SUVmax (lesion)" value={fmt(m.suv_max, 2)} hint={`whole body ${fmt(m.suv_max_whole_body, 2)}`} />
          <Metric label="SUVmean (lesion)" value={fmt(m.suv_mean, 2)} />
          <Metric label="Metabolic tumour volume" value={`${fmt(m.mtv_ml, 1)} ml`} hint={m.spacing_assumed ? "5 mm voxels assumed" : "from supplied spacing"} />
          <Metric label="Total lesion glycolysis" value={fmt(m.tlg, 1)} hint="SUVmean × MTV" />
        </div>
        <p className="text-[11px] text-subtle">Lesion mask from our PET+CT fusion U-Net (largest component). SUV values are taken as supplied in the upload.</p>
        <RadiomicsPanel results={results} />
      </div>
    );
  if (!m?.available)
    return results.radiomics ? (
      <RadiomicsPanel results={results} />
    ) : (
      <p className="p-4 text-sm text-muted">No automatic measurements for this modality.</p>
    );
  if (results.pipeline === "brain")
    return (
      <div className="space-y-4 p-1">
        <SectionTitle>Tumour segmentation (Attention U-Net)</SectionTitle>
        <div className="grid grid-cols-2 gap-3">
          <Metric label="Lesion area" value={`${(m.tumour_area_px ?? 0).toLocaleString()} px`} hint={m.tumour_area_mm2 ? `${m.tumour_area_mm2} mm²` : "pixel spacing unknown"} />
          <Metric label="Slice fraction" value={pct(m.tumour_fraction, 2)} />
          <Metric label="Consistent with classifier" value={m.consistent_with_classifier ? "Yes" : "No"} hint="mask present ↔ tumour class" />
        </div>
        <RadiomicsPanel results={results} />
      </div>
    );
  return (
    <div className="space-y-5 p-1">
      {m.ctr != null && (
        <div className="flex flex-col items-center rounded-2xl border border-border bg-surface-2 py-4">
          <Gauge value={m.ctr} label="Cardiothoracic ratio (estimate)" />
          <Badge tone={m.ctr_flag ? "amber" : "teal"} className="mt-2">
            {m.ctr_flag ? "Above 0.5 — correlate for cardiomegaly" : "Within the 0.5 PA cut-off"}
          </Badge>
        </div>
      )}
      <div className="grid grid-cols-2 gap-3">
        <Metric label="Right lung area" value={`${(m.right_lung_area_mm2 ?? m.right_lung_area_px ?? 0).toLocaleString()} ${m.right_lung_area_mm2 ? "mm²" : "px"}`} />
        <Metric label="Left lung area" value={`${(m.left_lung_area_mm2 ?? m.left_lung_area_px ?? 0).toLocaleString()} ${m.left_lung_area_mm2 ? "mm²" : "px"}`} />
        <Metric label="Symmetry index" value={fmt(m.symmetry_index)} hint="smaller / larger lung area" />
        <Metric label="Heart / thorax width" value={`${m.heart_width_px ?? "—"} / ${m.thorax_width_px ?? "—"} px`} />
      </div>
      <p className="flex items-start gap-2 text-[11px] leading-relaxed text-subtle">
        <Ruler className="mt-0.5 size-3.5 shrink-0" />
        From our lung U-Net masks. Heart width = widest gap between medial lung borders (lower half). AP films magnify the heart; treat as an estimate.
      </p>
      <RadiomicsPanel results={results} />
    </div>
  );
}

function Metric({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface-2 p-3">
      <div className="text-[11px] text-muted">{label}</div>
      <div className="num mt-1 text-base font-semibold text-fg">{value}</div>
      {hint && <div className="mt-0.5 text-[10px] text-subtle">{hint}</div>}
    </div>
  );
}
