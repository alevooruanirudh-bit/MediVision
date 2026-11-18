"use client";

import { useQuery } from "@tanstack/react-query";
import { Loader2, MoveHorizontal, Wand2 } from "lucide-react";
import * as React from "react";

import { EmptyState, PageHeader } from "@/components/common";
import { Badge, Card, CardHeader, Label, NativeSelect, Skeleton, Slider, Switch } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { fmt, ms } from "@/lib/utils";

function BeforeAfter({ before, after }: { before: string; after: string }) {
  const [pos, setPos] = React.useState(50);
  const ref = React.useRef<HTMLDivElement>(null);
  const move = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (r) setPos(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)));
  };
  return (
    <div
      ref={ref}
      className="relative mx-auto aspect-square w-full max-w-[620px] cursor-ew-resize touch-none select-none overflow-hidden rounded-2xl bg-black"
      onPointerDown={(e) => {
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        move(e.clientX);
      }}
      onPointerMove={(e) => e.buttons && move(e.clientX)}
      onKeyDown={(e) => {
        if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 5));
        if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 5));
      }}
      role="slider"
      aria-label="Before / after comparison"
      aria-valuenow={Math.round(pos)}
      aria-valuemin={0}
      aria-valuemax={100}
      tabIndex={0}
      data-testid="before-after"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={after} alt="Enhanced" className="absolute inset-0 h-full w-full object-contain" draggable={false} />
      <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={before} alt="Before" className="absolute inset-0 h-full w-full object-contain" draggable={false} />
      </div>
      <div className="absolute inset-y-0 w-0.5 bg-white/80 shadow-[0_0_12px_rgba(34,211,238,0.8)]" style={{ left: `${pos}%` }}>
        <div className="absolute left-1/2 top-1/2 grid size-9 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-white/30 bg-black/70 text-white backdrop-blur">
          <MoveHorizontal className="size-4" />
        </div>
      </div>
      <span className="absolute left-3 top-3 rounded-md bg-black/60 px-2 py-0.5 text-[11px] text-slate-200">Before</span>
      <span className="absolute right-3 top-3 rounded-md bg-black/60 px-2 py-0.5 text-[11px] text-cyan">After</span>
    </div>
  );
}

export default function EnhancePage() {
  const studies = useQuery({ queryKey: ["studies"], queryFn: api.studies });
  const xrays = (studies.data ?? []).filter((s) => s.pipeline && s.pipeline !== "ct_volume" && s.pipeline !== "fundus");
  const [sid, setSid] = React.useState<string>("");
  React.useEffect(() => {
    if (!sid && xrays.length) setSid(xrays[0].id);
  }, [sid, xrays]);
  const [lowDose, setLowDose] = React.useState(true);
  const [photons, setPhotons] = React.useState(60);
  const [denoise, setDenoise] = React.useState(true);
  const [clahe, setClahe] = React.useState(false);
  const [clip, setClip] = React.useState(2);
  const [unsharp, setUnsharp] = React.useState(0);
  const [gamma, setGamma] = React.useState(1);
  const [superres, setSuperres] = React.useState(false);
  const ops = {
    ...(lowDose ? { simulate_low_dose: { photons } } : {}),
    ...(denoise ? { denoise: true } : {}),
    ...(clahe ? { clahe: { clip } } : {}),
    ...(unsharp > 0 ? { unsharp: { amount: unsharp } } : {}),
    ...(gamma !== 1 ? { gamma } : {}),
    ...(superres ? { superres: true } : {}),
  };
  const res = useQuery({
    queryKey: ["enhance", sid, ops],
    queryFn: () => api.enhance(sid, ops),
    enabled: !!sid,
    placeholderData: (p) => p,
  });
  const m = res.data?.metrics;

  return (
    <div>
      <PageHeader
        eyebrow="Advanced imaging · III-F"
        title="Enhancement Lab"
        subtitle="Simulate a low-dose acquisition, then restore it with our DnCNN-style denoiser (trained on synthetic Poisson + Gaussian noise) and classical filters. PSNR / SSIM are measured against the original."
      />
      {!xrays.length && !studies.isLoading ? (
        <Card><EmptyState icon={<Wand2 />} title="No X-ray studies yet" body="Load the demo data or upload a chest / bone X-ray." /></Card>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[1fr_360px]">
          <Card className="p-4">
            {res.data ? <BeforeAfter before={res.data.before} after={res.data.image} /> : <Skeleton className="mx-auto aspect-square max-w-[620px]" />}
            <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs">
              {res.data?.steps.map((s) => <Badge key={s} tone="cyan">{s}</Badge>)}
              {res.isFetching && <Loader2 className="size-4 animate-spin text-cyan" />}
              {res.data && <span className="num text-subtle">{ms(res.data.ms)}</span>}
            </div>
          </Card>
          <div className="space-y-4">
            <Card className="p-4">
              <Label>Study</Label>
              <NativeSelect value={sid} onChange={(e) => setSid(e.target.value)} className="mt-1 w-full" aria-label="Study">
                {xrays.map((s) => <option key={s.id} value={s.id}>{s.patient?.pseudonym} · {s.filename}</option>)}
              </NativeSelect>
            </Card>
            <Card className="space-y-4 p-4">
              <Row label="Simulate low dose" on={lowDose} set={setLowDose}>
                <Slider value={[photons]} min={15} max={400} step={5} onValueChange={([v]) => setPhotons(v)} disabled={!lowDose} aria-label="Photons" />
                <span className="num text-[11px] text-subtle">{photons} photons/pixel (lower = noisier)</span>
              </Row>
              <Row label="DnCNN denoiser (ours)" on={denoise} set={setDenoise} />
              <Row label="Super-resolution ×2 (SR-GAN, ours · experimental)" on={superres} set={setSuperres}>
                <span className="text-[11px] leading-snug text-subtle">
                  Low-resolution enhancement. GAN texture can be invented — never diagnose from super-resolved pixels.
                </span>
              </Row>
              <Row label="CLAHE" on={clahe} set={setClahe}>
                <Slider value={[clip]} min={0.5} max={6} step={0.5} onValueChange={([v]) => setClip(v)} disabled={!clahe} aria-label="CLAHE clip" />
                <span className="num text-[11px] text-subtle">clip limit {clip}</span>
              </Row>
              <div>
                <Label>Unsharp mask · {fmt(unsharp, 1)}</Label>
                <Slider className="mt-2" value={[unsharp]} min={0} max={2} step={0.1} onValueChange={([v]) => setUnsharp(v)} aria-label="Unsharp amount" />
              </div>
              <div>
                <Label>Gamma · {fmt(gamma, 2)}</Label>
                <Slider className="mt-2" value={[gamma]} min={0.5} max={2} step={0.05} onValueChange={([v]) => setGamma(v)} aria-label="Gamma" />
              </div>
            </Card>
            <Card>
              <CardHeader title="Image quality" subtitle="vs. the original (higher is better)" />
              <div className="grid grid-cols-2 gap-3 px-5 pb-5">
                {lowDose && m?.psnr_noisy != null && (
                  <>
                    <Q label="PSNR low-dose" v={`${fmt(m.psnr_noisy, 1)} dB`} />
                    <Q label="SSIM low-dose" v={fmt(m.ssim_noisy, 3)} />
                  </>
                )}
                <Q label="PSNR result" v={m ? `${fmt(m.psnr, 1)} dB` : "—"} accent />
                <Q label="SSIM result" v={m ? fmt(m.ssim, 3) : "—"} accent />
              </div>
            </Card>
          </div>
          <FusionCard studies={(studies.data ?? []).filter((s) => s.pipeline && !["ct_volume", "ct_chest_scan", "petct"].includes(s.pipeline))} />
          <FmriCard />
        </div>
      )}
    </div>
  );
}

/** Multimodal-style fusion (C7): rigid phase-correlation registration + false-colour overlay. */
function FusionCard({ studies }: { studies: { id: string; filename: string; patient: { pseudonym: string } | null }[] }) {
  const [a, setA] = React.useState("");
  const [b, setB] = React.useState("");
  const [alpha, setAlpha] = React.useState(0.5);
  React.useEffect(() => {
    if (!a && studies.length > 1) {
      setA(studies[0].id);
      setB(studies[1].id);
    }
  }, [a, studies]);
  const f = useQuery({
    queryKey: ["fusion", a, b, alpha],
    queryFn: () => api.fusion(a, b, alpha),
    enabled: !!a && !!b && a !== b,
    placeholderData: (p) => p,
  });
  return (
    <Card className="p-4 xl:col-span-2" data-testid="fusion-card">
      <CardHeader
        title="Image fusion (experimental)"
        subtitle="Any two co-located 2-D images, registered by rigid phase correlation; base in gray, overlay in magenta. Real PET-CT fusion runs on PET-CT uploads (open a PET-CT study)."
      />
      <div className="grid gap-4 px-1 pb-1 md:grid-cols-[1fr_300px]">
        <div className="mx-auto aspect-square w-full max-w-[460px] overflow-hidden rounded-2xl bg-black">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          {f.data ? <img src={f.data.image} alt="Fused image" className="h-full w-full object-contain" /> : <Skeleton className="h-full" />}
        </div>
        <div className="space-y-3 text-xs">
          {[
            ["Base", a, setA],
            ["Overlay", b, setB],
          ].map(([l, v, set]) => (
            <div key={l as string}>
              <Label>{l as string}</Label>
              <NativeSelect value={v as string} onChange={(e) => (set as (x: string) => void)(e.target.value)} className="mt-1 w-full" aria-label={`${l} study`}>
                {studies.map((s) => <option key={s.id} value={s.id}>{s.patient?.pseudonym} · {s.filename}</option>)}
              </NativeSelect>
            </div>
          ))}
          <div>
            <Label>Overlay weight · {fmt(alpha, 2)}</Label>
            <Slider className="mt-2" value={[alpha]} min={0} max={1} step={0.05} onValueChange={([x]) => setAlpha(x)} aria-label="Overlay weight" />
          </div>
          {f.data && (
            <div className="num space-y-0.5 text-muted">
              <div>shift {f.data.shift_px.map((x) => x.toFixed(1)).join(", ")} px</div>
              <div>correlation after registration {fmt(f.data.correlation_after, 3)}</div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

/** Functional MRI (C7): GLM activation maps computed by ml/fmri_glm.py (statistics, not a learned model). */
function FmriCard() {
  const q = useQuery({ queryKey: ["fmri"], queryFn: api.fmri, retry: false });
  const subs = Object.entries(q.data?.subjects ?? {});
  const [sub, setSub] = React.useState<string>("");
  const cur = q.data?.subjects[sub || subs[0]?.[0] || ""];
  return (
    <Card className="p-4 xl:col-span-2" data-testid="fmri-card">
      <CardHeader
        title="Functional MRI — motor task activation"
        subtitle="Finger / foot / lips blocks (OpenNeuro ds000114). Our GLM t-maps on the subject's mean EPI: yellow = Bonferroni-significant, orange = t > 3.1 uncorrected."
      />
      {q.isError ? (
        <p className="px-1 pb-2 text-xs text-muted">fMRI maps not computed yet (python -m ml.fmri_glm).</p>
      ) : !cur ? (
        <Skeleton className="h-48" />
      ) : (
        <div className="space-y-3 px-1 pb-1">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <NativeSelect value={sub || subs[0][0]} onChange={(e) => setSub(e.target.value)} className="h-8 text-xs" aria-label="fMRI subject">
              {subs.map(([k]) => <option key={k} value={k}>{k}</option>)}
            </NativeSelect>
            <Badge tone={cur.somatotopy_pass ? "teal" : "amber"}>
              {cur.somatotopy_pass ? "Somatotopic order holds" : "Somatotopic order not reproduced"}
            </Badge>
            <span className="text-muted">Bonferroni t &gt; {fmt(cur.t_threshold_bonferroni, 2)} · motion regressors {cur.motion_regressors ? "yes" : "not available"}</span>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            {Object.entries(cur.peaks).map(([name, p]) => (
              <figure key={name} className="overflow-hidden rounded-xl border border-border bg-black">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={cur.images[name]} alt={`${name} activation, axial slice through the peak`} className="w-full [image-rendering:pixelated]" />
                <figcaption className="num space-y-0.5 bg-surface px-2 py-1.5 text-[11px] text-muted">
                  <div className="font-medium text-fg">{name}</div>
                  <div>peak (cerebrum) MNI {p.cerebral_mni.map((v) => v.toFixed(0)).join(", ")} · t {fmt(p.cerebral_t, 1)} {p.cerebral_significant ? "" : "(n.s.)"}</div>
                </figcaption>
              </figure>
            ))}
          </div>
          <p className="text-[11px] leading-relaxed text-subtle">{q.data?.caveat}</p>
        </div>
      )}
    </Card>
  );
}

function Row({ label, on, set, children }: { label: string; on: boolean; set: (b: boolean) => void; children?: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <label className="flex items-center justify-between text-sm text-fg">
        {label}
        <Switch checked={on} onCheckedChange={set} aria-label={label} />
      </label>
      {children && <div className="space-y-1">{children}</div>}
    </div>
  );
}

function Q({ label, v, accent }: { label: string; v: string; accent?: boolean }) {
  return (
    <div className="rounded-xl border border-border bg-surface-2 p-3">
      <div className="text-[11px] text-muted">{label}</div>
      <div className={`num mt-1 text-lg font-semibold ${accent ? "text-cyan" : "text-fg"}`}>{v}</div>
    </div>
  );
}
