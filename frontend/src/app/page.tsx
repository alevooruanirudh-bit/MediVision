"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Brain,
  CloudOff,
  Eye,
  FileText,
  GitCompareArrows,
  Gauge,
  Layers,
  Lock,
  MessagesSquare,
  Network,
  Scale,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Users,
  Wand2,
  Wallet,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { DISCLAIMER, DisclaimerBanner } from "@/components/common";
import { HeroXray } from "@/components/landing/hero-xray";
import { Logo } from "@/components/shell/logo";
import { Badge, Button } from "@/components/ui/primitives";
import { api } from "@/lib/api";

const PILLARS = [
  { tag: "III-A", icon: ScanLine, title: "Diagnostic accuracy", body: "Our own DenseNet-121 + EfficientNet-B2 + ConvNeXt-Tiny ensemble for 14 chest findings, a FracAtlas fracture detector and a 4-class brain-tumour MRI classifier." },
  { tag: "III-C", icon: Layers, title: "Segmentation", body: "Self-trained U-Net (lungs) and Attention U-Net (brain tumour) with automatic cardiothoracic ratio, lung areas and tumour area." },
  { tag: "III-D", icon: Gauge, title: "Triage", body: "Urgency 0–100 from calibrated probabilities, severity weights and uncertainty → CRITICAL / URGENT / ROUTINE / NORMAL worklist." },
  { tag: "III-E", icon: MessagesSquare, title: "Reporting & NLP", body: "MediVision Copilot: a self-built clinical language engine — structured report writer, intent model and retrieval. No LLM." },
  { tag: "III-F", icon: Wand2, title: "Advanced imaging", body: "Our DnCNN denoiser trained on simulated low-dose noise, plus CLAHE, unsharp masking and gamma with PSNR / SSIM." },
  { tag: "III-G", icon: GitCompareArrows, title: "Personalised", body: "Longitudinal comparison: per-finding deltas, CTR change, heatmap difference and an interval-change paragraph." },
  { tag: "IV", icon: Eye, title: "Explainability", body: "Grad-CAM, Grad-CAM++ and occlusion maps, ensemble + MC-dropout uncertainty and temperature-scaled calibration." },
  { tag: "IV", icon: Scale, title: "Bias & data quality", body: "Fairness dashboard by sex and age group, a modality router and an energy-score out-of-distribution guard." },
  { tag: "IV", icon: Lock, title: "Privacy & interoperability", body: "DICOM de-identification, fully local inference, audit log. DICOM in, FHIR-style JSON + PDF out." },
  { tag: "V", icon: Network, title: "Federated learning", body: "Real FedAvg across three simulated hospitals with non-IID data and live charts." },
  { tag: "V", icon: Users, title: "Human–AI collaboration", body: "Accept, reject or edit each finding; disagreements are logged and exportable as a re-training set." },
  { tag: "Ops", icon: Brain, title: "Starter → full models", body: "Laptop-trained starter weights work today; one command trains full models on Kaggle's free GPU." },
];

const STEPS = [
  ["Upload", "PNG / JPG / DICOM, de-identified on arrival"],
  ["Route", "Modality router + OOD guard pick the pipeline"],
  ["Analyse", "Ensemble, segmentation, detection on CPU (ONNX)"],
  ["Explain", "Grad-CAM, occlusion, uncertainty, zones"],
  ["Triage", "Urgency score orders the worklist"],
  ["Report", "Copilot drafts, you review and sign"],
];

export default function Landing() {
  const router = useRouter();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 20_000 });
  const loadDemo = async () => {
    const t = toast.loading("Loading demo studies (analysing with our models)…");
    try {
      const r = await api.loadDemo();
      toast.success(`Demo ready: ${r.created} new studies`, { id: t });
      router.push("/worklist");
    } catch (e) {
      toast.error(`Could not load demo: ${(e as Error).message}`, { id: t });
    }
  };
  return (
    <div className="relative min-h-dvh overflow-hidden">
      <div className="sticky top-0 z-40">
        <DisclaimerBanner />
      </div>
      <div className="app-aurora" />
      <div className="bg-grid pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_at_top,black,transparent_70%)]" />
      <header className="relative z-10 mx-auto flex max-w-7xl items-center justify-between px-5 py-5 md:px-8">
        <Logo />
        <nav className="flex items-center gap-2">
          <Link href="/lab" className="hidden px-3 text-sm text-muted transition hover:text-fg sm:block">Metrics</Link>
          <Link href="/model-card" className="hidden px-3 text-sm text-muted transition hover:text-fg sm:block">Model card</Link>
          <Button asChild variant="primary" size="sm">
            <Link href="/dashboard">Open app <ArrowRight /></Link>
          </Button>
        </nav>
      </header>

      <section className="relative z-10 mx-auto grid max-w-7xl items-center gap-12 px-5 pb-20 pt-8 md:px-8 lg:grid-cols-[1.1fr_0.9fr] lg:pt-16">
        <div>
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-wrap gap-2">
            <Badge tone="cyan"><Sparkles className="size-3" /> 100% self-trained</Badge>
            <Badge tone="violet"><CloudOff className="size-3" /> Runs offline</Badge>
            <Badge tone="teal"><Wallet className="size-3" /> Zero cost</Badge>
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="mt-6 text-4xl font-semibold leading-[1.05] tracking-tight text-fg sm:text-5xl lg:text-[64px]"
          >
            The imaging copilot that <span className="text-gradient">shows its work.</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.16, duration: 0.6 }}
            className="mt-5 max-w-xl text-base leading-relaxed text-muted md:text-lg"
          >
            MediVision AI reads chest X-rays, bone X-rays and brain MRI with models we designed and trained on free public
            data, explains every prediction, triages the worklist and drafts the report — with no external AI service
            anywhere in the loop.
          </motion.p>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="mt-8 flex flex-wrap gap-3">
            <Button asChild variant="primary" size="lg">
              <Link href="/dashboard">Launch dashboard <ArrowRight /></Link>
            </Button>
            <Button variant="secondary" size="lg" onClick={loadDemo}>
              Load demo data
            </Button>
          </motion.div>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }} className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-muted">
            <span className="flex items-center gap-2">
              <span className={`size-2 rounded-full ${health.data ? "bg-teal shadow-[0_0_10px_var(--teal)]" : "bg-red"}`} />
              {health.data ? `Engine online · ${health.data.n_models_loaded} models loaded` : "Engine offline — start the backend"}
            </span>
            {health.data && (
              <>
                <span className="num">weights: {health.data.active_profile}</span>
                <span className="num">runtime: {health.data.inference_backend} / CPU</span>
                <span className="num text-teal">external AI APIs: {health.data.external_ai_apis}</span>
              </>
            )}
          </motion.div>
        </div>
        <div className="flex justify-center lg:justify-end">
          <HeroXray />
        </div>
      </section>

      <section className="relative z-10 mx-auto max-w-7xl px-5 pb-20 md:px-8">
        <div className="mb-8 max-w-2xl">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan">Mapped to the paper</div>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight text-fg md:text-3xl">Every pillar, implemented — not mocked.</h2>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {PILLARS.map((p, i) => (
            <motion.div
              key={p.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ delay: (i % 3) * 0.06, duration: 0.5 }}
              className="glass group relative overflow-hidden rounded-2xl p-5 transition-colors hover:border-border-strong"
            >
              <div className="absolute -right-10 -top-10 size-32 rounded-full bg-accent-gradient opacity-0 blur-3xl transition-opacity duration-500 group-hover:opacity-20" />
              <div className="flex items-center justify-between">
                <div className="grid size-10 place-items-center rounded-xl border border-border bg-surface-2 text-cyan">
                  <p.icon className="size-5" />
                </div>
                <span className="num text-[10px] text-subtle">{p.tag}</span>
              </div>
              <h3 className="mt-4 text-[15px] font-semibold text-fg">{p.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-muted">{p.body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="relative z-10 mx-auto max-w-7xl px-5 pb-24 md:px-8">
        <h2 className="mb-8 text-2xl font-semibold tracking-tight text-fg md:text-3xl">How it works</h2>
        <ol className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
          {STEPS.map(([t, d], i) => (
            <motion.li
              key={t}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.07 }}
              className="relative"
            >
              <div className="glass h-full rounded-2xl p-4">
                <div className="num grid size-8 place-items-center rounded-full bg-accent-gradient text-xs font-semibold text-white">{i + 1}</div>
                <div className="mt-3 text-sm font-semibold text-fg">{t}</div>
                <div className="mt-1 text-xs leading-relaxed text-muted">{d}</div>
              </div>
            </motion.li>
          ))}
        </ol>
      </section>

      <footer className="relative z-10 border-t border-border">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-5 py-8 text-xs text-muted md:flex-row md:items-center md:justify-between md:px-8">
          <div className="flex items-center gap-2">
            <ShieldCheck className="size-4 text-amber" />
            {DISCLAIMER}
          </div>
          <div className="flex items-center gap-1.5">
            <FileText className="size-3.5" /> Datasets: NIH ChestX-ray14, FracAtlas, Brain Tumor MRI, LGG, IU X-ray, MedMNIST+ — see the model card.
          </div>
        </div>
      </footer>
    </div>
  );
}
