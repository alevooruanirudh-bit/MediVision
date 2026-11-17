"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Brain, Bone, Cpu, ImageOff, ScanLine, ShieldAlert } from "lucide-react";
import * as React from "react";

import { Badge } from "@/components/ui/primitives";
import type { Level } from "@/lib/types";
import { cn, modalityLabel } from "@/lib/utils";

export const DISCLAIMER = "Research & educational prototype — not a medical device. Not for clinical use.";

export function DisclaimerBanner({ compact = false }: { compact?: boolean }) {
  return (
    <div
      role="note"
      className={cn(
        "flex h-[29px] items-center justify-center gap-2 overflow-hidden border-b border-amber/25 bg-[color-mix(in_srgb,var(--amber)_9%,var(--bg))] px-3 text-amber",
        compact ? "text-[11px]" : "text-[11px] sm:text-xs",
      )}
    >
      <ShieldAlert className="size-3.5 shrink-0" />
      <span className="hidden truncate font-medium sm:inline">{DISCLAIMER}</span>
      <span className="truncate font-medium sm:hidden">Prototype — not a medical device, not for clinical use.</span>
    </div>
  );
}

const LEVEL_TONE: Record<Level, "red" | "amber" | "cyan" | "teal"> = {
  CRITICAL: "red",
  URGENT: "amber",
  ROUTINE: "cyan",
  NORMAL: "teal",
};

export function UrgencyPill({ level, score, className }: { level: Level | null; score?: number | null; className?: string }) {
  if (!level) return <Badge className={className}>PENDING</Badge>;
  return (
    <Badge tone={LEVEL_TONE[level]} className={cn("font-semibold", level === "CRITICAL" && "pulse-critical", className)}>
      <span
        className={cn(
          "size-1.5 rounded-full",
          level === "CRITICAL" ? "bg-red" : level === "URGENT" ? "bg-amber" : level === "ROUTINE" ? "bg-cyan" : "bg-teal",
        )}
      />
      {level}
      {score != null && <span className="num opacity-75">{Math.round(score)}</span>}
    </Badge>
  );
}

export function ModelBadge({ profile }: { profile: string | null | undefined }) {
  if (!profile) return null;
  return (
    <Badge tone={profile === "full" ? "violet" : "cyan"} title="Active model weights">
      <Cpu className="size-3" />
      {profile === "full" ? "Full model" : "Starter model"}
    </Badge>
  );
}

export function ModalityIcon({ modality, className }: { modality: string | null; className?: string }) {
  const c = cn("size-4", className);
  if (modality === "brain_mri") return <Brain className={c} />;
  if (modality === "bone_xray") return <Bone className={c} />;
  if (modality === "not_medical" || modality === "other_medical") return <ImageOff className={c} />;
  return <ScanLine className={c} />;
}

export function ModalityTag({ modality }: { modality: string | null }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted">
      <ModalityIcon modality={modality} className="size-3.5 text-cyan" />
      {modality ? (modalityLabel[modality] ?? modality) : "—"}
    </span>
  );
}

export function OODBanner({ message }: { message: string | null | undefined }) {
  if (!message) return null;
  return (
    <motion.div
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-start gap-2.5 rounded-xl border border-amber/35 bg-amber/10 px-3.5 py-2.5 text-sm text-amber"
      role="alert"
    >
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <span>{message}</span>
    </motion.div>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
  eyebrow,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  eyebrow?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div>
        {eyebrow && <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan">{eyebrow}</div>}
        <h1 className="text-2xl font-semibold tracking-tight text-fg md:text-[28px]">{title}</h1>
        {subtitle && <p className="mt-1.5 max-w-2xl text-sm text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function StatCard({
  label,
  value,
  hint,
  icon,
  tone = "cyan",
  delay = 0,
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  icon?: React.ReactNode;
  tone?: "cyan" | "violet" | "teal" | "amber" | "red";
  delay?: number;
}) {
  const toneCls = { cyan: "text-cyan", violet: "text-violet", teal: "text-teal", amber: "text-amber", red: "text-red" }[tone];
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="glass group relative overflow-hidden rounded-2xl p-4"
    >
      <div className="absolute -right-6 -top-6 size-24 rounded-full bg-accent-gradient opacity-[0.07] blur-2xl transition-opacity group-hover:opacity-[0.14]" />
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted">{label}</span>
        <span className={cn("[&_svg]:size-4", toneCls)}>{icon}</span>
      </div>
      <div className="num mt-2 text-2xl font-semibold tracking-tight text-fg">{value}</div>
      {hint && <div className="mt-1 text-[11px] text-subtle">{hint}</div>}
    </motion.div>
  );
}

export function EmptyState({ icon, title, body, action }: { icon?: React.ReactNode; title: string; body?: string; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
      <div className="grid size-12 place-items-center rounded-2xl border border-border bg-surface-2 text-cyan [&_svg]:size-5">{icon}</div>
      <div className="text-sm font-semibold text-fg">{title}</div>
      {body && <p className="max-w-sm text-sm text-muted">{body}</p>}
      {action}
    </div>
  );
}

export function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-subtle">{children}</div>;
}
