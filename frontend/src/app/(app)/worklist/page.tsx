"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, Database, FileUp, Loader2, Search, Trash2, UploadCloud, XCircle } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import * as React from "react";
import { toast } from "sonner";

import { EmptyState, ModalityTag, ModelBadge, PageHeader, UrgencyPill } from "@/components/common";
import { Badge, Button, Card, Input, NativeSelect, Progress, Skeleton } from "@/components/ui/primitives";
import { api, assetUrl, uploadStudy } from "@/lib/api";
import { cn, pct, timeAgo } from "@/lib/utils";

const LEVELS = ["ALL", "CRITICAL", "URGENT", "ROUTINE", "NORMAL"] as const;

interface UploadItem {
  id: string;
  name: string;
  progress: number;
  state: "uploading" | "analysing" | "done" | "error";
  error?: string;
  studyId?: string;
}

function Uploader({ onDone }: { onDone: () => void }) {
  const [drag, setDrag] = React.useState(false);
  const [hint, setHint] = React.useState("");
  const [items, setItems] = React.useState<UploadItem[]>([]);
  const input = React.useRef<HTMLInputElement>(null);

  const upload = async (files: FileList | File[]) => {
    const list = Array.from(files);
    for (const f of list) {
      const id = `${f.name}-${Math.random().toString(36).slice(2, 7)}`;
      setItems((s) => [{ id, name: f.name, progress: 0, state: "uploading" } as UploadItem, ...s].slice(0, 8));
      const set = (p: Partial<UploadItem>) => setItems((s) => s.map((x) => (x.id === id ? { ...x, ...p } : x)));
      try {
        const study = await uploadStudy(f, { modality_hint: hint || undefined }, (p) => {
          set({ progress: p, state: p >= 1 ? "analysing" : "uploading" });
        });
        set({ state: "done", progress: 1, studyId: study.id });
        onDone();
      } catch (e) {
        set({ state: "error", error: (e as Error).message });
      }
    }
  };

  return (
    <Card className="p-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (e.dataTransfer.files.length) upload(e.dataTransfer.files);
        }}
        onClick={() => input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        aria-label="Upload studies"
        className={cn(
          "relative flex cursor-pointer flex-col items-center justify-center gap-2 overflow-hidden rounded-xl border border-dashed px-6 py-8 text-center transition-all",
          drag ? "border-cyan bg-cyan/10" : "border-border-strong hover:border-cyan/60 hover:bg-surface-2",
        )}
      >
        {drag && <div className="scan-line" />}
        <UploadCloud className={cn("size-8 transition", drag ? "text-cyan" : "text-subtle")} />
        <div className="text-sm font-medium text-fg">Drop chest X-rays, bone X-rays or brain MRI here</div>
        <div className="text-xs text-muted">PNG · JPG · DICOM ≤ 20 MB · multiple files · DICOM is de-identified on arrival</div>
        <input
          ref={input}
          type="file"
          multiple
          accept=".png,.jpg,.jpeg,.dcm,.dicom"
          className="hidden"
          data-testid="file-input"
          onChange={(e) => e.target.files && upload(e.target.files)}
        />
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted">
        <span>Modality hint (optional — the router decides):</span>
        <NativeSelect value={hint} onChange={(e) => setHint(e.target.value)} className="h-8 text-xs" aria-label="Modality hint">
          <option value="">Auto-detect</option>
          <option value="chest">Chest X-ray</option>
          <option value="bone">Bone X-ray</option>
          <option value="brain">Brain MRI</option>
        </NativeSelect>
      </div>
      <AnimatePresence initial={false}>
        {items.map((it) => (
          <motion.div
            key={it.id}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2 flex items-center gap-3 rounded-xl border border-border bg-surface-2 px-3 py-2 text-xs"
          >
            {it.state === "done" ? (
              <CheckCircle2 className="size-4 text-teal" />
            ) : it.state === "error" ? (
              <XCircle className="size-4 text-red" />
            ) : (
              <Loader2 className="size-4 animate-spin text-cyan" />
            )}
            <span className="max-w-[40%] truncate text-fg">{it.name}</span>
            <div className="flex-1">
              {it.state === "error" ? (
                <span className="text-red">{it.error}</span>
              ) : (
                <Progress value={it.state === "analysing" ? 100 : it.progress * 100} indicatorClassName={it.state === "analysing" ? "animate-pulse" : ""} />
              )}
            </div>
            <span className="w-20 text-right text-muted">
              {it.state === "uploading" ? pct(it.progress) : it.state === "analysing" ? "analysing…" : it.state === "done" ? "" : "failed"}
            </span>
            {it.studyId && (
              <Link href={`/studies/${it.studyId}`} className="text-cyan hover:underline">
                Open
              </Link>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </Card>
  );
}

export default function WorklistPage() {
  const qc = useQueryClient();
  const router = useRouter();
  const [level, setLevel] = React.useState<(typeof LEVELS)[number]>("ALL");
  const [q, setQ] = React.useState("");
  const [modality, setModality] = React.useState("");
  const wl = useQuery({
    queryKey: ["worklist", level, q, modality],
    queryFn: () => api.worklist({ level: level === "ALL" ? undefined : level, q: q || undefined, modality: modality || undefined }),
    refetchInterval: 15_000,
  });
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["worklist"] });
    qc.invalidateQueries({ queryKey: ["dashboard"] });
  };
  const loadDemo = async () => {
    const t = toast.loading("Loading demo studies — each one is analysed by our models…");
    try {
      const r = await api.loadDemo();
      toast.success(`Demo data loaded (${r.created} new, ${r.skipped} already present)`, { id: t });
      refresh();
    } catch (e) {
      toast.error((e as Error).message, { id: t });
    }
  };
  const del = async (id: string) => {
    await api.deleteStudy(id);
    toast.success("Study deleted");
    refresh();
  };
  const counts = wl.data?.counts ?? {};

  return (
    <div>
      <PageHeader
        eyebrow="Triage"
        title="Worklist"
        subtitle="Studies ordered by AI urgency. CRITICAL studies pulse. Click a row to open the viewer."
        actions={
          <Button variant="secondary" onClick={loadDemo}>
            <Database /> Load demo data
          </Button>
        }
      />
      <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
        <Card className="overflow-hidden">
          <div className="flex flex-col gap-3 border-b border-border p-4 md:flex-row md:items-center">
            <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Filter by urgency">
              {LEVELS.map((l) => (
                <button
                  key={l}
                  role="tab"
                  aria-selected={level === l}
                  onClick={() => setLevel(l)}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs font-medium transition",
                    level === l ? "border-cyan/50 bg-cyan/10 text-cyan" : "border-border text-muted hover:text-fg",
                  )}
                >
                  {l === "ALL" ? "All" : l}
                  {l !== "ALL" && counts[l] != null && <span className="num ml-1.5 opacity-70">{counts[l]}</span>}
                </button>
              ))}
            </div>
            <div className="flex flex-1 items-center gap-2 md:justify-end">
              <NativeSelect value={modality} onChange={(e) => setModality(e.target.value)} className="h-9 text-xs" aria-label="Modality filter">
                <option value="">All modalities</option>
                <option value="chest">Chest X-ray</option>
                <option value="bone">Bone X-ray</option>
                <option value="brain">Brain MRI</option>
                <option value="not">Not medical</option>
              </NativeSelect>
              <div className="relative w-full max-w-xs">
                <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-subtle" />
                <Input placeholder="Search pseudonym, finding…" value={q} onChange={(e) => setQ(e.target.value)} className="pl-9" aria-label="Search" />
              </div>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-[11px] uppercase tracking-wider text-subtle">
                  <th className="px-4 py-2.5 font-medium">Urgency</th>
                  <th className="px-4 py-2.5 font-medium">Patient</th>
                  <th className="px-4 py-2.5 font-medium">Modality</th>
                  <th className="px-4 py-2.5 font-medium">Top finding</th>
                  <th className="px-4 py-2.5 font-medium">Model</th>
                  <th className="px-4 py-2.5 font-medium">Received</th>
                  <th className="px-2 py-2.5" />
                </tr>
              </thead>
              <tbody>
                {wl.isLoading &&
                  Array.from({ length: 6 }, (_, i) => (
                    <tr key={i}>
                      <td colSpan={7} className="px-4 py-2">
                        <Skeleton className="h-10" />
                      </td>
                    </tr>
                  ))}
                {wl.data?.items.map((s, i) => (
                  <motion.tr
                    key={s.id}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: Math.min(i * 0.02, 0.3) }}
                    onClick={() => router.push(`/studies/${s.id}`)}
                    className="group cursor-pointer border-b border-border/60 transition-colors hover:bg-surface-2"
                    data-testid="worklist-row"
                  >
                    <td className="px-4 py-2.5">
                      <UrgencyPill level={s.urgency_level} score={s.urgency_score} />
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-3">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={assetUrl(s.image_url)} alt="" className="size-9 rounded-lg border border-border object-cover" />
                        <div>
                          <div className="num text-[13px] text-fg">{s.patient?.pseudonym}</div>
                          <div className="text-[11px] text-subtle">
                            {[s.patient?.sex, s.patient?.age && `${s.patient.age}y`].filter(Boolean).join(" · ") || s.filename}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-2.5">
                      <ModalityTag modality={s.modality} />
                      {s.ood && <Badge tone="amber" className="ml-2">OOD</Badge>}
                    </td>
                    <td className="px-4 py-2.5">
                      {s.top_finding ? (
                        <span className="text-fg">
                          {s.top_finding.label}
                          <span className="num ml-2 text-xs text-muted">{pct(s.top_finding.probability)}</span>
                        </span>
                      ) : (
                        <span className="text-muted">{s.pipeline ? "No finding above threshold" : "Not analysed"}</span>
                      )}
                    </td>
                    <td className="px-4 py-2.5">
                      <ModelBadge profile={s.model_profile} />
                    </td>
                    <td className="num px-4 py-2.5 text-xs text-muted">{timeAgo(s.created_at)}</td>
                    <td className="px-2 py-2.5">
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label="Delete study"
                        className="opacity-0 group-hover:opacity-100"
                        onClick={(e) => {
                          e.stopPropagation();
                          del(s.id);
                        }}
                      >
                        <Trash2 />
                      </Button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
            {wl.data && !wl.data.items.length && (
              <EmptyState
                icon={<FileUp />}
                title="The worklist is empty"
                body="Drop images on the right, or load the demo studies (chest, fracture, brain MRI and one non-medical photo)."
                action={<Button variant="primary" onClick={loadDemo}>Load demo data</Button>}
              />
            )}
          </div>
        </Card>
        <div className="space-y-4">
          <Uploader onDone={refresh} />
          <Card className="p-4 text-xs leading-relaxed text-muted">
            <div className="mb-1.5 font-medium text-fg">Triage levels</div>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2"><UrgencyPill level="CRITICAL" /> score ≥ 75</div>
              <div className="flex items-center gap-2"><UrgencyPill level="URGENT" /> score ≥ 45</div>
              <div className="flex items-center gap-2"><UrgencyPill level="ROUTINE" /> any positive finding or doubtful input</div>
              <div className="flex items-center gap-2"><UrgencyPill level="NORMAL" /> nothing above threshold</div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
