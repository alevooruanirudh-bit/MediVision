"use client";

import { useQuery } from "@tanstack/react-query";
import { ChevronRight, Download, ScrollText, Search } from "lucide-react";
import * as React from "react";

import { EmptyState, PageHeader } from "@/components/common";
import { Badge, Button, Card, Input, Skeleton } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const ACTOR_TONE: Record<string, "cyan" | "violet" | "teal" | "neutral"> = { ai: "cyan", copilot: "violet", user: "neutral" };

export default function AuditPage() {
  const q = useQuery({ queryKey: ["audit"], queryFn: () => api.audit(), refetchInterval: 15_000 });
  const [filter, setFilter] = React.useState("");
  const [open, setOpen] = React.useState<number | null>(null);
  const rows = (q.data ?? []).filter((r) => !filter || JSON.stringify(r).toLowerCase().includes(filter.toLowerCase()));
  return (
    <div>
      <PageHeader
        eyebrow="Privacy & governance"
        title="Audit log"
        subtitle="Append-only record of uploads (with PHI tags removed), analyses, explanations, Copilot answers, reviews, edits, sign-offs and exports."
        actions={
          <Button variant="secondary" asChild>
            <a href={api.reviewsCsvUrl()}><Download /> Re-training CSV (disagreements)</a>
          </Button>
        }
      />
      <Card className="overflow-hidden">
        <div className="border-b border-border p-4">
          <div className="relative max-w-sm">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-subtle" />
            <Input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter by action, actor, id…" className="pl-9" aria-label="Filter audit log" />
          </div>
        </div>
        {q.isLoading ? (
          <Skeleton className="m-4 h-64" />
        ) : !rows.length ? (
          <EmptyState icon={<ScrollText />} title="No audit entries" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-[11px] uppercase tracking-wider text-subtle">
                  <th className="w-8" />
                  <th className="px-4 py-2 font-medium">Time (UTC)</th>
                  <th className="px-4 py-2 font-medium">Actor</th>
                  <th className="px-4 py-2 font-medium">Action</th>
                  <th className="px-4 py-2 font-medium">Entity</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <React.Fragment key={r.id}>
                    <tr className="cursor-pointer border-b border-border/60 hover:bg-surface-2" onClick={() => setOpen(open === r.id ? null : r.id)}>
                      <td className="pl-3 text-subtle"><ChevronRight className={cn("size-4 transition", open === r.id && "rotate-90")} /></td>
                      <td className="num px-4 py-2 text-xs text-muted">{r.ts.replace("T", " ").slice(0, 19)}</td>
                      <td className="px-4 py-2"><Badge tone={ACTOR_TONE[r.actor] ?? "teal"}>{r.actor}</Badge></td>
                      <td className="num px-4 py-2 text-[13px] text-fg">{r.action}</td>
                      <td className="num px-4 py-2 text-xs text-muted">{r.entity}{r.entity_id ? ` · ${r.entity_id}` : ""}</td>
                    </tr>
                    {open === r.id && (
                      <tr className="border-b border-border/60 bg-surface-2">
                        <td />
                        <td colSpan={4} className="px-4 py-3">
                          <pre className="num whitespace-pre-wrap text-[11px] text-muted">{JSON.stringify(r.details, null, 2)}</pre>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
