"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Building2, Hospital, Loader2, Play, Server, ShieldCheck } from "lucide-react";
import * as React from "react";
import { toast } from "sonner";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { ChartTooltip, Legend, SERIES, axis, grid } from "@/components/charts";
import { PageHeader } from "@/components/common";
import { Badge, Button, Card, CardHeader, Label, Slider } from "@/components/ui/primitives";
import { api, streamSSE } from "@/lib/api";
import { fmt, pct } from "@/lib/utils";

interface HospitalInfo {
  name: string;
  n: number;
  pneumonia_rate: number;
  scanner: string;
}
interface RoundEv {
  round: number;
  global: { accuracy: number; auroc: number };
  hospitals: { name: string; auroc: number; accuracy: number; loss: number }[];
  elapsed_s: number;
}

function Diagram({ hospitals, active, round }: { hospitals: HospitalInfo[]; active: boolean; round: number }) {
  const pos = [
    { x: 90, y: 60 },
    { x: 90, y: 180 },
    { x: 90, y: 300 },
  ];
  const server = { x: 470, y: 180 };
  return (
    <svg viewBox="0 0 600 360" className="h-full w-full" role="img" aria-label="Three hospitals exchange model weights with a federated server">
      <defs>
        <linearGradient id="fl" x1="0" x2="1">
          <stop offset="0" stopColor="var(--cyan)" />
          <stop offset="1" stopColor="var(--violet)" />
        </linearGradient>
      </defs>
      {pos.map((p, i) => (
        <g key={i}>
          <path d={`M${p.x + 70} ${p.y} C 300 ${p.y}, 330 ${server.y}, ${server.x - 60} ${server.y}`} fill="none" stroke="var(--border-strong)" strokeWidth="1.5" strokeDasharray="4 5" />
          {active && (
            <>
              <motion.circle
                r="5"
                fill="var(--cyan)"
                initial={{ offsetDistance: "0%" }}
                animate={{ offsetDistance: ["0%", "100%"] }}
                transition={{ duration: 1.4, repeat: Infinity, delay: i * 0.25, ease: "easeInOut" }}
                style={{ offsetPath: `path("M${p.x + 70} ${p.y} C 300 ${p.y}, 330 ${server.y}, ${server.x - 60} ${server.y}")` } as React.CSSProperties}
              />
              <motion.circle
                r="4"
                fill="var(--violet)"
                initial={{ offsetDistance: "100%" }}
                animate={{ offsetDistance: ["100%", "0%"] }}
                transition={{ duration: 1.4, repeat: Infinity, delay: 0.7 + i * 0.25, ease: "easeInOut" }}
                style={{ offsetPath: `path("M${p.x + 70} ${p.y} C 300 ${p.y}, 330 ${server.y}, ${server.x - 60} ${server.y}")` } as React.CSSProperties}
              />
            </>
          )}
          <foreignObject x={p.x - 80} y={p.y - 38} width="160" height="76">
            <div className="glass flex h-full items-center gap-2.5 rounded-2xl px-3">
              <div className="grid size-9 shrink-0 place-items-center rounded-xl bg-cyan/10 text-cyan">{i === 2 ? <Building2 className="size-4" /> : <Hospital className="size-4" />}</div>
              <div className="min-w-0 text-[11px] leading-tight">
                <div className="truncate font-medium text-fg">{hospitals[i]?.name ?? `Hospital ${i + 1}`}</div>
                <div className="num text-muted">{hospitals[i] ? `${hospitals[i].n} imgs · ${pct(hospitals[i].pneumonia_rate)} +` : "local data"}</div>
                <div className="truncate text-subtle">{hospitals[i]?.scanner ?? ""}</div>
              </div>
            </div>
          </foreignObject>
        </g>
      ))}
      <foreignObject x={server.x - 70} y={server.y - 52} width="140" height="104">
        <div className="glass flex h-full flex-col items-center justify-center rounded-2xl border-cyan/40 text-center">
          <motion.div animate={active ? { rotate: 360 } : {}} transition={{ duration: 6, repeat: Infinity, ease: "linear" }} className="grid size-10 place-items-center rounded-full bg-accent-gradient text-white">
            <Server className="size-5" />
          </motion.div>
          <div className="mt-1.5 text-xs font-semibold text-fg">FedAvg server</div>
          <div className="num text-[11px] text-muted">round {round}</div>
        </div>
      </foreignObject>
      <text x="300" y="350" textAnchor="middle" fill="var(--subtle)" fontSize="11">only model weights travel — images never leave a hospital</text>
    </svg>
  );
}

export default function FederatedPage() {
  const qc = useQueryClient();
  const [rounds, setRounds] = React.useState(10);
  const [running, setRunning] = React.useState(false);
  const [hospitals, setHospitals] = React.useState<HospitalInfo[]>([]);
  const [evs, setEvs] = React.useState<RoundEv[]>([]);
  const [central, setCentral] = React.useState<{ accuracy: number; auroc: number }[]>([]);
  const [local, setLocal] = React.useState<Record<string, { accuracy: number; auroc: number }>>({});
  const [elapsed, setElapsed] = React.useState(0);

  React.useEffect(() => {
    api.federatedLast().then((s) => {
      const last = s as { hospitals?: HospitalInfo[]; federated_curve?: { accuracy: number; auroc: number }[]; centralized_curve?: { accuracy: number; auroc: number }[]; local_only?: Record<string, { accuracy: number; auroc: number }>; seconds?: number };
      if (!last?.federated_curve) return;
      setHospitals(last.hospitals ?? []);
      setEvs(last.federated_curve.map((g, i) => ({ round: i + 1, global: g, hospitals: [], elapsed_s: 0 })));
      setCentral(last.centralized_curve ?? []);
      setLocal(last.local_only ?? {});
      setElapsed(last.seconds ?? 0);
    }).catch(() => undefined);
  }, []);

  const run = async () => {
    setRunning(true);
    setEvs([]);
    setCentral([]);
    setLocal({});
    try {
      await streamSSE("/api/federated/run", { rounds, local_epochs: 1 }, (event, data) => {
        const d = data as Record<string, unknown>;
        if (event === "setup") setHospitals(d.hospitals as HospitalInfo[]);
        if (event === "round") {
          setEvs((e) => [...e, d as unknown as RoundEv]);
          setElapsed(d.elapsed_s as number);
        }
        if (event === "baseline" && d.name === "centralized") setCentral(d.curve as { accuracy: number; auroc: number }[]);
        if (event === "baseline" && d.name === "local_only") setLocal(d.results as Record<string, { accuracy: number; auroc: number }>);
        if (event === "done") {
          toast.success("Federated run complete");
          qc.invalidateQueries({ queryKey: ["federated-last"] });
        }
      });
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const curve = evs.map((e, i) => ({ round: e.round, federated: e.global.auroc, centralized: central[i]?.auroc }));
  const localMean = Object.values(local).length ? Object.values(local).reduce((a, b) => a + b.auroc, 0) / Object.values(local).length : null;
  const final = [
    { name: "Federated (FedAvg)", auroc: evs.at(-1)?.global.auroc },
    { name: "Centralised (pooled)", auroc: central.at(-1)?.auroc },
    ...Object.entries(local).map(([k, v]) => ({ name: `Local only · ${k.split(" ")[0]}`, auroc: v.auroc })),
  ].filter((x) => x.auroc != null);

  return (
    <div>
      <PageHeader
        eyebrow="Federated learning · V"
        title="Federated training across 3 hospitals"
        subtitle="Real FedAvg on PneumoniaMNIST (64 px) with non-IID label mixes and scanner styles. Compared against centralised training (pooling data — usually not allowed) and each hospital training alone."
        actions={
          <Button variant="primary" onClick={run} disabled={running} data-testid="run-federated">
            {running ? <Loader2 className="animate-spin" /> : <Play />} {running ? `Round ${evs.length}/${rounds}…` : "Run federated training"}
          </Button>
        }
      />
      <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <Card className="p-4">
          <div className="aspect-[5/3] w-full">
            <Diagram hospitals={hospitals} active={running} round={evs.length} />
          </div>
          <div className="mt-2 flex items-center gap-4 px-2">
            <Label>Rounds · {rounds}</Label>
            <Slider className="max-w-56" value={[rounds]} min={3} max={20} step={1} onValueChange={([v]) => setRounds(v)} disabled={running} aria-label="Rounds" />
            <span className="num ml-auto text-xs text-subtle">{elapsed ? `${fmt(elapsed, 1)} s on CPU` : ""}</span>
          </div>
        </Card>
        <Card>
          <CardHeader title="Global model AUROC per round" subtitle="Held-out test set mixing all three scanner styles" action={<Legend items={[{ label: "Federated", color: SERIES[0] }, { label: "Centralised", color: SERIES[1], dashed: true }]} />} />
          <div className="h-72 px-2 pb-4">
            <ResponsiveContainer>
              <LineChart data={curve} margin={{ left: 0, right: 16, top: 6 }}>
                <CartesianGrid {...grid} />
                <XAxis dataKey="round" {...axis} />
                <YAxis {...axis} width={40} domain={[0.5, 1]} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip content={<ChartTooltip format={(v) => Number(v).toFixed(3)} labelFormat={(l) => `Round ${l}`} />} cursor={{ stroke: "var(--border-strong)" }} />
                <Line type="monotone" dataKey="federated" name="Federated" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="centralized" name="Centralised" stroke={SERIES[1]} strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_1fr]">
        <Card>
          <CardHeader title="Final comparison" subtitle="Test AUROC after the same number of local epochs" />
          <div className="h-64 px-2 pb-4">
            {final.length ? (
              <ResponsiveContainer>
                <BarChart data={final} layout="vertical" margin={{ left: 8, right: 24 }}>
                  <CartesianGrid {...grid} horizontal={false} vertical />
                  <XAxis type="number" {...axis} domain={[0.5, 1]} tickFormatter={(v) => v.toFixed(2)} />
                  <YAxis type="category" dataKey="name" {...axis} width={150} />
                  <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => Number(v).toFixed(3)} />} />
                  <Bar dataKey="auroc" name="AUROC" fill={SERIES[0]} radius={[0, 4, 4, 0]} maxBarSize={22} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="p-6 text-sm text-muted">Run the experiment to see results.</p>
            )}
          </div>
        </Card>
        <Card className="p-5 text-sm leading-relaxed text-muted">
          <div className="mb-2 font-medium text-fg">What this shows</div>
          <ul className="list-disc space-y-1.5 pl-5">
            <li>Each round, every hospital trains a copy of the global CNN on its own images for one epoch; the server averages the weights, weighted by dataset size.</li>
            <li>The rural clinic has few pneumonia cases and a noisy, low-contrast scanner — alone it learns a poor model{localMean != null ? ` (mean local-only AUROC ${fmt(localMean, 3)})` : ""}.</li>
            <li>FedAvg approaches centralised performance without moving a single image.</li>
          </ul>
          <div className="mt-3 rounded-xl border border-teal/30 bg-teal/10 p-3 text-xs text-teal">
            <div className="mb-1 flex items-center gap-1.5 font-medium"><ShieldCheck className="size-3.5" /> Privacy &amp; compliance</div>
            Only model weights leave each hospital — never images or labels. This simulation has no secure aggregation or
            differential privacy (weights can leak information); a real deployment would add both, and HIPAA (US) / GDPR (EU)
            duties — lawful basis, data-processing agreements, minimisation, audit — still apply.
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge tone="cyan">McMahan et al. 2017</Badge>
            <Badge tone="violet">non-IID</Badge>
            <Badge tone="teal">CPU · &lt; 2 min</Badge>
          </div>
        </Card>
      </div>
      <PerSiteCard />
    </div>
  );
}

/** Generalisation & site bias (S4): every model evaluated on each hospital's own scanner style. */
function PerSiteCard() {
  const q = useQuery({ queryKey: ["federated-last"], queryFn: api.federatedLast });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ps = (q.data as any)?.per_site as
    | { auroc: Record<string, Record<string, number>>; summary: Record<string, { worst_site_auroc: number; site_gap: number }> }
    | undefined;
  if (!ps) return null;
  const sites = Object.keys(Object.values(ps.auroc)[0] ?? {});
  const best = Math.max(...Object.values(ps.summary).map((v) => v.worst_site_auroc));
  return (
    <Card className="mt-4 overflow-hidden" data-testid="per-site">
      <CardHeader
        title="Generalisation across hospitals"
        subtitle="Test AUROC on each hospital's scanner style. Worst-site AUROC and the best–worst gap show how fair a model is to every site."
      />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-y border-border text-left text-[11px] uppercase tracking-wider text-subtle">
              <th className="px-5 py-2 font-medium">Model</th>
              {sites.map((s) => <th key={s} className="px-3 py-2 text-right font-medium">{s}</th>)}
              <th className="px-3 py-2 text-right font-medium">Worst site</th>
              <th className="px-5 py-2 text-right font-medium">Gap</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(ps.auroc).map(([m, row]) => (
              <tr key={m} className="border-b border-border/60">
                <td className="px-5 py-2 text-fg">{m}</td>
                {sites.map((s) => <td key={s} className="num px-3 py-2 text-right text-muted">{fmt(row[s], 3)}</td>)}
                <td className={`num px-3 py-2 text-right ${ps.summary[m].worst_site_auroc === best ? "font-semibold text-teal" : "text-fg"}`}>{fmt(ps.summary[m].worst_site_auroc, 3)}</td>
                <td className="num px-5 py-2 text-right text-fg">{fmt(ps.summary[m].site_gap, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="px-5 py-3 text-xs text-muted">
        The rural clinic&apos;s own model is weakest everywhere; FedAvg lifts the worst site without pooling any images — training on
        diverse sites improves generalisation and narrows the gap between hospitals.
      </p>
    </Card>
  );
}

