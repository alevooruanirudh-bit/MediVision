"use client";

import { motion } from "framer-motion";
import { Building2, Hospital, Loader2, Play, Server } from "lucide-react";
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
        if (event === "done") toast.success("Federated run complete");
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
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge tone="cyan">McMahan et al. 2017</Badge>
            <Badge tone="violet">non-IID</Badge>
            <Badge tone="teal">CPU · &lt; 2 min</Badge>
          </div>
        </Card>
      </div>
    </div>
  );
}
