"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Cpu, KeyRound, Loader2, LogOut, Moon, PlugZap, RefreshCcw, Sun, UserRound } from "lucide-react";
import { useTheme } from "next-themes";
import * as React from "react";
import { toast } from "sonner";

import { PageHeader } from "@/components/common";
import { Badge, Button, Card, CardHeader, Input, Label, NativeSelect, Skeleton, Slider, Switch } from "@/components/ui/primitives";
import { API_URL, api } from "@/lib/api";
import { type Role, useSettings } from "@/lib/store";

const PERMS: Record<string, string> = {
  "study.write": "upload / analyse",
  "study.delete": "delete studies",
  "finding.review": "review findings",
  "report.edit": "edit reports",
  "report.sign": "sign reports",
  "alert.ack": "acknowledge alerts",
  "users.manage": "manage users",
};

function AccessCard() {
  const qc = useQueryClient();
  const s = useSettings();
  const me = useQuery({ queryKey: ["me", s.token, s.demoRole], queryFn: api.me });
  const [u, setU] = React.useState("dr.lee");
  const [pw, setPw] = React.useState("");
  const refresh = () => qc.invalidateQueries({ queryKey: ["me"] });
  const login = async () => {
    try {
      const r = await api.login(u, pw);
      s.set({ token: r.token, reviewer: r.user.name });
      toast.success(`Signed in as ${r.user.name} (${r.user.role})`);
      refresh();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };
  const m = me.data;
  return (
    <Card className="md:col-span-2" data-testid="access-card">
      <CardHeader
        icon={<KeyRound />}
        title="Access control"
        subtitle={
          m?.auth_enabled
            ? "AUTH_ENABLED=true — sign in with a local account; roles are enforced on every write."
            : "Demo mode (AUTH_ENABLED=false): choose the acting role to try the permission checks. Enable auth for any shared deployment."
        }
      />
      <div className="grid gap-4 px-5 pb-5 md:grid-cols-2">
        <div className="space-y-2 text-sm">
          <div className="text-muted">
            Acting as <span className="text-fg">{m?.name}</span> <Badge>{m?.role}</Badge>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(PERMS).map(([p, l]) => (
              <Badge key={p} tone={m?.permissions.includes(p) ? "teal" : "neutral"} className={m?.permissions.includes(p) ? "" : "line-through opacity-60"}>{l}</Badge>
            ))}
          </div>
        </div>
        {m?.auth_enabled ? (
          s.token ? (
            <Button
              variant="secondary"
              onClick={async () => {
                await api.logout().catch(() => undefined);
                s.set({ token: null });
                refresh();
              }}
            >
              <LogOut /> Sign out
            </Button>
          ) : (
            <form className="flex flex-wrap gap-2" onSubmit={(e) => (e.preventDefault(), login())}>
              <Input value={u} onChange={(e) => setU(e.target.value)} placeholder="username" aria-label="Username" className="w-36" />
              <Input value={pw} onChange={(e) => setPw(e.target.value)} type="password" placeholder="password" aria-label="Password" className="w-40" />
              <Button type="submit" variant="primary">Sign in</Button>
            </form>
          )
        ) : (
          <div>
            <Label>Demo role</Label>
            <NativeSelect
              value={s.demoRole}
              onChange={(e) => {
                s.set({ demoRole: e.target.value as Role });
                refresh();
              }}
              className="mt-1 w-full"
              aria-label="Demo role"
            >
              <option value="radiologist">Radiologist (sign, review, upload)</option>
              <option value="resident">Resident (review, draft — cannot sign)</option>
              <option value="viewer">Viewer (read-only)</option>
              <option value="admin">Admin</option>
            </NativeSelect>
          </div>
        )}
      </div>
    </Card>
  );
}

export default function SettingsPage() {
  const qc = useQueryClient();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const card = useQuery({ queryKey: ["model-card"], queryFn: api.modelCard });
  const s = useSettings();
  const { resolvedTheme, setTheme } = useTheme();
  const [reloading, setReloading] = React.useState(false);
  const h = health.data;

  return (
    <div className="max-w-5xl">
      <PageHeader eyebrow="Configuration" title="Settings" subtitle="Client preferences are stored in this browser. Model and server settings come from environment variables (.env)." />
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader icon={resolvedTheme === "dark" ? <Moon /> : <Sun />} title="Appearance" />
          <div className="flex items-center justify-between px-5 pb-5 text-sm text-muted">
            Dark mode
            <Switch checked={resolvedTheme === "dark"} onCheckedChange={(d) => setTheme(d ? "dark" : "light")} aria-label="Dark mode" />
          </div>
        </Card>
        <Card>
          <CardHeader icon={<UserRound />} title="Reviewer" subtitle="Default name for sign-off and reviews" />
          <div className="px-5 pb-5">
            <Input value={s.reviewer} onChange={(e) => s.set({ reviewer: e.target.value })} aria-label="Reviewer name" />
          </div>
        </Card>
        <Card>
          <CardHeader title="Copilot streaming speed" subtitle="Delay between streamed words" />
          <div className="flex items-center gap-4 px-5 pb-5">
            <Slider value={[s.chatDelay]} min={0} max={80} step={2} onValueChange={([v]) => s.set({ chatDelay: v })} aria-label="Streaming delay" />
            <span className="num w-14 text-right text-sm text-fg">{s.chatDelay} ms</span>
          </div>
        </Card>
        <AccessCard />
        <Card>
          <CardHeader title="Autonomous triage (well-defined tasks)" subtitle="AUTONOMOUS_TRIAGE — AI pre-clears confident normals and raises critical alerts; a human always confirms" />
          <div className="flex items-center justify-between px-5 pb-5 text-sm">
            <span className="text-muted">Status</span>
            <Badge tone={h?.settings?.autonomous_triage ? "teal" : "neutral"}>{h?.settings?.autonomous_triage ? "on" : "off"}</Badge>
          </div>
        </Card>
        <Card>
          <CardHeader icon={<PlugZap />} title="Local LLM plug-in" subtitle="Optional Ollama adapter for Copilot wording only — localhost, never cloud" />
          <div className="flex items-center justify-between px-5 pb-5 text-sm">
            <span className="text-muted">Status (COPILOT_LOCAL_LLM)</span>
            <Badge tone={h?.local_llm.enabled ? "violet" : "neutral"}>{h?.local_llm.enabled ? `on · ${h.local_llm.model}` : "off (default)"}</Badge>
          </div>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader
          icon={<Cpu />}
          title="Active model weights"
          subtitle={`Backend ${API_URL} · runtime ${h?.inference_backend ?? "—"} · full weights are preferred automatically when present`}
          action={
            <Button
              variant="secondary"
              size="sm"
              onClick={async () => {
                setReloading(true);
                await api.reload();
                await qc.invalidateQueries();
                setReloading(false);
                toast.success("Model weights reloaded from artifacts/");
              }}
            >
              {reloading ? <Loader2 className="animate-spin" /> : <RefreshCcw />} Reload
            </Button>
          }
        />
        {!h ? (
          <Skeleton className="m-5 h-40" />
        ) : (
          <div className="grid gap-2 px-5 pb-5 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(h.models).map(([k, m]) => (
              <div key={k} className="flex items-center justify-between rounded-lg border border-border bg-surface-2 px-3 py-2 text-xs">
                <span className="num text-fg">{k}</span>
                {m.loaded ? (
                  <span className="flex gap-1.5">
                    <Badge tone={m.profile === "full" ? "violet" : "cyan"}>{m.profile}</Badge>
                    <Badge>{m.backend}</Badge>
                  </span>
                ) : (
                  <Badge>not trained</Badge>
                )}
              </div>
            ))}
          </div>
        )}
        {h && (
          <div className="num border-t border-border px-5 py-3 text-[11px] text-subtle">
            python {h.versions.python} · torch {h.versions.torch} · onnxruntime {h.versions.onnxruntime} · MC-dropout samples {String(h.settings.mc_dropout_samples)} · max upload {String(h.settings.max_upload_mb)} MB · external AI APIs: {h.external_ai_apis}
          </div>
        )}
      </Card>

      <Card className="mt-4">
        <CardHeader title="Decision thresholds" subtitle="Per-label operating points are learned on validation data (Youden's J). Change them by re-running evaluation, not here." />
        <div className="grid gap-2 px-5 pb-5 text-xs sm:grid-cols-3 lg:grid-cols-5">
          {Object.entries(card.data?.thresholds ?? {}).map(([k, v]) => (
            <div key={k} className="flex justify-between rounded-lg border border-border bg-surface-2 px-2.5 py-1.5">
              <Label className="text-muted">{k.replace("_", " ")}</Label>
              <span className="num text-fg">{v.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
