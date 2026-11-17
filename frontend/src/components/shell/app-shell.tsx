"use client";

import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BookOpenText,
  FlaskConical,
  GitCompareArrows,
  Keyboard,
  LayoutDashboard,
  ListChecks,
  Menu,
  Moon,
  Network,
  ScrollText,
  Settings,
  Sparkles,
  Sun,
  Wand2,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import * as React from "react";

import { DisclaimerBanner, ModelBadge } from "@/components/common";
import { Logo } from "@/components/shell/logo";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Badge, Button, Kbd, Tip } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

export const NAV = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, key: "d" },
  { href: "/worklist", label: "Worklist", icon: ListChecks, key: "w" },
  { href: "/compare", label: "Compare", icon: GitCompareArrows, key: "c" },
  { href: "/enhance", label: "Enhancement Lab", icon: Wand2, key: "e" },
  { href: "/federated", label: "Federated", icon: Network, key: "f" },
  { href: "/lab", label: "Model Performance", icon: FlaskConical, key: "l" },
  { href: "/model-card", label: "Model Card", icon: BookOpenText, key: "m" },
  { href: "/audit", label: "Audit Log", icon: ScrollText, key: "a" },
  { href: "/settings", label: "Settings", icon: Settings, key: "s" },
];

function useShortcuts(onHelp: () => void) {
  const router = useRouter();
  React.useEffect(() => {
    let g = false;
    let t: ReturnType<typeof setTimeout>;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || (e.target as HTMLElement)?.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "?") return onHelp();
      if (e.key === "g") {
        g = true;
        clearTimeout(t);
        t = setTimeout(() => (g = false), 900);
        return;
      }
      if (g) {
        const item = NAV.find((n) => n.key === e.key);
        if (item) router.push(item.href);
        g = false;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [router, onHelp]);
}

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);
  const dark = !mounted || resolvedTheme === "dark";
  return (
    <Tip content={dark ? "Light mode" : "Dark mode"}>
      <Button variant="ghost" size="icon" aria-label="Toggle theme" onClick={() => setTheme(dark ? "light" : "dark")}>
        {dark ? <Sun /> : <Moon />}
      </Button>
    </Tip>
  );
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const path = usePathname();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000 });
  return (
    <div className="flex h-full flex-col">
      <Link href="/" className="flex items-center px-5 py-5" onClick={onNavigate}>
        <Logo />
      </Link>
      <nav className="flex-1 space-y-0.5 px-3" aria-label="Main">
        {NAV.map((n) => {
          const active = path === n.href || path.startsWith(n.href + "/") || (n.href === "/worklist" && path.startsWith("/studies"));
          return (
            <Link
              key={n.href}
              href={n.href}
              onClick={onNavigate}
              className={cn(
                "group relative flex items-center gap-3 rounded-xl px-3 py-2 text-[13px] font-medium transition-colors",
                active ? "text-fg" : "text-muted hover:bg-surface-2 hover:text-fg",
              )}
            >
              {active && (
                <motion.span
                  layoutId="nav-active"
                  className="absolute inset-0 rounded-xl border border-border bg-surface-2"
                  transition={{ type: "spring", stiffness: 500, damping: 40 }}
                />
              )}
              <n.icon className={cn("relative size-4", active ? "text-cyan" : "text-subtle group-hover:text-muted")} />
              <span className="relative">{n.label}</span>
              <span className="relative ml-auto hidden text-[10px] text-subtle opacity-0 transition group-hover:opacity-100 lg:inline">
                g {n.key}
              </span>
            </Link>
          );
        })}
      </nav>
      <div className="m-3 rounded-2xl border border-border bg-surface-2 p-3.5">
        <div className="flex items-center gap-2 text-xs font-medium text-fg">
          <span className={cn("size-2 rounded-full", health.data ? "bg-teal shadow-[0_0_8px_var(--teal)]" : "bg-red")} />
          {health.data ? "Engine online" : health.isLoading ? "Connecting…" : "Backend offline"}
        </div>
        {health.data && (
          <div className="mt-2 space-y-1.5 text-[11px] text-muted">
            <div className="flex justify-between">
              <span>Models loaded</span>
              <span className="num text-fg">{health.data.n_models_loaded}</span>
            </div>
            <div className="flex justify-between">
              <span>Runtime</span>
              <span className="num text-fg">{health.data.inference_backend.toUpperCase()} · CPU</span>
            </div>
            <div className="flex justify-between">
              <span>External AI</span>
              <span className="num text-teal">none</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [help, setHelp] = React.useState(false);
  const [mobile, setMobile] = React.useState(false);
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000 });
  const onHelp = React.useCallback(() => setHelp(true), []);
  useShortcuts(onHelp);
  const path = usePathname();
  return (
    <div className="relative min-h-dvh">
      <div className="app-aurora" />
      <div className="sticky top-0 z-40">
        <DisclaimerBanner />
      </div>
      <aside className="fixed bottom-0 left-0 top-[29px] z-30 hidden w-64 border-r border-border bg-bg/70 backdrop-blur-xl lg:block">
        <Sidebar />
      </aside>
      <AnimatePresence>
        {mobile && (
          <motion.div className="fixed inset-0 z-50 lg:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="absolute inset-0 bg-black/60" onClick={() => setMobile(false)} />
            <motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: "spring", stiffness: 400, damping: 40 }}
              className="absolute bottom-0 left-0 top-0 w-64 border-r border-border bg-elev"
            >
              <button className="absolute right-3 top-5 text-muted" onClick={() => setMobile(false)} aria-label="Close menu">
                <X className="size-5" />
              </button>
              <Sidebar onNavigate={() => setMobile(false)} />
            </motion.aside>
          </motion.div>
        )}
      </AnimatePresence>
      <div className="relative z-10 lg:pl-64">
        <header className="sticky top-[29px] z-20 flex h-14 items-center gap-3 border-b border-border bg-bg/70 px-4 backdrop-blur-xl md:px-6">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobile(true)} aria-label="Open menu">
            <Menu />
          </Button>
          <div className="lg:hidden">
            <Logo withText={false} />
          </div>
          <div className="hidden items-center gap-2 text-xs text-muted md:flex">
            <Activity className="size-3.5 text-cyan" />
            <span>Fully local inference · self-trained models</span>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <ModelBadge profile={health.data?.active_profile} />
            <Badge tone="teal" className="hidden sm:inline-flex">
              <Sparkles className="size-3" /> Copilot: built-in engine
            </Badge>
            <Tip content="Keyboard shortcuts (?)">
              <Button variant="ghost" size="icon" aria-label="Keyboard shortcuts" onClick={() => setHelp(true)}>
                <Keyboard />
              </Button>
            </Tip>
            <ThemeToggle />
          </div>
        </header>
        <AnimatePresence mode="wait">
          <motion.main
            key={path}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            className="mx-auto w-full max-w-[1600px] px-4 py-6 md:px-6 md:py-8"
          >
            {children}
          </motion.main>
        </AnimatePresence>
      </div>
      <Dialog open={help} onOpenChange={setHelp}>
        <DialogContent title="Keyboard shortcuts" description="Press g then a letter to navigate.">
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
            {NAV.map((n) => (
              <div key={n.href} className="flex items-center justify-between text-muted">
                {n.label}
                <span className="flex gap-1">
                  <Kbd>g</Kbd>
                  <Kbd>{n.key}</Kbd>
                </span>
              </div>
            ))}
            {[
              ["Zoom in / out", "+ / −"],
              ["Reset view", "0"],
              ["Invert", "i"],
              ["Heatmap overlay", "h"],
              ["Segmentation mask", "m"],
              ["Boxes / lines", "b"],
              ["Fullscreen viewer", "f"],
              ["Viewer tabs", "1–4"],
              ["This help", "?"],
            ].map(([a, b]) => (
              <div key={a} className="flex items-center justify-between text-muted">
                {a}
                <Kbd>{b}</Kbd>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
