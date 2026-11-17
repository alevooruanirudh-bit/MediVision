"use client";

import { Contrast, Expand, Minus, Plus, RotateCcw, SunMedium } from "lucide-react";
import * as React from "react";

import { Button, Tip } from "@/components/ui/primitives";
import { assetUrl } from "@/lib/api";
import { cn } from "@/lib/utils";

export interface View {
  zoom: number;
  x: number;
  y: number;
}
export interface Box {
  box: [number, number, number, number];
  label: string;
  color?: string;
  dashed?: boolean;
}
export interface Line {
  points: [number, number][];
  label: string;
  color: string;
}

interface Props {
  src: string;
  width: number;
  height: number;
  heatmap?: string | null;
  heatOpacity?: number;
  mask?: string | null;
  boxes?: Box[];
  lines?: Line[];
  invert?: boolean;
  view?: View;
  onViewChange?: (v: View) => void;
  scanning?: boolean;
  className?: string;
  toolbar?: boolean;
  label?: string;
  onToggleInvert?: () => void;
}

const DEFAULT: View = { zoom: 1, x: 0, y: 0 };

/** Canvas viewer: true window/level on grayscale pixels, invert, zoom/pan (wheel/drag), fullscreen.
 *  Overlays (heatmap, masks, boxes, measurement lines) share the same transform. */
export function ImageViewer({
  src,
  width,
  height,
  heatmap,
  heatOpacity = 0.7,
  mask,
  boxes = [],
  lines = [],
  invert = false,
  view: controlled,
  onViewChange,
  scanning,
  className,
  toolbar = true,
  label,
  onToggleInvert,
}: Props) {
  const wrap = React.useRef<HTMLDivElement>(null);
  const canvas = React.useRef<HTMLCanvasElement>(null);
  const pixels = React.useRef<ImageData | null>(null);
  const [inner, setInner] = React.useState<View>(DEFAULT);
  const view = controlled ?? inner;
  const setView = React.useCallback(
    (v: View) => (onViewChange ? onViewChange(v) : setInner(v)),
    [onViewChange],
  );
  const [wl, setWl] = React.useState({ w: 255, l: 128 });
  const [loaded, setLoaded] = React.useState(false);
  const drag = React.useRef<{ x: number; y: number; mode: "pan" | "wl"; v: View; wl: { w: number; l: number } } | null>(null);

  React.useEffect(() => {
    setLoaded(false);
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const c = canvas.current;
      if (!c) return;
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0);
      try {
        pixels.current = ctx.getImageData(0, 0, c.width, c.height);
      } catch {
        pixels.current = null; // tainted canvas: fall back to plain drawing
      }
      setLoaded(true);
    };
    img.src = assetUrl(src);
  }, [src]);

  React.useEffect(() => {
    const c = canvas.current;
    const base = pixels.current;
    if (!c || !base || !loaded) return;
    const ctx = c.getContext("2d")!;
    const out = new ImageData(new Uint8ClampedArray(base.data), base.width, base.height);
    const lo = wl.l - wl.w / 2;
    const lut = new Uint8ClampedArray(256);
    for (let i = 0; i < 256; i++) {
      let v = ((i - lo) / Math.max(wl.w, 1)) * 255;
      v = Math.max(0, Math.min(255, v));
      lut[i] = invert ? 255 - v : v;
    }
    const d = out.data;
    for (let i = 0; i < d.length; i += 4) {
      d[i] = lut[d[i]];
      d[i + 1] = lut[d[i + 1]];
      d[i + 2] = lut[d[i + 2]];
    }
    ctx.putImageData(out, 0, 0);
  }, [wl, invert, loaded]);

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const z = Math.max(0.5, Math.min(8, view.zoom * factor));
    setView({ ...view, zoom: z });
  };
  const onDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, mode: e.shiftKey || e.button === 2 ? "wl" : "pan", v: view, wl };
  };
  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (d.mode === "pan") setView({ ...d.v, x: d.v.x + dx, y: d.v.y + dy });
    else setWl({ w: Math.max(8, d.wl.w + dx), l: Math.max(0, Math.min(255, d.wl.l + dy)) });
  };
  const onUp = () => (drag.current = null);
  const fullscreen = () => wrap.current?.requestFullscreen?.();
  const reset = () => {
    setView(DEFAULT);
    setWl({ w: 255, l: 128 });
  };

  React.useEffect(() => {
    if (!toolbar) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || e.metaKey || e.ctrlKey) return;
      if (e.key === "+" || e.key === "=") setView({ ...view, zoom: Math.min(8, view.zoom * 1.2) });
      else if (e.key === "-") setView({ ...view, zoom: Math.max(0.5, view.zoom / 1.2) });
      else if (e.key === "0") reset();
      else if (e.key === "f") fullscreen();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, toolbar]);

  const aspect = width && height ? width / height : 1;

  return (
    <div
      ref={wrap}
      className={cn("relative flex h-full min-h-[320px] w-full select-none items-center justify-center overflow-hidden rounded-2xl bg-black", className)}
      onContextMenu={(e) => e.preventDefault()}
      data-testid="image-viewer"
    >
      <div
        className="relative cursor-grab touch-none active:cursor-grabbing"
        style={{
          aspectRatio: String(aspect),
          height: aspect < 1.25 ? "94%" : "auto",
          width: aspect >= 1.25 ? "94%" : "auto",
          maxWidth: "94%",
          maxHeight: "94%",
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
          transition: drag.current ? "none" : "transform 120ms ease-out",
        }}
        onWheel={onWheel}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onDoubleClick={reset}
      >
        <canvas ref={canvas} className="absolute inset-0 h-full w-full" aria-label={label ?? "Medical image"} role="img" />
        {mask && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={assetUrl(mask)} alt="" className="pointer-events-none absolute inset-0 h-full w-full" />
        )}
        {heatmap && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={assetUrl(heatmap)}
            alt="Explainability heatmap"
            className="pointer-events-none absolute inset-0 h-full w-full transition-opacity duration-300"
            style={{ opacity: heatOpacity, mixBlendMode: "screen" }}
            data-testid="heatmap-overlay"
          />
        )}
        <svg viewBox="0 0 1 1" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full overflow-visible">
          {lines.map((l) => (
            <g key={l.label}>
              <line
                x1={l.points[0][0]}
                y1={l.points[0][1]}
                x2={l.points[1][0]}
                y2={l.points[1][1]}
                stroke={l.color}
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
              />
              {l.points.map((p, i) => (
                <line key={i} x1={p[0]} x2={p[0]} y1={p[1] - 0.015} y2={p[1] + 0.015} stroke={l.color} strokeWidth={2} vectorEffect="non-scaling-stroke" />
              ))}
            </g>
          ))}
          {boxes.map((b, i) => (
            <rect
              key={i}
              x={b.box[0]}
              y={b.box[1]}
              width={b.box[2] - b.box[0]}
              height={b.box[3] - b.box[1]}
              fill="none"
              stroke={b.color ?? "#22D3EE"}
              strokeWidth={1.6}
              strokeDasharray={b.dashed ? "5 4" : undefined}
              vectorEffect="non-scaling-stroke"
              rx={0.004}
            />
          ))}
        </svg>
        {boxes.map((b, i) => (
          <span
            key={i}
            className="num pointer-events-none absolute -translate-y-full whitespace-nowrap rounded-md bg-black/70 px-1.5 py-0.5 text-[10px] text-cyan"
            style={{ left: `${b.box[0] * 100}%`, top: `${b.box[1] * 100}%`, transform: `translateY(-110%) scale(${1 / view.zoom})`, transformOrigin: "bottom left" }}
          >
            {b.label}
          </span>
        ))}
        {lines.map((l) => (
          <span
            key={l.label}
            className="num pointer-events-none absolute whitespace-nowrap rounded bg-black/70 px-1 text-[10px]"
            style={{ left: `${l.points[1][0] * 100}%`, top: `${l.points[1][1] * 100}%`, color: l.color, transform: `translate(6px, -50%) scale(${1 / view.zoom})`, transformOrigin: "left" }}
          >
            {l.label}
          </span>
        ))}
        {scanning && <div className="scan-line" />}
      </div>
      {!loaded && <div className="skeleton absolute inset-6" />}
      {toolbar && (
        <div className="absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-0.5 rounded-xl border border-white/10 bg-black/60 p-1 backdrop-blur">
          <Tip content="Zoom out (−)">
            <Button variant="ghost" size="icon" className="text-slate-300 hover:text-white" onClick={() => setView({ ...view, zoom: Math.max(0.5, view.zoom / 1.2) })} aria-label="Zoom out"><Minus /></Button>
          </Tip>
          <span className="num w-12 text-center text-[11px] text-slate-300">{Math.round(view.zoom * 100)}%</span>
          <Tip content="Zoom in (+)">
            <Button variant="ghost" size="icon" className="text-slate-300 hover:text-white" onClick={() => setView({ ...view, zoom: Math.min(8, view.zoom * 1.2) })} aria-label="Zoom in"><Plus /></Button>
          </Tip>
          <div className="mx-1 h-5 w-px bg-white/10" />
          <Tip content="Window/level: shift-drag or right-drag. Click for lung preset.">
            <Button variant="ghost" size="icon" className="text-slate-300 hover:text-white" onClick={() => setWl((w) => (w.w === 255 ? { w: 170, l: 118 } : { w: 255, l: 128 }))} aria-label="Window level preset"><SunMedium /></Button>
          </Tip>
          {onToggleInvert && (
            <Tip content="Invert (i)">
              <Button variant="ghost" size="icon" className={cn("text-slate-300 hover:text-white", invert && "text-cyan")} onClick={onToggleInvert} aria-label="Invert"><Contrast /></Button>
            </Tip>
          )}
          <Tip content="Reset (0 or double-click)">
            <Button variant="ghost" size="icon" className="text-slate-300 hover:text-white" onClick={reset} aria-label="Reset view"><RotateCcw /></Button>
          </Tip>
          <Tip content="Fullscreen (f)">
            <Button variant="ghost" size="icon" className="text-slate-300 hover:text-white" onClick={fullscreen} aria-label="Fullscreen"><Expand /></Button>
          </Tip>
          <span className="num hidden px-2 text-[10px] text-slate-400 sm:inline">W {Math.round(wl.w)} · L {Math.round(wl.l)}</span>
        </div>
      )}
    </div>
  );
}
