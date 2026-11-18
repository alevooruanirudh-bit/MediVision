"use client";

import { Pause, Play, RotateCw } from "lucide-react";
import * as React from "react";

import { Button, Slider } from "@/components/ui/primitives";
import { assetUrl } from "@/lib/api";

/** 3-D view for CT volumes (C4): rotating maximum-intensity projection with the segmented
 * nodule tinted, plus axial / coronal / sagittal multiplanar slices through the centre. */
export function VolumeViewer({ mpr, frames }: { mpr: Record<string, string>; frames: string[] }) {
  const [i, setI] = React.useState(0);
  const [playing, setPlaying] = React.useState(true);
  React.useEffect(() => {
    if (!playing || !frames.length) return;
    const t = setInterval(() => setI((x) => (x + 1) % frames.length), 140);
    return () => clearInterval(t);
  }, [playing, frames.length]);
  return (
    <div className="grid h-full gap-2 md:grid-cols-[minmax(0,1fr)_200px]" data-testid="volume-viewer">
      <div className="relative flex min-h-[300px] flex-col items-center justify-center overflow-hidden rounded-xl bg-black">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        {frames[i] && <img src={assetUrl(frames[i])} alt={`3-D maximum-intensity projection, ${Math.round((360 * i) / frames.length)}°`} className="h-full max-h-[60vh] w-auto [image-rendering:pixelated]" />}
        <div className="absolute inset-x-3 bottom-3 flex items-center gap-3 rounded-xl bg-black/60 px-3 py-2 text-[11px] text-white/80 backdrop-blur">
          <Button size="icon" variant="ghost" className="size-7 text-white" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause rotation" : "Play rotation"}>
            {playing ? <Pause /> : <Play />}
          </Button>
          <RotateCw className="size-3.5" />
          <Slider value={[i]} max={Math.max(frames.length - 1, 0)} step={1} onValueChange={([x]) => { setPlaying(false); setI(x); }} aria-label="Rotation angle" />
          <span className="num w-10 text-right">{Math.round((360 * i) / Math.max(frames.length, 1))}°</span>
        </div>
        <span className="absolute left-3 top-3 rounded bg-black/60 px-2 py-0.5 text-[10px] text-white/70">3-D MIP · violet = segmented nodule</span>
      </div>
      <div className="grid grid-cols-3 gap-2 md:grid-cols-1">
        {Object.entries(mpr).map(([plane, url]) => (
          <figure key={plane} className="overflow-hidden rounded-xl border border-border bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={assetUrl(url)} alt={`${plane} slice`} className="aspect-square w-full [image-rendering:pixelated]" />
            <figcaption className="px-2 py-1 text-[10px] capitalize text-muted">{plane}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
