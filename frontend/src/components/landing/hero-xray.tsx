"use client";

import { motion } from "framer-motion";

/** Hand-drawn SVG chest radiograph silhouette with a sweeping AI heatmap + scan line (no images). */
export function HeroXray() {
  const ribs = Array.from({ length: 8 }, (_, i) => i);
  return (
    <div className="relative aspect-[4/5] w-full max-w-[460px]">
      <div className="absolute inset-0 rounded-[28px] bg-accent-gradient opacity-25 blur-3xl" />
      <div className="glass relative h-full w-full overflow-hidden rounded-[28px] !bg-[#03060d]">
        <svg viewBox="0 0 400 500" className="absolute inset-0 h-full w-full" aria-label="Illustrated chest X-ray">
          <defs>
            <radialGradient id="body" cx="50%" cy="42%" r="60%">
              <stop offset="0" stopColor="#1b2230" />
              <stop offset="1" stopColor="#05070c" />
            </radialGradient>
            <radialGradient id="lung" cx="50%" cy="45%" r="55%">
              <stop offset="0" stopColor="#0b0f17" />
              <stop offset="1" stopColor="#141b27" />
            </radialGradient>
            <radialGradient id="heat" r="50%">
              <stop offset="0" stopColor="#fde047" stopOpacity="0.95" />
              <stop offset="0.35" stopColor="#f97316" stopOpacity="0.7" />
              <stop offset="0.7" stopColor="#8b5cf6" stopOpacity="0.35" />
              <stop offset="1" stopColor="#8b5cf6" stopOpacity="0" />
            </radialGradient>
            <linearGradient id="bone" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="#d7e1ee" stopOpacity="0.85" />
              <stop offset="1" stopColor="#9fb0c6" stopOpacity="0.55" />
            </linearGradient>
            <filter id="soft">
              <feGaussianBlur stdDeviation="1.4" />
            </filter>
            <filter id="glow">
              <feGaussianBlur stdDeviation="6" />
            </filter>
          </defs>
          <rect width="400" height="500" fill="#03060d" />
          {/* torso */}
          <path d="M70 70 Q200 30 330 70 L360 470 Q200 500 40 470 Z" fill="url(#body)" />
          {/* lungs */}
          <path d="M190 105 C130 100 92 160 88 250 C84 330 96 395 120 420 C150 428 178 410 186 380 Z" fill="url(#lung)" />
          <path d="M210 105 C270 100 310 160 314 250 C318 330 306 395 282 420 C252 428 228 412 220 382 Z" fill="url(#lung)" />
          {/* heart shadow */}
          <path d="M198 230 C240 228 272 270 268 320 C264 365 230 392 196 392 C176 380 168 330 172 280 Z" fill="#2a3242" opacity="0.85" filter="url(#soft)" />
          {/* diaphragm */}
          <path d="M92 400 Q150 360 196 395 Q250 365 312 402" fill="none" stroke="#3b4658" strokeWidth="3" opacity="0.7" />
          {/* spine */}
          {Array.from({ length: 14 }, (_, i) => (
            <rect key={i} x="192" y={80 + i * 27} width="16" height="22" rx="4" fill="url(#bone)" opacity="0.35" />
          ))}
          {/* clavicles */}
          <path d="M196 98 Q150 80 92 92" fill="none" stroke="url(#bone)" strokeWidth="7" strokeLinecap="round" opacity="0.75" filter="url(#soft)" />
          <path d="M204 98 Q250 80 308 92" fill="none" stroke="url(#bone)" strokeWidth="7" strokeLinecap="round" opacity="0.75" filter="url(#soft)" />
          {/* ribs */}
          {ribs.map((i) => {
            const y = 118 + i * 34;
            const w = 92 + Math.sin((i / 8) * Math.PI) * 30;
            return (
              <g key={i} opacity={0.62 - i * 0.035} filter="url(#soft)">
                <path d={`M196 ${y} C${196 - w * 0.6} ${y - 18} ${196 - w * 1.18} ${y + 8} ${196 - w * 1.12} ${y + 46}`} fill="none" stroke="url(#bone)" strokeWidth="5" strokeLinecap="round" />
                <path d={`M204 ${y} C${204 + w * 0.6} ${y - 18} ${204 + w * 1.18} ${y + 8} ${204 + w * 1.12} ${y + 46}`} fill="none" stroke="url(#bone)" strokeWidth="5" strokeLinecap="round" />
              </g>
            );
          })}
          {/* sweeping AI heatmap */}
          <motion.circle
            r="78"
            fill="url(#heat)"
            style={{ mixBlendMode: "screen" }}
            initial={{ cx: 135, cy: 330, opacity: 0 }}
            animate={{ cx: [135, 140, 265, 260, 135], cy: [330, 200, 190, 340, 330], opacity: [0, 0.9, 0.75, 0.9, 0] }}
            transition={{ duration: 9, repeat: Infinity, ease: "easeInOut" }}
          />
          {/* detection box + zone label */}
          <motion.g
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 0, 1, 1, 0] }}
            transition={{ duration: 9, repeat: Infinity, times: [0, 0.55, 0.62, 0.92, 1] }}
          >
            <rect x="214" y="290" width="92" height="92" rx="10" fill="none" stroke="#22D3EE" strokeWidth="1.6" strokeDasharray="5 4" />
            <rect x="214" y="268" width="122" height="18" rx="5" fill="#22D3EE" opacity="0.15" />
            <text x="220" y="281" fill="#67e8f9" fontSize="10" fontFamily="JetBrains Mono Variable, monospace">
              Effusion · 0.72 · L lower
            </text>
          </motion.g>
        </svg>
        <div className="scan-line" />
        <div className="absolute inset-x-0 bottom-0 flex items-center justify-between border-t border-white/10 bg-black/40 px-4 py-2.5 text-[11px] text-slate-300 backdrop-blur">
          <span className="num">ensemble · 3 networks · Grad-CAM</span>
          <span className="flex items-center gap-1.5">
            <span className="size-1.5 animate-pulse rounded-full bg-cyan" /> analysing
          </span>
        </div>
      </div>
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.6 }}
        className="glass absolute -right-4 top-10 hidden w-44 rounded-2xl p-3 text-xs sm:block md:-right-10"
      >
        <div className="text-muted">Urgency</div>
        <div className="mt-1 flex items-center gap-2">
          <span className="num text-xl font-semibold text-amber">58</span>
          <span className="rounded-full border border-amber/40 bg-amber/10 px-2 py-0.5 text-[10px] font-semibold text-amber">URGENT</span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/5">
          <motion.div className="h-full bg-gradient-to-r from-amber to-red" initial={{ width: 0 }} animate={{ width: "58%" }} transition={{ delay: 0.9, duration: 1.2 }} />
        </div>
      </motion.div>
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.8 }}
        className="glass absolute -left-4 bottom-20 hidden w-52 rounded-2xl p-3 text-xs sm:block md:-left-12"
      >
        <div className="mb-2 text-muted">Copilot</div>
        <div className="rounded-xl bg-white/5 p-2 text-[11px] leading-relaxed text-slate-200">
          Effusion is likely (p 0.72 ± 0.04), strongest in the left lower zone. CTR 0.49.
        </div>
      </motion.div>
    </div>
  );
}
