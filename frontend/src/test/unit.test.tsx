import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DISCLAIMER, DisclaimerBanner, UrgencyPill } from "@/components/common";
import { TooltipProvider } from "@/components/ui/primitives";
import { ProbabilityBar } from "@/components/viewer/panels";
import { assetUrl, streamSSE } from "@/lib/api";
import type { Finding } from "@/lib/types";
import { cn, fmt, ms, pct, titleCase } from "@/lib/utils";

describe("utils", () => {
  it("formats numbers", () => {
    expect(pct(0.1234)).toBe("12%");
    expect(pct(0.1234, 1)).toBe("12.3%");
    expect(pct(null)).toBe("—");
    expect(fmt(0.4567)).toBe("0.46");
    expect(ms(1234)).toBe("1.23 s");
    expect(ms(80)).toBe("80 ms");
    expect(titleCase("pleural_thickening")).toBe("Pleural Thickening");
    expect(cn("px-2", "px-4")).toBe("px-4");
  });
  it("builds asset URLs", () => {
    expect(assetUrl("/files/a.png")).toMatch(/\/files\/a\.png$/);
    expect(assetUrl("data:image/png;base64,xx")).toBe("data:image/png;base64,xx");
    expect(assetUrl(null)).toBe("");
  });
});

describe("components", () => {
  it("shows the disclaimer", () => {
    render(<DisclaimerBanner />);
    expect(screen.getByText(DISCLAIMER)).toBeInTheDocument();
  });
  it("renders urgency pills with level and score", () => {
    render(<UrgencyPill level="CRITICAL" score={81.2} />);
    expect(screen.getByText("CRITICAL")).toBeInTheDocument();
    expect(screen.getByText("81")).toBeInTheDocument();
  });
  it("renders a probability bar", () => {
    const f: Finding = {
      label: "Effusion", key: "Effusion", probability: 0.6, uncertainty: 0.05, threshold: 0.2,
      positive: true, certainty: "Likely", ratio_to_threshold: 3, kind: "classification",
    };
    const { container } = render(
      <TooltipProvider>
        <ProbabilityBar f={f} />
      </TooltipProvider>,
    );
    expect(container.firstChild).toBeTruthy();
  });
});

describe("SSE parser", () => {
  afterEach(() => vi.restoreAllMocks());
  it("parses events across chunk boundaries", async () => {
    const chunks = ['event: token\ndata: "Hel', 'lo "\n\nevent: token\ndata: "world"\n\n', 'event: done\ndata: {"intent":"x"}\n\n'];
    const stream = new ReadableStream({
      start(c) {
        chunks.forEach((s) => c.enqueue(new TextEncoder().encode(s)));
        c.close();
      },
    });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(stream, { status: 200 }));
    const got: [string, unknown][] = [];
    await streamSSE("/x", {}, (e, d) => got.push([e, d]));
    expect(got).toEqual([["token", "Hello "], ["token", "world"], ["done", { intent: "x" }]]);
  });
});
