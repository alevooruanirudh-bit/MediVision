import { expect, test } from "@playwright/test";
import path from "node:path";

const OUT = path.resolve(__dirname, "../../docs/screenshots");

async function firstStudy(request: import("@playwright/test").APIRequestContext, pred: (s: Record<string, unknown>) => boolean) {
  const list = (await (await request.get("http://localhost:8000/api/studies")).json()) as Record<string, unknown>[];
  return list.find(pred)?.id as string | undefined;
}

test("screenshots of every page", async ({ page, request }) => {
  test.setTimeout(300_000);
  await request.post("http://localhost:8000/api/demo/load", { timeout: 240_000 });
  const chest = await firstStudy(request, (s) => s.pipeline === "chest" && (s.n_positive as number) > 0);
  const shots: [string, string, (() => Promise<void>)?][] = [
    ["01-landing", "/"],
    ["02-dashboard", "/dashboard"],
    ["03-worklist", "/worklist"],
    ["04-study-viewer", `/studies/${chest}`, async () => {
      await expect(page.getByTestId("heatmap-overlay")).toBeVisible();
    }],
    ["04b-study-copilot", `/studies/${chest}`, async () => {
      await page.getByTestId("tab-copilot").click();
      await page.getByTestId("chat-input").fill("why is this flagged and where?");
      await page.getByTestId("chat-input").press("Enter");
      await expect(page.getByTestId("action-chip").first()).toBeVisible({ timeout: 30_000 });
    }],
    ["05-report-editor", `/studies/${chest}/report`],
    ["06-compare", "/compare", async () => {
      await expect(page.getByText(/Compared with the prior study/)).toBeVisible({ timeout: 60_000 });
    }],
    ["07-enhancement-lab", "/enhance", async () => {
      await expect(page.getByTestId("before-after")).toBeVisible({ timeout: 30_000 });
    }],
    ["08-federated", "/federated"],
    ["09-lab", "/lab"],
    ["10-model-card", "/model-card"],
    ["11-audit", "/audit"],
    ["12-settings", "/settings"],
  ];
  for (const [name, url, ready] of shots) {
    await page.goto(url, { waitUntil: "networkidle" });
    if (ready) await ready();
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${OUT}/${name}.png` });
  }
  await page.setViewportSize({ width: 390, height: 844 });
  for (const [name, url] of [["13-mobile-worklist", "/worklist"], ["14-mobile-viewer", `/studies/${chest}`]]) {
    await page.goto(url, { waitUntil: "networkidle" });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${OUT}/${name}.png` });
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/dashboard");
  await page.evaluate(() => localStorage.setItem("theme", "light"));
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/15-light-dashboard.png` });
  await page.evaluate(() => localStorage.setItem("theme", "dark"));
});
