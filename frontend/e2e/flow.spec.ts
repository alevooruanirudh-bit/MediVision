import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const SAMPLES = path.resolve(__dirname, "../../sample_data");

test.describe.serial("MediVision end-to-end", () => {
  test("landing → dashboard", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("not a medical device").first()).toBeVisible();
    await expect(page.getByRole("heading", { name: /shows its work/i })).toBeVisible();
    await page.getByRole("link", { name: /Launch dashboard/i }).click();
    await expect(page).toHaveURL(/dashboard/);
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();
  });

  test("upload → findings + heatmap → report → PDF → copilot", async ({ page, request }) => {
    await page.goto("/worklist");
    await page.getByTestId("consent-checkbox").check();
    await page.getByTestId("file-input").setInputFiles(path.join(SAMPLES, "chest_effusion.png"));
    const open = page.getByRole("link", { name: "Open", exact: true }).first();
    await expect(open).toBeVisible({ timeout: 60_000 });
    await open.click();
    await expect(page).toHaveURL(/studies\//);
    await expect(page.getByTestId("finding-row").first()).toBeVisible();
    await expect(page.getByTestId("heatmap-overlay")).toBeVisible();
    await page.getByTestId("finding-row").nth(1).click();
    await expect(page.getByTestId("heatmap-overlay")).toBeVisible();

    // report editor + PDF
    await page.getByRole("link", { name: /Report editor/ }).click();
    await expect(page.getByRole("heading", { name: "Report editor" })).toBeVisible();
    const pdfHref = await page.getByTestId("pdf-link").getAttribute("href");
    const pdf = await request.get(pdfHref!);
    expect(pdf.status()).toBe(200);
    expect((await pdf.body()).subarray(0, 4).toString()).toBe("%PDF");
    await page.getByRole("tab", { name: "Patient-friendly" }).click();
    await expect(page.getByText(/In simple terms/).first()).toBeVisible();

    // copilot chat (streaming)
    await page.getByRole("link", { name: /Back to viewer/ }).click();
    await page.getByTestId("tab-copilot").click();
    await page.getByTestId("chat-input").fill("why effusion?");
    await page.getByTestId("chat-input").press("Enter");
    await expect(page.getByTestId("assistant-message").last()).toContainText(/threshold/i, { timeout: 30_000 });
    await expect(page.getByTestId("action-chip").first()).toBeVisible();
  });

  test("compare → federated", async ({ page, request }) => {
    for (const f of ["chest_followup_prior.png", "chest_followup_current.png"]) {
      const res = await request.post("http://localhost:8000/api/studies", {
        multipart: {
          file: { name: f, mimeType: "image/png", buffer: fs.readFileSync(path.join(SAMPLES, f)) },
          patient_ref: "E2E-PAIR",
          modality_hint: "chest",
          consent_basis: "public_research_dataset",
        },
      });
      expect(res.ok()).toBeTruthy();
    }
    await page.goto("/compare");
    await expect(page.getByText(/Compared with the prior study/)).toBeVisible({ timeout: 60_000 });
    await page.goto("/federated");
    await page.getByTestId("run-federated").click();
    await expect(page.getByText(/Round \d+\/10/)).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId("run-federated")).toHaveText(/Run federated training/, { timeout: 240_000 });
  });
});
