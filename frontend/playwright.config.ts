import { defineConfig, devices } from "@playwright/test";

const reuse = !process.env.CI;

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: "http://localhost:3000", trace: "retain-on-failure", viewport: { width: 1440, height: 900 } },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: [
    {
      command: "cd .. && .venv/bin/python -m uvicorn app.main:app --port 8000 --app-dir backend",
      url: "http://localhost:8000/api/system/health",
      reuseExistingServer: reuse,
      timeout: 120_000,
      env: { PYTHONPATH: "..:../backend" },
    },
    { command: "npm run dev", url: "http://localhost:3000", reuseExistingServer: reuse, timeout: 120_000 },
  ],
});
