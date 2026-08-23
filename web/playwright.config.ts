import { defineConfig } from "@playwright/test";

const baseURL = process.env.PUFFERLAB_E2E_BASE_URL;
if (baseURL === undefined) {
  throw new Error("PUFFERLAB_E2E_BASE_URL is required; run pnpm test:e2e through the checked-in harness");
}

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.spec.ts",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: process.env.CI === "true",
  reporter: "list",
  outputDir: "test-results",
  use: {
    baseURL,
    browserName: "chromium",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium-desktop",
      use: { viewport: { width: 1280, height: 800 } },
    },
    {
      name: "chromium-mobile-390",
      use: {
        viewport: { width: 390, height: 844 },
        hasTouch: true,
        isMobile: true,
      },
    },
  ],
});
