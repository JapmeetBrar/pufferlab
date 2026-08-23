import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const runId = process.env.PUFFERLAB_E2E_RUN_ID;
if (runId === undefined) throw new Error("PUFFERLAB_E2E_RUN_ID is required");

async function expectContained(page: Page) {
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
    .toBe(true);
}

async function expectNoSeriousAxeViolations(page: Page) {
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  expect(
    results.violations.filter((violation) =>
      violation.impact === "serious" || violation.impact === "critical"),
  ).toEqual([]);
}

test("provider-free synthetic dashboard journey is actionable, navigable, and accessible", async ({ page }) => {
  const browserPosts: string[] = [];
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const failedRequests: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST") browserPosts.push(request.url());
  });
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("requestfailed", (request) => failedRequests.push(`${request.method()} ${request.url()}`));

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "One query. Two retrieval instincts." })).toBeFocused();
  await expect(page.getByText(/API .* alive/)).toBeVisible();
  await expect(page.getByText("Live-search setup needed")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Configure the server API key" })).toBeVisible();
  await expect(page.getByText("uv run pufferlab doctor --mode live-tiny")).toBeVisible();
  await page.getByLabel("Search query").fill("permission mode");
  await expect(page.getByRole("button", { name: "Compare results" })).toBeDisabled();
  expect(
    await page.evaluate(() => performance.getEntriesByType("resource").map((entry) => entry.name)),
  ).not.toContainEqual(expect.stringContaining("/api/v1/configs"));
  await expectContained(page);
  await expectNoSeriousAxeViolations(page);

  await page.getByRole("link", { name: "Evaluation runs" }).click();
  await expect(page.getByRole("heading", { name: "Evaluation runs", level: 1 })).toBeFocused();
  await expect(page.getByText("Synthetic demo · read-only.")).toBeVisible();
  await expect(page.locator(`a[href="/runs/${runId}"]`)).toBeVisible();
  await expectContained(page);
  await expectNoSeriousAxeViolations(page);

  await page.locator(`a[href="/runs/${runId}"]`).click();
  await expect(page.getByRole("heading", { name: "Evaluation run", level: 1 })).toBeFocused();
  await expect(page.getByRole("heading", { name: "Regressions and gains" })).toBeVisible();
  await expect.poll(() => new URL(page.url()).searchParams.get("order")).toBe("regressions");
  await expect(page.getByRole("link", { name: "Inspect recorded query" }).first()).toBeVisible();
  await Promise.all([
    page.waitForResponse((response) =>
      response.url().includes("/regressions?") &&
      response.url().includes("order=gains") &&
      response.status() === 200),
    page.getByLabel("Order").selectOption("gains"),
  ]);
  await expect.poll(() => new URL(page.url()).searchParams.get("order")).toBe("gains");
  await Promise.all([
    page.waitForResponse((response) =>
      response.url().includes("/regressions?") &&
      response.url().includes("limit=5") &&
      response.status() === 200),
    page.getByLabel("Rows").fill("5"),
  ]);
  await expect.poll(() => new URL(page.url()).searchParams.get("limit")).toBe("5");
  await expectContained(page);
  await expectNoSeriousAxeViolations(page);

  await page.getByRole("link", { name: "Inspect recorded query" }).first().click();
  const queryHeading = page.getByRole("heading", { name: "Query forensics", level: 1 });
  await expect(queryHeading).toBeFocused();
  await expect(page).toHaveURL(/\/playground\?run=/);
  await expect(page.getByText("NOT_OBSERVABLE · original stages")).toBeVisible();
  await expect(page.getByText("Synthetic demo · replay disabled.")).toBeVisible();
  await expectContained(page);
  await expectNoSeriousAxeViolations(page);

  const opener = page.getByRole("button", { name: "Inspect document" }).first();
  await opener.click();
  const dialog = page.getByRole("dialog", { name: "Document evidence" });
  const close = dialog.getByRole("button", { name: "Close document evidence" });
  await expect(dialog).toBeVisible();
  await expect(close).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(close).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(close).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();

  await opener.click();
  await expect(dialog).toBeVisible();
  await page.reload();
  await expect(dialog).toBeVisible();
  await expect(close).toBeFocused();
  await close.click();
  await expect(dialog).toBeHidden();
  await page.goBack();
  await expect(dialog).toBeVisible();
  await page.goForward();
  await expect(dialog).toBeHidden();
  await expectContained(page);

  expect(browserPosts).toEqual([]);
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(failedRequests).toEqual([]);
});
