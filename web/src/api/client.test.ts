import { describe, expect, it } from "vitest";

import { readApiError } from "./client";

describe("readApiError", () => {
  it("preserves the generated configuration_required code", async () => {
    const response = new Response(
      JSON.stringify({
        code: "configuration_required",
        message: "local search configuration is required",
        retryable: false,
        trace_id: "00000000-0000-0000-0000-000000000001",
        details: { operation: "search_configuration" },
      }),
      {
        status: 503,
        headers: { "Content-Type": "application/json" },
      },
    );

    const error = await readApiError(response);

    expect(error.status).toBe(503);
    expect(error.detail.code).toBe("configuration_required");
    expect(error.detail.details).toEqual({ operation: "search_configuration" });
  });
});
