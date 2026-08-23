import { describe, expect, it } from "vitest";

import {
  forensicHref,
  isUuid,
  readPlaygroundForensicIdentity,
  resolveForensicSelection,
} from "./queryState";

const runId = "10000000-0000-4000-8000-000000000001";
const queryId = "20000000-0000-4000-8000-000000000002";
const baselineId = "30000000-0000-4000-8000-000000000003";
const candidates = [
  "40000000-0000-4000-8000-000000000004",
  "50000000-0000-4000-8000-000000000005",
  "60000000-0000-4000-8000-000000000006",
] as const;
const documentId = "70000000-0000-4000-8000-000000000007";

describe("forensic URL state", () => {
  it("recognizes only an exact UUID run/query/config playground context", () => {
    expect(
      readPlaygroundForensicIdentity(
        `?run=${runId}&query=${queryId}&left=${baselineId}&right=${candidates[0]}`,
      ),
    ).toEqual({ runId, queryId });
    expect(readPlaygroundForensicIdentity(`?run=${runId}&query=${queryId}`)).toBeNull();
    expect(
      readPlaygroundForensicIdentity(
        `?run=${runId}&run=${runId}&query=${queryId}&left=${baselineId}&right=${candidates[0]}`,
      ),
    ).toBeNull();
    expect(
      readPlaygroundForensicIdentity(
        `?run=${runId}&query=${queryId}&left=${baselineId}&right=${baselineId}`,
      ),
    ).toBeNull();
    expect(isUuid("not-a-uuid")).toBe(false);
  });

  it("canonicalizes invalid pair/document state without preserving licensed text", () => {
    const selection = resolveForensicSelection(
      `?left=invalid&right=invalid&document=invalid&q=licensed&query_text=licensed`,
      baselineId,
      candidates,
    );
    expect(selection).toEqual({ left: baselineId, right: candidates[0], document: null });
    expect(forensicHref("run-query", { runId, queryId }, selection)).toBe(
      `/runs/${runId}/queries/${queryId}?left=${baselineId}&right=${candidates[0]}`,
    );
  });

  it("serializes the frozen playground link in stable allowlisted order", () => {
    expect(
      forensicHref("playground", { runId, queryId }, {
        left: candidates[1],
        right: candidates[2],
        document: documentId,
      }),
    ).toBe(
      `/playground?run=${runId}&query=${queryId}&left=${candidates[1]}&right=${candidates[2]}&document=${documentId}`,
    );
  });
});
