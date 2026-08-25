import { describe, expect, it } from "vitest";

import { relevanceLabel } from "./relevance";

describe("relevanceLabel", () => {
  it("translates graded relevance without flattening stronger positive grades", () => {
    expect(relevanceLabel(0)).toBe("Not relevant");
    expect(relevanceLabel(1)).toBe("Relevant");
    expect(relevanceLabel(2)).toBe("Highly relevant");
    expect(relevanceLabel(4)).toBe("Highly relevant");
  });
});
