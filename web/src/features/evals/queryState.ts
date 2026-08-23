const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export interface ForensicIdentity {
  runId: string;
  queryId: string;
}

export interface ForensicSelection {
  left: string;
  right: string;
  document: string | null;
}

export type ForensicRouteKind = "playground" | "run-query";

export function isUuid(value: string): boolean {
  return uuidPattern.test(value);
}

function exactParameter(parameters: URLSearchParams, name: string): string | null {
  const values = parameters.getAll(name);
  return values.length === 1 ? values[0] ?? null : null;
}

export function readPlaygroundForensicIdentity(search: string): ForensicIdentity | null {
  const parameters = new URLSearchParams(search);
  const runId = exactParameter(parameters, "run");
  const queryId = exactParameter(parameters, "query");
  const left = exactParameter(parameters, "left");
  const right = exactParameter(parameters, "right");
  if (
    runId === null ||
    queryId === null ||
    left === null ||
    right === null ||
    !isUuid(runId) ||
    !isUuid(queryId) ||
    !isUuid(left) ||
    !isUuid(right) ||
    left === right
  ) {
    return null;
  }
  return { runId, queryId };
}

export function resolveForensicSelection(
  search: string,
  baselineConfigId: string,
  candidateConfigIds: readonly string[],
): ForensicSelection {
  const parameters = new URLSearchParams(search);
  const configIds = [baselineConfigId, ...candidateConfigIds];
  const requestedLeft = exactParameter(parameters, "left");
  const requestedRight = exactParameter(parameters, "right");
  const left = requestedLeft !== null && configIds.includes(requestedLeft)
    ? requestedLeft
    : baselineConfigId;
  const fallbackRight = candidateConfigIds.find((configId) => configId !== left)
    ?? configIds.find((configId) => configId !== left)
    ?? "";
  const right = requestedRight !== null && requestedRight !== left && configIds.includes(requestedRight)
    ? requestedRight
    : fallbackRight;
  const requestedDocument = exactParameter(parameters, "document");
  return {
    left,
    right,
    document: requestedDocument !== null && isUuid(requestedDocument) ? requestedDocument : null,
  };
}

export function forensicHref(
  routeKind: ForensicRouteKind,
  identity: ForensicIdentity,
  selection: ForensicSelection,
): string {
  const parameters = new URLSearchParams();
  if (routeKind === "playground") {
    parameters.set("run", identity.runId);
    parameters.set("query", identity.queryId);
  }
  parameters.set("left", selection.left);
  parameters.set("right", selection.right);
  if (selection.document !== null) parameters.set("document", selection.document);
  const pathname = routeKind === "playground"
    ? "/playground"
    : `/runs/${encodeURIComponent(identity.runId)}/queries/${encodeURIComponent(identity.queryId)}`;
  return `${pathname}?${parameters.toString()}`;
}
