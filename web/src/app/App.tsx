import { useQuery } from "@tanstack/react-query";

import { getHealth } from "../api/client";
import { RunDetailPage } from "../features/evals/RunDetailPage";
import { RunListPage } from "../features/evals/RunListPage";
import { QueryDetailPage } from "../features/evals/QueryDetailPage";
import {
  isUuid,
  readPlaygroundForensicIdentity,
} from "../features/evals/queryState";
import { Playground } from "../features/playground/Playground";
import "./app.css";
import { AppLink, RouteHeading } from "./router";
import { useAppLocation } from "./routing";

function decodePathId(value: string): string | null {
  try {
    return decodeURIComponent(value);
  } catch {
    return null;
  }
}

export function App() {
  const location = useAppLocation();
  const health = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => getHealth(signal),
    retry: false,
  });
  const queryMatch = /^\/runs\/([^/]+)\/queries\/([^/]+)$/.exec(location.pathname);
  const queryRunId = queryMatch?.[1] === undefined ? null : decodePathId(queryMatch[1]);
  const queryId = queryMatch?.[2] === undefined ? null : decodePathId(queryMatch[2]);
  const nestedQueryIdentity =
    queryRunId !== null && queryId !== null && isUuid(queryRunId) && isUuid(queryId)
      ? { runId: queryRunId, queryId }
      : null;
  const playgroundForensicIdentity = location.pathname === "/playground"
    ? readPlaygroundForensicIdentity(location.search)
    : null;
  const runMatch = /^\/runs\/([^/]+)$/.exec(location.pathname);
  const runId = runMatch?.[1] === undefined ? null : decodePathId(runMatch[1]);
  const playgroundRoute =
    (location.pathname === "/" || location.pathname === "/playground") &&
    playgroundForensicIdentity === null;
  const runsRoute =
    location.pathname === "/runs" ||
    runId !== null ||
    nestedQueryIdentity !== null ||
    playgroundForensicIdentity !== null;

  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <header className="topbar">
        <AppLink className="wordmark" href="/" aria-label="PufferLab home">
          <span className="puffer-mark" aria-hidden="true">P</span>
          PufferLab
        </AppLink>
        <nav className="primary-nav" aria-label="Primary navigation">
          <AppLink href="/" aria-current={playgroundRoute ? "page" : undefined}>
            Playground
          </AppLink>
          <AppLink href="/runs" aria-current={runsRoute ? "page" : undefined}>
            Evaluation runs
          </AppLink>
        </nav>
        <div className="service-status" aria-live="polite">
          <span className={`status-dot ${health.isSuccess ? "is-ready" : ""}`} aria-hidden="true" />
          {health.isPending && "Connecting to API"}
          {health.isSuccess && `API ${health.data.version} ready`}
          {health.isError && "API unavailable"}
        </div>
      </header>

      <main id="main-content">
        {playgroundRoute && <Playground />}
        {location.pathname === "/runs" && <RunListPage routeKey={location.pathname} />}
        {runId !== null && (
          <RunDetailPage runId={runId} routeKey={location.pathname} search={location.search} />
        )}
        {nestedQueryIdentity !== null && (
          <QueryDetailPage
            runId={nestedQueryIdentity.runId}
            queryId={nestedQueryIdentity.queryId}
            routeKey={`${location.pathname}:${nestedQueryIdentity.queryId}`}
            routeKind="run-query"
            search={location.search}
          />
        )}
        {playgroundForensicIdentity !== null && (
          <QueryDetailPage
            runId={playgroundForensicIdentity.runId}
            queryId={playgroundForensicIdentity.queryId}
            routeKey={`/playground:${playgroundForensicIdentity.runId}:${playgroundForensicIdentity.queryId}`}
            routeKind="playground"
            search={location.search}
          />
        )}
        {!playgroundRoute &&
          location.pathname !== "/runs" &&
          runId === null &&
          nestedQueryIdentity === null &&
          playgroundForensicIdentity === null && (
          <section className="route-state route-not-found">
            <p className="eyebrow">404</p>
            <RouteHeading routeKey={location.pathname}>Page not found</RouteHeading>
            <p>The requested PufferLab page does not exist.</p>
            <AppLink className="text-link" href="/runs">
              View evaluation runs
            </AppLink>
          </section>
          )}
      </main>
    </>
  );
}
