import { useQuery } from "@tanstack/react-query";

import { getHealth } from "../api/client";
import { Playground } from "../features/playground/Playground";
import "./app.css";

export function App() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => getHealth(signal),
    retry: false,
  });

  return (
    <main>
      <header className="topbar">
        <a className="wordmark" href="/" aria-label="PufferLab home">
          <span className="puffer-mark" aria-hidden="true">P</span>
          PufferLab
        </a>
        <div className="service-status" aria-live="polite">
          <span className={`status-dot ${health.isSuccess ? "is-ready" : ""}`} />
          {health.isPending && "Connecting to API"}
          {health.isSuccess && `API ${health.data.version} ready`}
          {health.isError && "API unavailable"}
        </div>
      </header>

      <Playground />
    </main>
  );
}
