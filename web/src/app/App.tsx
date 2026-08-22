import { useQuery } from "@tanstack/react-query";

import { getHealth } from "../api/client";
import "./app.css";

const milestones = [
  ["01", "Compare", "Inspect lexical and vector results side by side."],
  ["02", "Evaluate", "Run judged queries against immutable configurations."],
  ["03", "Debug", "Open real regressions with observable stage evidence."],
] as const;

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

      <section className="hero">
        <p className="eyebrow">Search quality, made inspectable</p>
        <h1>Find the query that got worse.</h1>
        <p className="lede">
          Compare retrieval configurations, measure relevance, and trace regressions through
          observable candidate and ranking stages—without invented explanations.
        </p>
        <div className="hero-actions">
          <button type="button" disabled>Run an evaluation</button>
          <span>Contract scaffold · Milestone 0</span>
        </div>
      </section>

      <section className="workflow" aria-labelledby="workflow-heading">
        <div className="section-heading">
          <p className="eyebrow">The loop</p>
          <h2 id="workflow-heading">From experiment to evidence</h2>
        </div>
        <ol>
          {milestones.map(([number, title, copy]) => (
            <li key={number}>
              <span className="step-number">{number}</span>
              <h3>{title}</h3>
              <p>{copy}</p>
            </li>
          ))}
        </ol>
      </section>
    </main>
  );
}
