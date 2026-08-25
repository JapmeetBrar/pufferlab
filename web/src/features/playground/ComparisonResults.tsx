import type { SearchCompareResponse } from "../../api/client";
import { configLabel } from "../configLabels";
import { safeSourceUrl } from "./safeUrl";

type ConfigResult = SearchCompareResponse["results"][number];
type SearchHit = ConfigResult["hits"][number];

function formatScore(hit: SearchHit): string {
  if (hit.final_score === null || hit.final_score === undefined) {
    return "Score not observed";
  }
  const { value, kind } = hit.final_score;
  return [
    value.toLocaleString(undefined, { maximumFractionDigits: 5 }),
    kind.replaceAll("_", " "),
  ].join(" · ");
}

function ResultColumn({
  result,
  selectedDocumentId,
  onInspectDocument,
}: {
  result: ConfigResult;
  selectedDocumentId?: string | null;
  onInspectDocument?: (documentId: string, trigger: HTMLButtonElement) => void;
}) {
  const productionTimings = result.timings.filter((timing) => timing.stage !== "provenance_probe");
  const debugTimings = result.timings.filter((timing) => timing.stage === "provenance_probe");
  const label = configLabel(result.config);

  return (
    <article className="result-column" aria-labelledby={`config-${result.config.id}`}>
      <header className="result-column-header">
        <div>
          <h3 id={`config-${result.config.id}`}>{label}</h3>
        </div>
        <span className="result-count">{result.hits.length} hits</span>
      </header>

      <div className="timing-strip" aria-label={`${label} request timings`}>
        {productionTimings.map((timing) => (
          <span key={timing.stage}>
            <strong>{timing.stage === "turbopuffer" ? "Provider" : timing.stage}</strong>
            {timing.duration_ms.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms client
            wall clock
          </span>
        ))}
      </div>
      {debugTimings.map((timing) => (
        <p className="debug-timing" key={timing.stage}>
          Debug provenance probe · {timing.duration_ms.toLocaleString(undefined, {
            maximumFractionDigits: 1,
          })} ms client wall clock · measured separately
        </p>
      ))}

      {result.warnings.length > 0 && (
        <ul className="warnings" aria-label={`${label} warnings`}>
          {result.warnings.map((warning) => (
            <li key={`${warning.code}-${warning.message}`}>{warning.message}</li>
          ))}
        </ul>
      )}

      {result.hits.length === 0 ? (
        <div className="column-empty">No documents matched this configuration.</div>
      ) : (
        <ol className="hit-list">
          {result.hits.map((hit) => {
            const sourceUrl = safeSourceUrl(hit.url);
            const selected = hit.document_id === selectedDocumentId;
            return (
              <li key={hit.document_id} className={`hit-card${selected ? " is-selected" : ""}`}>
                <div className="rank" aria-label={`Rank ${hit.final_rank}`}>
                  {hit.final_rank.toString().padStart(2, "0")}
                </div>
                <div className="hit-content">
                  <h4>{hit.title}</h4>
                  <p className="external-id">{hit.external_id}</p>
                  <p className="excerpt">{hit.body_excerpt}</p>
                  <div className="hit-meta">
                    <span>{formatScore(hit)}</span>
                    {hit.relevance_grade !== null && hit.relevance_grade !== undefined && (
                      <span>Relevance grade {hit.relevance_grade}</span>
                    )}
                  </div>
                  <div className="hit-actions">
                    {sourceUrl !== null && (
                      <a
                        href={sourceUrl}
                        target="_blank"
                        rel="noreferrer"
                        aria-label={`Open source for ${hit.title}`}
                      >
                        Open source<span className="visually-hidden"> for {hit.title}</span>
                      </a>
                    )}
                    {onInspectDocument !== undefined && (
                      <button
                        type="button"
                        onClick={(event) => onInspectDocument(hit.document_id, event.currentTarget)}
                      >
                        Inspect evidence<span className="visually-hidden"> for {hit.title}</span>
                      </button>
                    )}
                  </div>
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </article>
  );
}

function ComparisonSummary({ response }: { response: SearchCompareResponse }) {
  const overlap = response.overlap[0];
  const moved = response.rank_movements.filter(
    (movement) => movement.max_absolute_delta !== null && movement.max_absolute_delta !== undefined,
  );
  return (
    <section className="evidence-summary" aria-labelledby="evidence-heading">
      <div>
        <p className="eyebrow">Observed evidence</p>
        <h2 id="evidence-heading">Where the rankings differ</h2>
      </div>
      <dl>
        <div>
          <dt>Shared results</dt>
          <dd>{overlap === undefined ? "Not available" : `${overlap.intersection_count} documents`}</dd>
        </div>
        <div>
          <dt>Jaccard overlap</dt>
          <dd>{overlap === undefined ? "Not available" : overlap.jaccard.toFixed(2)}</dd>
        </div>
        <div>
          <dt>Rank movement</dt>
          <dd>{moved.length} observed</dd>
        </div>
      </dl>
      <p className="observability-notice">{response.observability_notice}</p>
    </section>
  );
}

export function ComparisonResults({
  response,
  selectedDocumentId,
  onInspectDocument,
}: {
  response: SearchCompareResponse;
  selectedDocumentId?: string | null;
  onInspectDocument?: (documentId: string, trigger: HTMLButtonElement) => void;
}) {
  if (response.results.length === 0) {
    return (
      <div className="empty-state" role="status">
        <h3>No comparison results were returned.</h3>
        <p>Adjust the query or confirm that the namespace is ready, then try again.</p>
      </div>
    );
  }
  return (
    <>
      <ComparisonSummary response={response} />
      <div className="result-grid">
        {response.results.map((result) => (
          <ResultColumn
            result={result}
            selectedDocumentId={selectedDocumentId}
            onInspectDocument={onInspectDocument}
            key={result.config.id}
          />
        ))}
      </div>
    </>
  );
}
