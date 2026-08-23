import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";

import {
  createEvaluationRun,
  type CreateEvaluationRunRequest,
  listDatasetEvaluationConfigs,
  listEvaluationDatasets,
  listEvaluationQuerySets,
  listEvaluationRuns,
} from "../../api/evaluations";
import { AppLink, RouteHeading } from "../../app/router";
import { navigate } from "../../app/routing";
import {
  OriginBadge,
  PageIntro,
  RequestErrorPanel,
  StatusBadge,
} from "./components";
import { formatDate } from "./formatters";

function CreateRunPanel() {
  const queryClient = useQueryClient();
  const [requestedDatasetId, setRequestedDatasetId] = useState("");
  const [requestedQuerySetId, setRequestedQuerySetId] = useState("");
  const datasets = useQuery({
    queryKey: ["evaluation-datasets"],
    queryFn: ({ signal }) => listEvaluationDatasets(signal),
    retry: false,
  });
  const datasetItems = datasets.data?.datasets ?? [];
  const datasetId = datasetItems.some((item) => item.dataset.id === requestedDatasetId)
    ? requestedDatasetId
    : (datasetItems[0]?.dataset.id ?? "");
  const selectedDataset = datasetItems.find((item) => item.dataset.id === datasetId);
  const querySets = useQuery({
    queryKey: ["evaluation-query-sets", datasetId],
    queryFn: ({ signal }) => listEvaluationQuerySets(datasetId, signal),
    enabled: datasetId.length > 0,
    retry: false,
  });
  const configs = useQuery({
    queryKey: ["evaluation-configs", datasetId],
    queryFn: ({ signal }) => listDatasetEvaluationConfigs(datasetId, signal),
    enabled: datasetId.length > 0,
    retry: false,
  });
  const querySetItems = querySets.data?.query_sets ?? [];
  const querySetId = querySetItems.some((item) => item.query_set.id === requestedQuerySetId)
    ? requestedQuerySetId
    : (querySetItems[0]?.query_set.id ?? "");
  const selectedQuerySet = querySetItems.find((item) => item.query_set.id === querySetId);
  const orderedConfigs = configs.data?.configs ?? [];
  const synthetic =
    selectedDataset?.data_origin === "synthetic_demo" ||
    selectedQuerySet?.data_origin === "synthetic_demo" ||
    configs.data?.data_origin === "synthetic_demo";
  const canonicalCatalogReady = orderedConfigs.length === 4 && querySetId.length > 0;
  const createRun = useMutation({
    mutationFn: (request: CreateEvaluationRunRequest) => createEvaluationRun(request),
    onSuccess: (response) => {
      void queryClient.invalidateQueries({ queryKey: ["evaluation-runs"] });
      navigate(`/runs/${encodeURIComponent(response.result.run.id)}`);
    },
  });

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (synthetic || !canonicalCatalogReady) return;
    const baseline = orderedConfigs[0];
    const candidates = orderedConfigs.slice(1);
    if (baseline === undefined || candidates.length !== 3) return;
    const request: CreateEvaluationRunRequest = {
      contract_version: 1,
      query_set_id: querySetId,
      baseline_config_id: baseline.id,
      candidate_config_ids: candidates.map((config) => config.id),
      random_seed: 20260822,
      max_concurrency: 4,
      warmup_query_count: 5,
    };
    createRun.mutate(request);
  }

  return (
    <section className="create-run-panel" aria-labelledby="create-run-heading">
      <div>
        <p className="eyebrow">Canonical evaluation</p>
        <h2 id="create-run-heading">Start a 50-query run</h2>
        <p>
          One baseline and three candidates produce 200 durable retrieval attempts. Live runs may
          incur embedding and provider usage.
        </p>
      </div>

      {datasets.isPending && <p role="status">Loading evaluation catalogs…</p>}
      {datasets.isError && (
        <RequestErrorPanel
          error={datasets.error}
          heading="Evaluation catalogs are unavailable."
          onRetry={() => void datasets.refetch()}
        />
      )}
      {datasets.isSuccess && datasetItems.length === 0 && (
        <p className="inline-empty" role="status">No persisted evaluation datasets are available.</p>
      )}
      {datasetItems.length > 0 && (
        <form onSubmit={submit}>
          <div className="create-fields">
            <label>
              <span>Dataset revision</span>
              <select
                value={datasetId}
                onChange={(event) => {
                  setRequestedDatasetId(event.target.value);
                  setRequestedQuerySetId("");
                }}
                disabled={createRun.isPending}
              >
                {datasetItems.map((item) => (
                  <option key={item.dataset.id} value={item.dataset.id}>
                    {item.dataset.slug} · {item.dataset.version} · {item.data_origin.replaceAll("_", " ")}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Query set</span>
              <select
                value={querySetId}
                onChange={(event) => setRequestedQuerySetId(event.target.value)}
                disabled={!querySets.isSuccess || querySetItems.length === 0 || createRun.isPending}
              >
                {querySetItems.map((item) => (
                  <option key={item.query_set.id} value={item.query_set.id}>
                    {item.query_set.name} · {item.query_set.query_count} queries
                  </option>
                ))}
              </select>
            </label>
          </div>

          {(querySets.isPending || configs.isPending) && <p role="status">Loading canonical suite…</p>}
          {(querySets.isError || configs.isError) && (
            <div className="inline-warning" role="alert">
              <span>The selected dataset catalog could not be loaded.</span>
              <button
                type="button"
                onClick={() => {
                  void querySets.refetch();
                  void configs.refetch();
                }}
              >
                Retry catalogs
              </button>
            </div>
          )}
          {configs.isSuccess && (
            <ol className="canonical-configs" aria-label="Ordered run configurations">
              {orderedConfigs.map((config, index) => (
                <li key={config.id}>
                  <span>{index === 0 ? "Baseline" : `Candidate ${index}`}</span>
                  <strong>{config.name}</strong>
                  <small>{config.mode.replaceAll("_", " ")}</small>
                </li>
              ))}
            </ol>
          )}

          {synthetic && (
            <p className="synthetic-notice" role="note">
              <strong>Synthetic demo · read-only.</strong> This authored offline dataset has no
              provider timing. Starting a cost-bearing run is disabled.
            </p>
          )}
          {!synthetic && selectedDataset !== undefined && (
            <p className="cost-notice">Starting this live run may make provider-backed requests.</p>
          )}
          {createRun.isError && (
            <RequestErrorPanel
              error={createRun.error}
              heading="The run could not be started."
              onRetry={() => createRun.reset()}
            />
          )}
          <button
            className="primary-action"
            type="submit"
            disabled={synthetic || !canonicalCatalogReady || createRun.isPending}
          >
            {createRun.isPending ? "Starting run…" : "Start evaluation run"}
          </button>
        </form>
      )}
    </section>
  );
}

export function RunListPage({ routeKey }: { routeKey: string }) {
  const runs = useQuery({
    queryKey: ["evaluation-runs", 50],
    queryFn: ({ signal }) => listEvaluationRuns(50, signal),
    retry: false,
  });

  return (
    <section className="dashboard-page">
      <div className="page-heading">
        <p className="eyebrow">Durable evaluation history</p>
        <RouteHeading routeKey={routeKey}>Evaluation runs</RouteHeading>
        <PageIntro>
          Treat search-quality changes like code changes: inspect persisted metrics, coverage, and
          per-query regressions before shipping.
        </PageIntro>
      </div>

      <CreateRunPanel />

      <section className="run-history" aria-labelledby="run-history-heading">
        <div className="section-heading-row">
          <div>
            <p className="eyebrow">Newest first</p>
            <h2 id="run-history-heading">Run history</h2>
          </div>
          {runs.isSuccess && <span>{runs.data.runs.length} runs</span>}
        </div>

        {runs.isPending && <p className="route-loading" role="status">Loading evaluation runs…</p>}
        {runs.isError && (
          <RequestErrorPanel
            error={runs.error}
            heading="Evaluation runs are unavailable."
            onRetry={() => void runs.refetch()}
          />
        )}
        {runs.isSuccess && runs.data.runs.length === 0 && (
          <div className="dashboard-empty" role="status">
            <h3>No evaluation runs yet</h3>
            <p>Seed the offline demo or start a live canonical run when a dataset is ready.</p>
          </div>
        )}
        {runs.isSuccess && runs.data.runs.length > 0 && (
          <div className="table-scroll">
            <table className="run-table">
              <caption className="visually-hidden">Persisted evaluation runs</caption>
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Status</th>
                  <th scope="col">Progress</th>
                  <th scope="col">Origin</th>
                  <th scope="col">Created</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.runs.map((view) => (
                  <tr key={view.run.id}>
                    <th scope="row">
                      <AppLink href={`/runs/${encodeURIComponent(view.run.id)}`}>
                        {view.run.query_set.name}
                      </AppLink>
                      <span className="run-id">{view.run.id}</span>
                      <span className="config-line">
                        {view.configs.map((config) => config.name).join(" · ")}
                      </span>
                    </th>
                    <td><StatusBadge status={view.run.status} /></td>
                    <td>
                      <strong>{view.run.completed_queries} / {view.run.total_queries}</strong>
                      <span className="table-subcopy">query groups</span>
                      <span className="table-subcopy">
                        {view.completed_attempts} / {view.total_attempts} durable attempts
                      </span>
                    </td>
                    <td><OriginBadge origin={view.data_origin} /></td>
                    <td>{formatDate(view.run.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </section>
  );
}
