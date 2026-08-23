import type { CapabilitiesResponse } from "../../api/client";

type LivePlaygroundCapability = CapabilitiesResponse["live_playground"];

export type CapabilityReadiness =
  | { state: "checking" }
  | { state: "unavailable" }
  | { state: "action_required"; capability: LivePlaygroundCapability }
  | { state: "locally_configured"; capability: LivePlaygroundCapability };

export function currentCapabilityReadiness(query: {
  data: CapabilitiesResponse | undefined;
  isFetching: boolean;
  isSuccess: boolean;
}): CapabilityReadiness {
  if (query.isFetching) return { state: "checking" };
  if (!query.isSuccess || query.data === undefined) return { state: "unavailable" };
  return {
    state: query.data.live_playground.state,
    capability: query.data.live_playground,
  };
}
