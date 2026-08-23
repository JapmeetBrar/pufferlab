import { useMemo, useSyncExternalStore } from "react";

const navigationEvent = "pufferlab:navigation";

export interface AppLocation {
  pathname: string;
  search: string;
}

function locationSnapshot(): string {
  return `${window.location.pathname}${window.location.search}`;
}

function subscribeToLocation(onStoreChange: () => void): () => void {
  window.addEventListener("popstate", onStoreChange);
  window.addEventListener(navigationEvent, onStoreChange);
  return () => {
    window.removeEventListener("popstate", onStoreChange);
    window.removeEventListener(navigationEvent, onStoreChange);
  };
}

export function useAppLocation(): AppLocation {
  const snapshot = useSyncExternalStore(subscribeToLocation, locationSnapshot, locationSnapshot);
  return useMemo(() => {
    const url = new URL(snapshot, window.location.origin);
    return { pathname: url.pathname, search: url.search };
  }, [snapshot]);
}

export function navigate(to: string, options: { replace?: boolean } = {}): void {
  const target = new URL(to, window.location.origin);
  if (target.origin !== window.location.origin) {
    window.location.assign(target.href);
    return;
  }
  const next = `${target.pathname}${target.search}${target.hash}`;
  if (options.replace === true) {
    window.history.replaceState(null, "", next);
  } else {
    window.history.pushState(null, "", next);
  }
  window.dispatchEvent(new Event(navigationEvent));
}
