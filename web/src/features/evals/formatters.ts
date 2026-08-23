export function formatDate(value: string | null): string {
  if (value === null) return "Not yet";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? "Unavailable"
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
