export function safeSourceUrl(value: string | null | undefined): string | null {
  if (value === null || value === undefined) return null;
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.toString() : null;
  } catch {
    return null;
  }
}
