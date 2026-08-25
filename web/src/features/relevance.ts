export function relevanceLabel(grade: number): string {
  if (grade <= 0) return "Not relevant";
  if (grade === 1) return "Relevant";
  return "Highly relevant";
}
