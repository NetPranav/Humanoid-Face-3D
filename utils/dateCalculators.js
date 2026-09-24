// Utility: Date calculation helpers
export function addDays(date, days) {
  const result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
}

export function diffInHours(d1, d2) {
  return Math.abs(new Date(d1) - new Date(d2)) / 36e5;
}

// Revision 2 - 2026-09-24
