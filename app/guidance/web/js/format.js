// Small number-formatting helpers used by the result tables.

// fmt(3.14159) -> "3.14", fmt(null) -> "-"
export function fmt(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return '-';
  return Number(value).toFixed(digits);
}

// percentage of `part` in `total`, 0 when total is 0
export function percent(part, total) {
  return total ? (100 * part) / total : 0;
}
