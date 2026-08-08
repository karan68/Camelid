/* Shared metric formatting for the Observatory surface — one convention for
   the rail tiles, request log, and Run details panel (space before the unit;
   ms below one second, then seconds at one decimal; rates at one decimal).
   Delegates to lib/formatters so the app keeps a single formatting voice;
   missing values render as an em dash. */

import { formatBytes, formatDurationMs, formatRate } from '../../lib/formatters'

export const fmtMs = (value) => (Number.isFinite(value) ? formatDurationMs(value) : '—')
export const fmtRate = (value) => (Number.isFinite(value) ? formatRate(value) : '—')
export const fmtBytes = (value) => (Number.isFinite(value) && value > 0 ? formatBytes(value) : '—')
