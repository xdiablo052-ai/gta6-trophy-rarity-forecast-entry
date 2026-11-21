// gta6-trophy-rarity-forecast-01 — "Dilution-aware log-time forecaster".
//
// Model: an earn rate is earners / players_tracked. The two move differently in
// launch week: the tracked pool grows in rushes (each Exophase refresh dilutes
// every rate), while earners per trophy grow as players progress. So the
// forecast fits them separately, in log space against log-time since launch,
// over the case's distinct refresh states (consecutive identical reads are one
// state — Exophase only changes at refreshes), then divides at the target.
// Quantiles ride on the logit scale with a spread from the fit residual plus a
// random-walk term in the horizon. Deterministic, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Launch epoch the log-time axis is anchored to (first story trophies unlock). */
const E0 = Date.parse('2026-11-19T00:00:00Z');

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: Array<{ id: number }>;
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Hours since launch epoch for an ISO timestamp, floored at 0. */
const hoursOf = (iso: string): number => Math.max(0, (Date.parse(iso) - E0) / 3_600_000);

/** Weighted least squares y ~ a + b*x with recency weights; null if degenerate. */
function fitLine(xs: number[], ys: number[], ws: number[]): { a: number; b: number; resid: number } | null {
  const n = xs.length;
  if (n < 3) return null;
  let sw = 0, sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sw += ws[i]; sx += ws[i] * xs[i]; sy += ws[i] * ys[i]; }
  const mx = sx / sw, my = sy / sw;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += ws[i] * (xs[i] - mx) ** 2; sxy += ws[i] * (xs[i] - mx) * (ys[i] - my); }
  if (sxx <= 1e-12) return null;
  const b = sxy / sxx;
  const a = my - b * mx;
  let sr = 0;
  for (let i = 0; i < n; i++) sr += ws[i] * (ys[i] - a - b * xs[i]) ** 2;
  return { a, b, resid: Math.sqrt(sr / sw) };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hist = c.history;
  if (hist.length === 0) throw new Error(`${file} has no history`);
  const last = hist[hist.length - 1];

  // Collapse to distinct refresh states: a state ends when the page changes.
  const states: Read[] = [];
  for (const r of hist) {
    const prev = states[states.length - 1];
    if (!prev || prev.players_tracked !== r.players_tracked || prev.rates.some((v, i) => v !== r.rates[i])) {
      states.push(r);
    }
  }
  const days = c.horizon_minutes / 1440;
  const tTarget = Math.log1p(hoursOf(c.target));

  // Pool growth: ln(players) against ln(1 + hours), recent states weigh more.
  const poolFit = fitLine(
    states.map((s) => Math.log1p(hoursOf(s.at))),
    states.map((s) => Math.log(Math.max(1, s.players_tracked))),
    states.map((_, i) => 1 + i / Math.max(1, states.length - 1)),
  );
  let playersTarget = Math.max(1, last.players_tracked);
  if (poolFit) {
    const grown = Math.exp(poolFit.a + poolFit.b * tTarget);
    const ratio = Math.min(40, Math.max(1, grown / Math.max(1, last.players_tracked)));
    playersTarget = Math.max(1, last.players_tracked) * ratio;
  }

  const trophies = c.trophies.map((t, i) => {
    const lastRate = last.rates[i] ?? 0;
    const fallbackSigma = 0.5 * Math.sqrt(days) + 0.15;
    // Earners series over distinct states (skip reads missing this trophy).
    const xs: number[] = [], ys: number[] = [], ws: number[] = [];
    states.forEach((s, si) => {
      const r = s.rates[i];
      if (r === null || r === undefined) return;
      const earners = (r / 100) * s.players_tracked;
      xs.push(Math.log1p(hoursOf(s.at)));
      ys.push(Math.log(earners + 0.5));
      ws.push(1 + si / Math.max(1, states.length - 1));
    });
    const fit = fitLine(xs, ys, ws);
    let median = lastRate;
    let sigma = fallbackSigma;
    if (fit) {
      const earnersTarget = Math.max(0, Math.exp(fit.a + fit.b * tTarget) - 0.5);
      const rate = (100 * earnersTarget) / playersTarget;
      // Keep the trend honest: never move more than 3 logit units from the last read,
      // and shrink toward persistence when the fit rests on few refresh states —
      // a line through two refreshes is a guess, through six it is a trend.
      const lLast = logit(lastRate);
      const lTrend = Math.min(lLast + 3, Math.max(lLast - 3, logit(rate)));
      const trust = Math.min(1, Math.max(0.25, (xs.length - 2) / 4));
      median = invLogit(lLast + trust * (lTrend - lLast));
      sigma = Math.min(2.4, Math.max(0.25, 0.42 * Math.sqrt(days) + Math.min(1.2, fit.resid) + (1 - trust) * 0.2));
    }
    const l = logit(median);
    const q = Z.map((z) => round4(invLogit(l + z * sigma)));
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: t.id, q };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
