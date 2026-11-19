// GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).
//
// Forecasts, for every trophy, five quantiles of its Exophase earn rate at the
// case's target, as a pinball loss on the logit scale. Node 22, standard
// library only, no network, deterministic: the same case always writes the
// same bytes.
//
// How it thinks about the data
// ----------------------------
// A trophy's rate is a count over a player base: rate = holders / tracked.
// Both move. Holders only ever grow (players earn trophies and keep them);
// the tracked base only ever grows (more players sync profiles). What the
// page shows between Exophase's refreshes is a stale snapshot: long flat
// stretches, then a jump. So the program works in count space, on the
// distinct snapshots only (consecutive identical reads carry no new
// information), and never trusts a flat stretch to mean nothing is changing.
//
// Per trophy it fits the logit of the rate against log time since release,
// on the distinct snapshots only, recency-weighted: a trophy's rate is a
// completion curve, concave in log time once the early rush passes, so the
// fitted slope is damped the further the target lies past the last read
// (a straight logit line would run away to 100%). Slopes are shrunk toward
// their trophy group's slope (story chapters move together, grind trophies
// move together). The forecast is floored at the count-preserving level:
// today's holders over tomorrow's tracked base, since holders never fall.
// When the history has never moved at all -- the launch-night practice
// cases -- there is no slope to fit, so it falls back to a launch prior: the tracked base grows fast in the first hours after
// release, new players hold almost nothing yet, and a snapshot that has sat
// unchanged for most of a refresh gap is very likely to refresh before the
// target.
//
// The forecast is a two-branch mixture in logit space: with probability p a
// refresh (or the normal evolution) happens before the target, giving a
// lognormal around the trend median; with probability 1 - p the answering
// read shows the same stale snapshot, a point mass at the last rate. The
// five quantiles are read off that mixture exactly. Story chapters are
// nested sets of players (you cannot finish chapter 2 without chapter 1),
// so the chapter medians are forced non-increasing, and the platinum
// (every other trophy) is forced no rarer than the rarest trophy.
//
// Nothing here reads the clock, the network, or any randomness: every number
// comes from the case file, so two runs agree byte for byte.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

/** Release day begins at midnight UTC on the game's release date. */
const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z");
const MS_PER_HOUR = 3_600_000;

/** The levels a row reports, in order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

/** Exophase refreshes a game's counts hours apart; one observed gap ran ~4.7 h. */
const REFRESH_GAP_HOURS = 4.5;
/** Tracked-base growth per hour just after launch, before it decays with age. */
const LAUNCH_GROWTH_PER_HOUR = 0.4;
/** How fast the launch growth rate itself decays (hours). */
const GROWTH_DECAY_HOURS = 72;
/** Fractional holder growth a refresh reveals, on average, when history is flat. */
const FLAT_HOLDER_GROWTH = 0.3;
/** Logit spread pieces: fit noise floor, horizon term per sqrt day. */
const SPREAD_FLOOR = 0.3;
const SPREAD_HORIZON_PER_SQRT_DAY = 0.22;
const SPREAD_CAP = 1.15;
/** How far past the last read the fitted slope is trusted, in log-time units:
// the slope's pull is multiplied by 1 / (1 + dx / TREND_DAMP_X). */
const TREND_DAMP_X = 2;

type Trophy = {
  id: number;
  name: string;
  description: string;
  tier: string;
};

type Read = {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
};

type CaseFile = {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
};

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logitPct = (pct: number): number => {
  const p = clampPct(pct);
  return Math.log(p / (100 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const hoursSinceRelease = (iso: string): number =>
  (Date.parse(iso) - RELEASE_MS) / MS_PER_HOUR;
/** log-time axis: hours since release, compressed the way progress slows. */
const logTime = (hours: number): number => Math.log(1 + Math.max(0, hours));

/** Standard normal CDF via the Abramowitz-Stegun erf approximation. */
const normCdf = (z: number): number => {
  const sign = z < 0 ? -1 : 1;
  const ax = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * ax);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-ax * ax);
  return 0.5 * (1 + sign * erf);
};

/** Inverse standard normal CDF by bisection (deterministic, ~1e-12). */
const normQuantile = (p: number): number => {
  let lo = -10;
  let hi = 10;
  for (let k = 0; k < 80; k++) {
    const mid = (lo + hi) / 2;
    if (normCdf(mid) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
};

/** Weighted least squares of y on x; returns [intercept, slope]. */
const fitLine = (xs: number[], ys: number[], ws: number[]): [number, number] => {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let k = 0; k < xs.length; k++) {
    sw += ws[k];
    sx += ws[k] * xs[k];
    sy += ws[k] * ys[k];
  }
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let k = 0; k < xs.length; k++) {
    sxx += ws[k] * (xs[k] - mx) * (xs[k] - mx);
    sxy += ws[k] * (xs[k] - mx) * (ys[k] - my);
  }
  if (sxx <= 1e-12) return [my, 0];
  const slope = sxy / sxx;
  return [my - slope * mx, slope];
};

const medianOf = (values: number[]): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/** Which behavioural family a trophy belongs to, from its name and text. */
const familyOf = (t: Trophy): string => {
  const text = `${t.name} ${t.description}`.toLowerCase();
  if (/final mission for/.test(text)) return "strand";
  if (/complete chapter|complete the final mission/.test(text)) return "story";
  if (/complete \d+|gold medal|meditate at every|100 locations|collectible card sets|50 challenges/.test(text)) return "grind";
  return "misc";
};

/** A read counts as new only when the snapshot actually changed. */
const snapshotChanged = (a: Read, b: Read): boolean => {
  if (a.players_tracked !== b.players_tracked) return true;
  for (let i = 0; i < a.rates.length; i++) {
    if (a.rates[i] !== b.rates[i]) return true;
  }
  return false;
};

/**
 * Quantiles of the two-branch mixture at the case's levels. Branch one is a
 * normal in logit space around `centre` with sd `sd`, weight `p`; branch two
 * is a point mass at `flatLogit`, weight 1 - p. The point mass makes the CDF
 * jump at `flatLogit`: a level inside the jump reads exactly that value.
 */
const mixtureQuantiles = (
  centre: number,
  sd: number,
  flatLogit: number,
  p: number,
  levels: number[],
): number[] => {
  const below = p * normCdf((flatLogit - centre) / sd);
  return levels.map((tau) => {
    let g: number;
    if (tau < below) {
      g = centre + sd * normQuantile(tau / p);
    } else if (tau > below + (1 - p)) {
      g = centre + sd * normQuantile((tau - (1 - p)) / p);
    } else {
      g = flatLogit;
    }
    return invLogitPct(g);
  });
};

/** The persistence forecast, used only when the model cannot run at all. */
const fallbackRows = (c: CaseFile): Array<{ id: number; q: number[] }> => {
  const last = c.history[c.history.length - 1];
  const spread = 0.5 * Math.sqrt(c.horizon_minutes / 1440);
  return c.trophies.map((t, i) => {
    const rate = last?.rates[i] ?? 1;
    const l = logitPct(rate ?? 1);
    return {
      id: t.id,
      q: LEVELS.map((tau) => round4(invLogitPct(l + normQuantile(tau) * spread))),
    };
  });
};

const forecastCase = (c: CaseFile): Array<{ id: number; q: number[] }> => {
  const levels = c.levels.length === LEVELS.length ? c.levels : LEVELS;
  const horizonHours = c.horizon_minutes / 60;
  const horizonDays = c.horizon_minutes / 1440;
  const checkpointHours = hoursSinceRelease(c.at);
  const targetX = logTime(checkpointHours + horizonHours);

  // Distinct snapshots only, oldest first, with their log-time axis.
  const snaps: Read[] = [];
  for (const read of c.history) {
    if (snaps.length === 0 || snapshotChanged(snaps[snaps.length - 1], read)) snaps.push(read);
  }
  const last = snaps[snaps.length - 1];
  const lastChangeAtMs = Date.parse(last.at);
  // How stale the snapshot the checkpoint sees is, in hours.
  const stalenessHours = Math.max(0, (Date.parse(c.at) - lastChangeAtMs) / MS_PER_HOUR);

  // Recency weights: a read half a day old counts half, with a floor.
  const weights = snaps.map((s) => {
    const ageH = Math.max(0, (Date.parse(c.at) - Date.parse(s.at)) / MS_PER_HOUR);
    return Math.pow(0.5, ageH / 24) + 0.1;
  });
  const xs = snaps.map((s) => logTime(hoursSinceRelease(s.at)));

  // --- Tracked-base forecast -------------------------------------------
  const logN = snaps.map((s) => Math.log(Math.max(1, s.players_tracked)));
  let nSlope = 0;
  let nPred = last.players_tracked;
  if (snaps.length >= 2) {
    const [, slope] = fitLine(xs, logN, weights);
    nSlope = Math.min(2, Math.max(0, slope));
    // Shrink a two-point slope toward the launch's own power law (~1).
    if (snaps.length === 2) nSlope = (nSlope + 2 * 1) / 3;
    const [nIntercept] = fitLine(xs, logN, weights);
    nPred = Math.exp(nIntercept + nSlope * targetX);
    nPred = Math.min(Math.max(nPred, last.players_tracked), 5_000_000);
  } else {
    // No movement ever seen: the base grows through the stale window plus
    // the horizon at the launch rate, which itself decays with age.
    const growthRate = LAUNCH_GROWTH_PER_HOUR / (1 + Math.max(0, checkpointHours) / GROWTH_DECAY_HOURS);
    const window = Math.min(stalenessHours + horizonHours, 12);
    nPred = last.players_tracked * Math.exp(growthRate * window);
  }

  // --- Per-trophy holder fits ------------------------------------------
  // Holders implied by each snapshot; rounding the 2-decimal rates back to
  // whole players keeps the counts honest when the base is small.
  const allHolders: number[][] = c.trophies.map((_, trophyIndex) =>
    snaps.map((s) => {
      const r = s.rates[trophyIndex];
      if (r === null || r === undefined) return NaN;
      return Math.round((r * s.players_tracked) / 100);
    }),
  );
  const knownPoints = (i: number): number =>
    allHolders[i].filter((v) => Number.isFinite(v)).length;

  // --- Per-trophy rate fits ---------------------------------------------
  // Logit of the rate against log time, on the distinct snapshots only.
  // Also remembered per trophy: its last known logit and log-time (the anchor
  // the damped extrapolation starts from) and its last known holder count.
  const rawSlopes: number[] = [];
  const residSd: number[] = [];
  const lastLogit: number[] = [];
  const lastX: number[] = [];
  const lastKnownCount: number[] = [];
  const families = c.trophies.map(familyOf);
  for (let i = 0; i < c.trophies.length; i++) {
    const pts = snaps
      .map((s, k) => ({ x: xs[k], y: s.rates[i] === null || s.rates[i] === undefined ? NaN : logitPct(s.rates[i] as number), w: weights[k] }))
      .filter((p) => Number.isFinite(p.y));
    let anchor = pts.length > 0 ? pts[pts.length - 1] : { x: xs[xs.length - 1], y: logitPct(0.01), w: 1 };
    lastLogit.push(anchor.y);
    lastX.push(anchor.x);
    let known = NaN;
    for (let k = allHolders[i].length - 1; k >= 0; k--) {
      if (Number.isFinite(allHolders[i][k])) {
        known = allHolders[i][k];
        break;
      }
    }
    lastKnownCount.push(known);
    if (pts.length >= 2 && snaps.length >= 2) {
      const [intercept, slope] = fitLine(
        pts.map((p) => p.x),
        pts.map((p) => p.y),
        pts.map((p) => p.w),
      );
      let resid = 0;
      for (const p of pts) resid += p.w * (p.y - (intercept + slope * p.x)) ** 2;
      const wsum = pts.reduce((a, p) => a + p.w, 0);
      residSd.push(Math.sqrt(resid / Math.max(wsum, 1e-9)));
      rawSlopes.push(slope);
    } else {
      residSd.push(0.3);
      rawSlopes.push(0);
    }
  }

  // Shrink each slope toward its family's median slope (groups of trophies
  // that unlock the same way move together); fall back to the global median.
  const globalSlope = medianOf(rawSlopes.filter((s, i) => knownPoints(i) >= 2));
  const familySlope = new Map<string, number>();
  for (const fam of ["story", "strand", "grind", "misc"]) {
    const members = rawSlopes.filter(
      (s, i) => families[i] === fam && knownPoints(i) >= 2,
    );
    familySlope.set(fam, members.length > 0 ? medianOf(members) : globalSlope);
  }
  const slopes = rawSlopes.map((s, i) => {
    const group = familySlope.get(families[i]) ?? globalSlope;
    const w = Math.min(snaps.length, 8);
    return Math.min(3, Math.max(-1, (w * s + 3 * group) / (w + 3)));
  });

  // --- Refresh probability ------------------------------------------------
  // Roughly periodic refreshes: a snapshot older than most of a gap is due.
  let pRefresh: number;
  if (stalenessHours >= REFRESH_GAP_HOURS) {
    pRefresh = 0.97;
  } else {
    const dueIn = Math.max(1, REFRESH_GAP_HOURS - stalenessHours);
    pRefresh = Math.min(0.97, horizonHours / dueIn);
  }
  pRefresh = Math.max(pRefresh, 1 - Math.exp(-horizonHours / 9));
  pRefresh = Math.min(0.97, Math.max(0.05, pRefresh));

  // --- Per-trophy medians -------------------------------------------------
  // Easiness weight for the flat-history prior: higher-rate trophies are the
  // ones new players reach first. Dampened so the spread stays modest.
  const lastRates = c.trophies.map((_, i) => last.rates[i]);
  const easeBase = lastRates.map((r) => Math.sqrt(Math.max(0.01, r ?? 0.01)));
  const easeMean = easeBase.reduce((a, b) => a + b, 0) / Math.max(1, easeBase.length);
  const ease = easeBase.map((e) => Math.pow(e / Math.max(easeMean, 1e-9), 0.5));

  const medians: number[] = [];
  const spreads: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const rate = last.rates[i];
    const knownCount = lastKnownCount[i];
    let med: number;
    if (rate !== null && rate !== undefined && rate >= 99.995) {
      // Everyone holds it; the next refresh can only add players who hold it
      // too (it is the first thing a player does). Handled after the loop.
      med = 99.97;
    } else if (snaps.length >= 2 && Number.isFinite(knownCount)) {
      // Damped logit extrapolation from the last known rate: the slope's
      // pull fades with extrapolation distance, since completion curves
      // flatten as the pool of players yet to earn it shrinks.
      const dx = Math.max(0, targetX - lastX[i]);
      const damp = 1 / (1 + dx / TREND_DAMP_X);
      med = invLogitPct(lastLogit[i] + slopes[i] * dx * damp);
      // Holders cannot fall: today's holders over tomorrow's base is a floor.
      const floorRate = (Math.max(0, knownCount) / nPred) * 100 * 0.95;
      med = Math.max(med, floorRate);
    } else if (Number.isFinite(knownCount)) {
      // Flat history: holders grow a little, the base grows a lot. nPred
      // already holds the flat-history base forecast from above.
      const countPred = Math.max(0, knownCount) * (1 + FLAT_HOLDER_GROWTH * ease[i]);
      med = (countPred / nPred) * 100;
    } else {
      // Never seen held by anyone: rare, and the spread will say so.
      med = 0.5;
    }
    medians.push(clampPct(med));
    const resid = residSd[i];
    const horizonTerm = SPREAD_HORIZON_PER_SQRT_DAY * Math.sqrt(Math.max(0, horizonDays));
    spreads.push(Math.min(SPREAD_CAP, Math.max(SPREAD_FLOOR, Math.sqrt(resid * resid + horizonTerm * horizonTerm + 0.01))));
  }

  // --- Quantiles, then the structural constraints ------------------------
  let rows = c.trophies.map((t, i) => {
    const rate = last.rates[i];
    let q: number[];
    if (rate !== null && rate !== undefined && rate >= 99.995) {
      q = [99.0, 99.8, 99.97, 99.99, 99.99];
    } else {
      q = mixtureQuantiles(logitPct(medians[i]), spreads[i], logitPct(rate ?? 0.01), pRefresh, levels).map(
        (v) => round4(clampPct(v)),
      );
    }
    return { id: t.id, q };
  });

  // Chapter medians must not rise along the story (nested player sets), and
  // the platinum cannot be commoner than the rarest trophy. Fix the medians
  // (q at the 0.5 level), then repair each row's ordering.
  const midPos = levels.indexOf(0.5) >= 0 ? levels.indexOf(0.5) : 2;
  const atMedian = (idx: number): number => rows[idx].q[midPos];
  const chain = [2, 3, 4, 5, 6].filter((idx) => idx < rows.length);
  for (let k = 1; k < chain.length; k++) {
    if (atMedian(chain[k]) > atMedian(chain[k - 1])) {
      rows[chain[k]].q[midPos] = atMedian(chain[k - 1]);
    }
  }
  if (rows.length > 1) {
    let rarest = Infinity;
    for (let idx = 1; idx < rows.length; idx++) rarest = Math.min(rarest, atMedian(idx));
    if (atMedian(0) > rarest) rows[0].q[midPos] = rarest;
  }
  rows = rows.map((row) => {
    const q = [...row.q];
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    for (let k = q.length - 2; k >= 0; k--) q[k] = Math.min(q[k], q[k + 1]);
    return { id: row.id, q: q.map((v) => round4(clampPct(v))) };
  });
  return rows;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith(".in"))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as CaseFile;
  let rows: Array<{ id: number; q: number[] }>;
  try {
    rows = forecastCase(c);
  } catch {
    rows = fallbackRows(c);
  }
  if (rows.length !== c.trophies.length) rows = fallbackRows(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}
