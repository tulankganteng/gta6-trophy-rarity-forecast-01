#!/usr/bin/env python3
"""Calibrated saturating-growth trophy forecaster.

Forecasts, for every trophy of a game, five quantiles of its Exophase earn
rate (%) at a case's target time, from the history of page reads before the
case's checkpoint.

Method, per trophy (all on the logit scale, rates clamped to 0.01..99.99):

1.  Time axis x = ln(1 + hours since the game's release).  Earn rates grow
    roughly linearly against log-time early in a game's life and then
    saturate, so a straight line in (x, logit(rate)) is a saturating curve
    in ordinary time.
2.  Least-squares line through the distinct reads, with the slope shrunk
    toward a small positive tier prior.  Trophies whose history never moves
    (one distinct read, or one distinct rate) get slope 0: persistence plus
    the prior drift only.  The fitted median is capped by a tier ceiling
    (platinum stays rare; story bronzes may climb much higher).
3.  Spread: a normal band around the fitted median whose width combines
    the fit's residual scale with a random-walk term that grows with the
    horizon, plus an extra term for trophies whose curve is not identified
    by the history at all (flat histories, all-zero rates).  The band is
    deliberately honest: a refresh of Exophase's counts can jump every rate
    at once, and a narrow band pays for every trophy it misses.

Standard library only, no network, no randomness: two runs on the same
case write byte-identical output.  Reads $MAKEFUN_INPUT_DIR (else /input),
writes $MAKEFUN_OUTPUT_DIR (else /output), one .out file per .in file.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

# Standard-normal quantiles for the five scored levels.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

LO = 0.01          # clamp floor for rates, in percent
HI = 99.99         # clamp ceiling for rates, in percent
Q4 = Decimal("0.0001")


def clamp_rate(pct):
    """Clamp an earn rate in percent into the logit-safe range."""
    if pct is None:
        return None
    if pct < LO:
        return LO
    if pct > HI:
        return HI
    return pct


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(z):
    if z >= 0.0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def round4(value):
    """Round to 4 decimals, half-up, as a plain float for JSON output."""
    q = Decimal(repr(value)).quantize(Q4, rounding=ROUND_HALF_UP)
    return float(q)


def parse_time(text):
    """Parse an ISO 8601 timestamp into an aware UTC datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# Tier ceilings: the highest median earn rate (%) this forecaster will ever
# predict for a trophy of each tier by the end of week one.  Platinum is
# earned only by finishing everything; gold by 100% completion or the final
# mission; silver by long grinds; bronze by story chapters and side content.
TIER_CEILING = {
    "platinum": 3.0,
    "gold": 12.0,
    "silver": 25.0,
    "bronze": 60.0,
}

# Tier prior drift: logit units per unit of log-time, used to shrink a
# fitted slope when the history barely identifies it.  Story-adjacent
# bronzes climb fastest in week one; platinum barely moves at all.
TIER_PRIOR_SLOPE = {
    "platinum": 0.02,
    "gold": 0.10,
    "silver": 0.15,
    "bronze": 0.25,
}

# Tier floor for a trophy that has never registered above 0.00%: the median
# it is allowed to reach by the target, in percent.  Keeps an all-zero
# history from forecasting exactly 0.01% while staying honest about rarity.
TIER_ZERO_FLOOR = {
    "platinum": 0.01,
    "gold": 0.02,
    "silver": 0.03,
    "bronze": 0.05,
}


def is_story(description):
    """True when the description reads like a story-mission completion."""
    text = (description or "").lower()
    return ("complete chapter" in text
            or "complete the final mission" in text
            or text.startswith("complete "))


def _saturating_fit(points, tier, last_rate):
    """Fit rate(x) = A / (1 + exp(-(a + b*x))) by grid search over (A, b).

    A saturating curve in log-time: the earn rate climbs toward a tier
    ceiling instead of growing forever.  A straight line in (x, logit)
    extrapolates the early growth rate without end and overshoots the
    asymptote badly over a 4.5-day horizon, so the line is only used to
    seed the search, never to extrapolate.  Returns (a, b, A, resid) or
    None when the history cannot identify a curve.
    """
    ceiling = TIER_CEILING.get(tier, 60.0)
    max_rate = max(math.exp(y) / (1.0 + math.exp(y)) * 100.0
                   for _, y in points)
    if max_rate <= LO:
        return None
    # Asymptote grid: from just above the highest rate seen up to the tier
    # ceiling, geometrically spaced.
    lo_a = max(max_rate * 1.05, LO * 2.0)
    hi_a = ceiling
    if lo_a >= hi_a:
        return None
    n_a = 12
    a_grid = [lo_a * (hi_a / lo_a) ** (k / (n_a - 1.0)) for k in range(n_a)]
    b_grid = [0.2 + 0.2 * k for k in range(15)]  # 0.2 .. 3.0

    best = None  # (sse, a, b, A)
    for A in a_grid:
        for b in b_grid:
            # For fixed (A, b), a is the mean of u - b*x, where u is the
            # logit of the rate as a fraction of its asymptote.
            us = []
            ok = True
            for x, y in points:
                r = 100.0 / (1.0 + math.exp(-y))  # rate in percent
                frac = r / A
                if frac >= 0.999:
                    frac = 0.999
                if frac <= 0.001:
                    frac = 0.001
                us.append(math.log(frac / (1.0 - frac)) - b * x)
            a = sum(us) / len(us)
            sse = 0.0
            for (x, y), u in zip(points, us):
                pred_rate = A / (1.0 + math.exp(-(a + b * x)))
                sse += (y - logit(pred_rate)) ** 2
            if best is None or sse < best[0]:
                best = (sse, a, b, A)
    if best is None:
        return None
    sse, a, b, A = best
    resid = math.sqrt(sse / max(len(points) - 3, 1))
    return a, b, A, resid


def fit_trophy(series, tier, horizon_days, target_x, last_rate):
    """Forecast one trophy.

    series: list of (x, logit_rate) distinct reads, oldest first.
    Returns (median_pct, sigma_logit).
    """
    ceiling = TIER_CEILING.get(tier, 60.0)
    prior_slope = TIER_PRIOR_SLOPE.get(tier, 0.25)
    zero_floor = TIER_ZERO_FLOOR.get(tier, 0.05)

    # A trophy pinned at (near) 100% in the last read: Exophase says every
    # tracked player has it.  Forecast it to stay there, with a small band
    # in case the tracked population dilutes it down a little.
    if last_rate is not None and last_rate >= 99.5:
        return 99.9, 0.35

    # Collapse the series to distinct (x, y) points; identical reads carry
    # no information about movement.
    points = []
    seen = set()
    for x, y in series:
        key = (round(x, 6), round(y, 6))
        if key not in seen:
            seen.add(key)
            points.append((x, y))

    n = len(points)
    last_logit = logit(last_rate) if last_rate is not None else logit(LO)

    if n < 2:
        # No movement visible at all: persistence plus the tier drift.
        slope = 0.0
        intercept = last_logit
        resid = 0.0
        identified = False
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in points)
        if sxx <= 1e-12:
            slope = 0.0
            intercept = my
            resid = 0.0
            identified = False
        else:
            raw_slope = sxy / sxx
            # Shrink the fitted slope toward the tier prior.  The more
            # distinct reads, the more the data is trusted; with two reads
            # the prior still carries half the weight.
            w = n / (n + 2.0)
            slope = w * raw_slope + (1.0 - w) * prior_slope
            intercept = my - slope * mx
            resid = math.sqrt(
                sum((y - (intercept + slope * x)) ** 2 for x, y in points)
                / max(n - 2, 1))
            identified = len(set(round(y, 4) for _, y in points)) > 1

    if not identified:
        # Flat history: the curve is unidentified.  Let the median drift
        # gently from the last rate toward the tier prior, and widen the
        # band a lot -- a flat launch-night page says nothing about where
        # the first refresh lands.
        drift = prior_slope * max(target_x - points[-1][0], 0.0) if points else 0.0
        median_logit = last_logit + 0.5 * drift
        sigma = 0.55 * math.sqrt(horizon_days) + 0.85
        if last_rate is not None and last_rate <= LO + 1e-9:
            # Never above zero yet: lift the median to the tier floor.
            median_logit = max(median_logit, logit(zero_floor))
            sigma = max(sigma, 1.6)
    else:
        sat = _saturating_fit(points, tier, last_rate)
        if sat is not None:
            a, b, asym, resid = sat
            median_logit = logit(asym / (1.0 + math.exp(-(a + b * target_x))))
            # Residual scale plus a random-walk term that grows with
            # horizon; the asymptote itself is uncertain, so the band
            # also widens when the fit rests on few distinct reads.
            sigma = math.sqrt(resid * resid + 0.15 * horizon_days + 0.04)
            if n < 5:
                sigma += 0.15
        else:
            # Fallback: shrunken straight line in (x, logit), used only
            # when no saturating curve fits (e.g. all-zero history).
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            mx = sum(xs) / n
            my = sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in points)
            raw_slope = sxy / sxx if sxx > 1e-12 else 0.0
            w = n / (n + 2.0)
            slope = w * raw_slope + (1.0 - w) * prior_slope
            intercept = my - slope * mx
            median_logit = intercept + slope * target_x
            sigma = math.sqrt(0.25 * horizon_days + 0.25) + 0.3

    # Keep the median inside the tier ceiling and the logit-safe range.
    median_logit = min(median_logit, logit(ceiling))
    median_logit = max(median_logit, logit(LO))
    median = sigmoid(median_logit)
    return median, sigma


def forecast_case(case):
    """Return the output object for one parsed case dict."""
    release = parse_time(case["game"]["release_date"])
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    target_x = math.log1p(max((target - release).total_seconds(), 0.0) / 3600.0)

    trophies = case["trophies"]
    history = case.get("history", [])

    # Per-trophy series of (x, logit rate), oldest first.
    series = {t["id"]: [] for t in trophies}
    last_rate = {}
    for read in history:
        read_at = parse_time(read["at"])
        hours = max((read_at - release).total_seconds(), 0.0) / 3600.0
        x = math.log1p(hours)
        rates = read.get("rates") or []
        for idx, trophy in enumerate(trophies):
            if idx >= len(rates):
                break
            rate = rates[idx]
            if rate is None:
                continue
            last_rate[trophy["id"]] = rate
            series[trophy["id"]].append((x, logit(rate)))

    rows = []
    for trophy in trophies:
        tid = trophy["id"]
        tier = (trophy.get("tier") or "bronze").lower()
        rate = last_rate.get(tid)
        if rate is None:
            rate = LO
        median, sigma = fit_trophy(
            series[tid], tier, horizon_days, target_x, rate)
        med_logit = logit(median)
        qs = []
        prev = -1.0
        for z in Z:
            q = sigmoid(med_logit + z * sigma)
            q = round4(q)
            if q < 0.0:
                q = 0.0
            if q > 100.0:
                q = 100.0
            if q < prev:
                q = prev
            prev = q
            qs.append(q)
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":"), sort_keys=False)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
