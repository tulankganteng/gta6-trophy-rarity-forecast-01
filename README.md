# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tulankganteng/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tulankganteng/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

**Calibrated saturating-growth trophy forecaster — logit/log-time curves, tier priors, honest spread.** Forecasts, for every GTA VI PlayStation 5 trophy, five quantiles of its Exophase earn rate (%) at 2026-11-26T00:00:00Z, from the house's recording of the trophy page up to the deadline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, no install. In the grader the case is mounted at `/input` and the output goes to `/output`. The program is deterministic: two runs on the same case write byte-identical output. One case takes ~0.06 s on the flat practice history and ~1.9 s on a full 289-read final-like history — well under the 8 s limit.

## The model

Score is mean pinball loss on the **logit** scale, so a miss from 1% to 2% costs about what a miss from 33% to 50% does, and calibration decides the bounty. The model is built for the final's shape — two days of real movement plus a 4.5-day horizon — not for the practice cases, whose histories are a single unchanged launch-night page and whose answers jump on Exophase's first refresh.

Per trophy, all on the logit scale (rates clamped to 0.01–99.99%):

1. **Time axis** `x = ln(1 + hours since release)`. Earn rates grow roughly linearly against log-time early in a game's life and then saturate.
2. **Saturating curve, not a straight line.** A straight line in (x, logit) extrapolates the early growth rate without end and overshoots the asymptote badly over 4.5 days (in validation it scored 0.56 vs the persistence baseline's 0.28). Instead the model fits `rate(x) = A / (1 + exp(-(a + b·x)))` by grid search over the asymptote `A` (from just above the highest rate seen up to a tier ceiling) and the growth rate `b`, with `a` solved by least squares. The median can never exceed the fitted asymptote.
3. **Tier priors.** Ceilings: platinum 3%, gold 12%, silver 25%, bronze 60%. Prior drift and zero-rate floors are tiered the same way (platinum barely moves in week one; story bronzes climb fastest).
4. **Flat histories stay honest.** A trophy whose history never moves (one distinct read, or one distinct rate — every practice-case trophy) gets persistence plus a gentle tier drift and a *wide* band: a flat launch-night page says nothing about where the first refresh lands.
5. **The 100% trophy.** A trophy pinned at (near) 100% in the last read (id 2, "Just Like Paradise", reads 100.00% in every practice read and still 100% on the live page ~12 h after launch) is forecast to stay there, median 99.9% with a small band, instead of letting its logit blow up the spread logic.
6. **Spread.** Normal band on the logit scale: fit-residual scale plus a random-walk term growing with the horizon (`0.15·horizon_days`), plus an extra widening for unidentified curves and all-zero histories. The band is deliberately honest — Exophase refreshes its counts on its own schedule and one refresh can jump every rate at once (launch night: 638 → 3,868 players tracked, every rate but the 100% one lower).

No trophy is ever special-cased by name. Everything is derived from the case history at run time; the live-page rates seen during development were used only as a prior sanity check.

## Validation (see TESTLOG.md)

Two synthetic final-like worlds, 12 cases each, with known truth — one where the truth is a logistic curve in log-time, one where it is a Weibull saturation in linear time (a shape the forecaster never assumes):

| world | model loss | persistence baseline | typical miss (model / baseline) | 90% coverage |
|---|---|---|---|---|
| logistic in log-time | **0.210** | 0.276 | 1.30 / 2.62 pp | 1.00 |
| Weibull in linear time | **0.410** | 0.528 | 3.35 / 3.65 pp | 1.00 |

The model beats the house persistence baseline by ~24% and ~22% in the two worlds. Coverage reads 1.00 in the synthetic worlds because their refreshes are gentler than launch night's; the band is kept wide on purpose for the real refresh jumps. A stress test that halves every rate at the target (a dilution shock no history could predict) still leaves the model at 0.33 vs the baseline's 0.28 — the one regime where persistence wins, and the price of hedging it would be paid in every calm world, so no downward hedge is applied.

Practice cases p00/p01 run clean (valid shape, 55 rows, deterministic, 0.06 s); their true answers are held by the server, so they are a format check only — they are not tuned to and cannot validate the final's week.

## Files

- `main.py` — the whole entry (stdlib only)
- `LICENSE` — MIT
