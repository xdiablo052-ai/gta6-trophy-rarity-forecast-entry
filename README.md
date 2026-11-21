# GTA VI Trophy Rarity Forecaster — dilution-aware log-time model

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xdiablo052-ai/gta6-trophy-rarity-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xdiablo052-ai/gta6-trophy-rarity-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast every GTA VI PS5 trophy's Exophase earn rate at 2026-11-26, as five quantiles per trophy.

## Model

An earn rate is `earners / players_tracked`, and in launch week the two move differently:

- the tracked pool grows in rushes — each Exophase refresh dilutes every rate at once;
- earners per trophy grow as players actually progress through the game.

So `main.ts` fits them separately, in log space against log-time since launch, over the case's **distinct refresh states** (consecutive identical reads collapse into one state — the page only changes at refreshes):

1. pool growth: weighted least squares of `ln(players)` on `ln(1 + hours)`, extrapolated to the target with the growth ratio clamped;
2. per trophy: weighted least squares of `ln(earners + 0.5)` on the same axis;
3. median rate = predicted earners / predicted players, shrunk toward the last read when the fit rests on few refreshes, and never moved more than 3 logit units from it;
4. quantiles on the logit scale with a spread from the fit residual plus a horizon random-walk term.

Deterministic (two runs byte-identical), Node 22 standard library only, no network, runs in well under a second per case. Run it the way the grader does:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

MIT licensed.
