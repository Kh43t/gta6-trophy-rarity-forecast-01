[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kh43t/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kh43t/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
# GTA VI trophy rarity forecast — make.fun entry

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of Exophase-tracked players who will hold it at 2026-11-26T00:00:00Z, as five
quantiles per trophy. Entry for
[`gta6-trophy-rarity-forecast-01`](https://make.fun) ($250, scored by mean
pinball loss on the logit scale).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` is Node 22, standard library only, no network, deterministic. It
reads every `*.in` case in the input directory and writes one
`<id>.out` per case to the output directory.

## Model (in brief)

A trophy's rate is holders over tracked players, and both only ever grow —
but Exophase's page is a stale snapshot between its refreshes (hours apart),
so the program works on the distinct snapshots only. Per trophy it fits the
logit of the rate against log time since release, recency-weighted, with the
slope shrunk toward its trophy family's slope and damped the further the
target lies past the last read (completion curves flatten). The forecast is
floored at today's holders over tomorrow's tracked base, and the quantiles
come from a two-branch mixture: a refresh happens before the target (the
trend branch) or the answering read still shows the stale snapshot. When the
history has never moved — launch night — a launch prior takes over: the
tracked base grows fast, new players hold almost nothing yet, and a snapshot
that has sat unchanged for most of a refresh gap is very likely to refresh.
Chapter medians are forced non-increasing (nested player sets) and the
platinum no commoner than the rarest trophy.

## Local validation

No local grade is possible for this kit (the scorer and reference are
server-only), so the entry was validated with a from-spec scorer and
reconstructed practice answers: on the two practice cases it scores
~0.12–0.13 mean pinball loss across four plausible answer scenarios, against
~1.4–1.6 for the house persistence baseline. On four synthetic final-like
cases (60 h of history, 4.5-day horizon, saturating and non-saturating
truths) it scores 0.12–0.20. Runs in ~0.2 s per case, byte-identical on
repeat runs.

## License

MIT — see `LICENSE`.
