# GTA VI trophy rarity forecast — entry

`main.py` (Python 3.13, standard library only) forecasts, for each of the
55 GTA VI PS5 trophies, five quantiles of its Exophase earn rate at the
case's target, from the house's recorded reads of the trophy page.

## The idea

An earn rate is a ratio of two counts that only move when Exophase
refreshes the page: the players tracked, N, and the earners of each trophy,
E_i. So the program models the two counts separately and divides at the
target, instead of extrapolating the printed percentages directly.

**When the history moves (the final case).** Both counts grow roughly as
power laws in the time since release, so each is fitted in log space
against u = ln(1 + hours since release midnight), with an exponentially
recency-weighted least-squares line — a read's weight halves every ~0.55
units of u, about the last day of history at the deadline. The recency
weighting matters because earners saturate: a trophy everyone gets early
stops growing long before the target, and a fit over the whole history
would carry its steep early climb forward and forecast it near 100%.
Forecasts are anchored on the last read, not on the fitted intercept.
Per-trophy earner slopes are shrunk toward the cross-trophy median slope,
and clamped so earners never shrink and never outrun players by much.

On top of that, each trophy gets a saturating-rate fit — r(t) = cap ·
(1 − exp(−t/τ)) and r(t) = cap · t/(t + τ), fitted by a grid search over τ
with the cap solved by weighted least squares. The lower of the trusted
fits caps the power-law forecast: it only ever lowers it, never raises
it. The second (Hill) form matters near 100%: a trophy at 96% creeps
toward 100% slowly, and the exponential form would call it 100% at the
target, which the logit scale punishes severely if the truth is 98%.

**When the history is flat (the practice cases).** No slope can be
estimated, so the program falls back to a refresh mixture: with some
probability the page is still unrefreshed at the target (rate unchanged),
and otherwise one refresh lands before the target and dilutes every rate.
The dilution is not a uniform discount — the launch-night refresh the
rules describe proves the new players a refresh adds hold almost nothing
beyond the one trophy everyone holds. Tracked players went 638 → 3,868
(+3,230) and trophies earned 943 → 4,276 (+3,333); the trophy at 100%
stayed at 100%, so it alone accounts for +3,230 of the +3,333, leaving
only +103 trophies spread over the other 54 — about 0.03 trophies per new
player beyond the first one. So a refresh is modelled as: every new
player holds the saturated trophies, and each other trophy at a rate
proportional to its current rate, scaled so the non-saturated trophies
total ~3.2 per 100 new players. The refresh probability grows with the
horizon and with how stale the page already is, against a ~4.5-hour
refresh interval (the first refresh came after ~4.5 hours of staleness).

**Spreads.** Quantiles are logit-normal bands around the median: a base
term for refresh timing and model error, a term growing with the square
root of the horizon, each trophy's own fit residual, a wider band for
trophies held by very few players, and extra width when the power-law
and saturating forecasts disagree. The trophy at 100% gets a tight band
near 100%.

## Local checks

- Practice cases p00/p01: valid outputs (55 trophies, ids 1–55, five
  finite non-decreasing quantiles each, ~2.8 KB per file), byte-identical
  across two runs, ~0.1 s for both cases.
- Practice comparison against the starter baseline, scored with the
  pinball loss from the rules under the stated assumption that the
  practice answers are the refreshed page (new players all holding the
  100% trophy and sharing only +103 trophies over the other 54, in
  proportion to each trophy's rate — the only distribution consistent
  with the published aggregates): this entry 0.38 vs starter 1.40. The
  starter's 1.40 under that assumption is close to the ~1.51 the rules
  state for it, which supports the assumption. Varying the assumed split
  by ±50% gives 0.44 / 0.35 for this entry vs 1.53 / 1.28 for the
  starter. Under a uniform 0.748× discount (which the aggregates
  refute — it would predict +4,310 trophies, not +3,333) the starter
  would score 0.20 vs this entry's 0.28.
- Synthetic final-like cases (358 reads, ~200+ distinct refreshes):
  median absolute logit error of the median forecast 0.03 against a
  truth from saturating-exponential curves and 0.17 against a truth from
  a different (Hill-type) family, with 90% coverage 0.98–1.00 and a
  0.1 s run time. These test the machinery, not real-world accuracy —
  the synthetic truths are invented.
- Edge cases (single read, empty history, null rates, 600 reads) all run
  and produce valid output.
