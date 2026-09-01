# 1200 rebuild audit — 2026-08-31

## Metric and target

- Official metric: `100000 * (1 - Brier / (r * (1-r)))`.
- A score of 1200 requires prediction/label correlation about `0.10954` after
  affine calibration; 1150 corresponds to about `0.10724`.
- Test rows must remain independent. Only the row itself, official train, and
  official Trackman history are permitted.

## Evidence policy

1. Never convert a new local direction to a site score.
2. A correction must improve every chronological transfer used for selection.
3. Prefer a broad plateau over a single optimum.
4. Reject features whose sign changes by season.
5. An actual failed site direction overrides attractive local validation.

## Re-analysis results

### Pressure/form ridge

Low-dimensional count pressure, LI, runners, score, recent success and recent
middle-rate interactions were trained on 2022 and audited on 2023, then trained
on 2022+2023 and audited on 2024.

- best 2023 gain: `-5.12`
- corresponding 2024 gain: `+0.60`
- verdict: reject; seasonal sign reversal.

### Post-Tensor residual CatBoost

Sixteen depth/regularization models were trained on reconstructed 2023
RobustTensor residuals and audited on 2024. Full and R-only scopes and nine
strengths were tested.

- best gain at the weakest 0.025 scale: `-0.31`
- verdict: reject; the anchor already absorbs general row-feature residuals.

### Season maturity

Current-season success was reconstructed row-locally from each row's official
`asof_pitcher_n/rate` and a frozen prior-season train snapshot.

- best robust setting: 2023 `+42.37`, 2024 `+1.38`
- actual SeasonMaturity submission: `1148.577`, below OfficialQuad
- verdict: do not deploy; direction is too unstable and has negative site evidence.

### New Tensor axes

Eight previous-season residual axes were screened after the reconstructed JM
Tensor anchor. The best was pitcher × batter hand × base state.

- 2023 `+2.59`, 2024 `+0.99`
- verdict: valid but immaterial; cannot support a 50-point claim.

### Archived correction hull after Tensor

- split/platoon directions can show `+17..21` locally after Tensor.
- those directions are highly correlated (`0.90..0.92`).
- split prior fell on site despite `+32` local; hand175 fell from 1150 to 1119.
- verdict: local improvement is not transferable enough for the final submission.

## Feature decisions

Keep as primary signals:

- pitcher career success and reverse rates
- recent 3/5-game success, conservatively shrunk
- official as-of sample sizes
- low-cardinality count/base/game-type state
- frozen, high-confidence Trackman profiles already present in the anchor

Do not add as new high-weight corrections:

- batter career rate (relationship collapses after 2022)
- raw pitch-mix rates (seasonal sign changes)
- win expectancy and generic pressure crosses (near-zero incremental value)
- high-dimensional player/context residual tables
- hand/platoon corrections on top of JM Tensor
- leaderboard-fitted affine or anti directions

## Conclusion

No newly tested legal direction supplies a defensible 50-point improvement over
the 1150 OfficialQuad anchor. The evidence-supported incremental ceiling among
available artifacts is roughly 20 local points, and its dominant directions
have already failed live transfer. No file should be labelled or presented as a
1200 candidate without new model/data evidence.
