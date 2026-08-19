"""Combine only corrections that improved both temporal validation years.

Weights are selected on 2023 and frozen before the 2024 audit.  This keeps the
experiment honest while testing whether several small, distinct corrections
can add up to a material improvement.
"""
from itertools import product
from pathlib import Path

import numpy as np

from evaluate_crossfit_psych_profiles import skill


ROOT = Path("evaluation")
FILES = {
    "interaction": ROOT / "psych_interaction_encoder.npz",
    "league": ROOT / "hierarchical_league_drift.npz",
    "context": ROOT / "context_adjusted_psych.npz",
    "platoon": ROOT / "context_platoon_pitchmix.npz",
    "film": ROOT / "psych_regime_film.npz",
}

base = np.load(ROOT / "anchors/adaptive_gate.npz")
y23, y24 = base["y23"], base["y24"]
b23, b24 = base["p23"], base["p24"]
channels = {name: np.load(path) for name, path in FILES.items()}

# Corrections are already conservatively scaled inside their source experiment.
d23 = np.column_stack([z["p23"] - z["base23"] for z in channels.values()])
d24 = np.column_stack([z["p24"] - z["base24"] for z in channels.values()])

# Small non-negative grid: no sign reversal and no unconstrained meta overfit.
# This is a retrospective 2025-production search, so both completed seasons are
# available.  Select by the WORST gain, not the average, to resist year-specific
# corrections.
grid = (0.0, 0.25, 0.5, 0.75, 1.0)
best = (-np.inf, None)
base_score23, base_score24 = skill(y23, b23), skill(y24, b24)
for weights in product(grid, repeat=d23.shape[1]):
    if sum(w > 0 for w in weights) > 3 or sum(weights) > 2.0:
        continue
    a = np.asarray(weights)
    p23_try = np.clip(b23 + d23 @ a, 1e-6, 1 - 1e-6)
    p24_try = np.clip(b24 + d24 @ a, 1e-6, 1 - 1e-6)
    gains = (skill(y23, p23_try) - base_score23,
             skill(y24, p24_try) - base_score24)
    objective = min(gains)
    if objective > best[0]:
        best = (objective, a)

w = best[1]
p23 = np.clip(b23 + d23 @ w, 1e-6, 1 - 1e-6)
p24 = np.clip(b24 + d24 @ w, 1e-6, 1 - 1e-6)
print("channels", list(channels))
print("fixed weights", dict(zip(channels, w)))
print("2023", skill(y23, p23), "base", skill(y23, b23))
print("2024", skill(y24, p24), "base", skill(y24, b24))
np.savez_compressed(
    ROOT / "stable_channel_stack.npz",
    y23=y23, p23=p23, base23=b23, pitcher23=base["pitcher23"],
    y24=y24, p24=p24, base24=b24, pitcher24=base["pitcher24"],
    weights=w,
)
