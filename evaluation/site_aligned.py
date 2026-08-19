"""Local 2024 evaluation in the same parameterisation the leaderboard sees.

Until now the local number and the site number described different objects. A
candidate was measured locally at full strength, then shipped as one channel
scaled to about half - so `split_prior_stack` read +32 locally and -4.8 live.
This evaluates the exact shipped form: the gate prediction, the channels at the
weights the manifest carries, and the affine constants solved off the leaderboard.

The target follows from score = 1e5 * rho^2. Site refine is 1121.27, so reaching
1200 needs rho up 3.45%, which on the 927.40 anchor means 992.5.
"""
from __future__ import annotations

import numpy as np

# solved from four scored submissions; see submissions/ready/README.md
SITE = {"K": 402463.0, "ybar": 0.46088, "K_cov": 1073.361, "K_var": 1027.503}
SITE_REFINE = SITE["K_cov"] ** 2 / SITE["K_var"]
ANCHOR_2024 = 927.4038
# Current best adds a genuinely new F-only signal, so its forward-2024 affine
# ceiling and live score are the relevant target-conversion anchor going forward.
PREVIOUS_LOCAL_CEILING = 928.6524664793
PREVIOUS_SITE_SCORE = 1122.2576768019
CURRENT_LOCAL_CEILING = 956.6019414908
CURRENT_SITE_SCORE = 1126.4544105659
# The latest same-family experiment gained 27.95 local ceiling points but only
# 4.20 live points. Future R/F refinements are discounted by this measured
# incremental transfer instead of the old proportional mapping.
INCREMENTAL_TRANSFER = ((CURRENT_SITE_SCORE - PREVIOUS_SITE_SCORE) /
                        (CURRENT_LOCAL_CEILING - PREVIOUS_LOCAL_CEILING))
CHANNEL_WEIGHTS = {"psych": 0.6782, "split": 0.5098, "film": -0.3057,
                   "minimax": -0.5914, "hier": -0.3688}


def local_target(site_target: float, anchor: float = CURRENT_LOCAL_CEILING) -> float:
    """Local 2024 score that corresponds to a given site score."""
    return anchor + (site_target - CURRENT_SITE_SCORE) / INCREMENTAL_TRANSFER


def score(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    v = y.mean() * (1 - y.mean())
    return 1e5 * (1 - ((p - y) ** 2).mean() / v)


def affine_ceiling(y, p):
    """Score after the best possible global shift and stretch - what the site
    constants already buy, so a candidate must beat this to be worth anything."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    v = y.mean() * (1 - y.mean())
    cov = ((p - p.mean()) * (y - y.mean())).mean()
    return 1e5 * cov * cov / (p.var() * v)


def report(y, p, name="candidate", baseline=None):
    s, ceil = score(y, p), affine_ceiling(y, p)
    print(f"{name:28s} score={s:8.2f}  affine-ceiling={ceil:8.2f}  "
          f"site-equivalent={CURRENT_SITE_SCORE + INCREMENTAL_TRANSFER * (ceil-CURRENT_LOCAL_CEILING):8.1f}")
    if baseline is not None:
        print(f"{'':28s} vs baseline {ceil - affine_ceiling(y, baseline):+8.2f} (ceiling basis)")
    return ceil


if __name__ == "__main__":
    print(f"site refine        = {SITE_REFINE:.2f}")
    print(f"current site best  = {CURRENT_SITE_SCORE:.4f}")
    print(f"current local ceil = {CURRENT_LOCAL_CEILING:.2f}")
    print(f"increment transfer = {INCREMENTAL_TRANSFER:.4f} site/local")
    for t in (1150, 1180, 1200):
        print(f"  site {t}  ->  local 2024 target {local_target(t):.1f}"
              f"   (+{local_target(t)-CURRENT_LOCAL_CEILING:.1f})")
