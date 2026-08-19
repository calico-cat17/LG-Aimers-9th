"""Hierarchical state-space model for pitch control on the logit scale.

The signal in this problem lives almost entirely in a pitcher's latent level and
how it moves between seasons, plus how that pitcher personally responds to a
handful of game situations. A gradient-boosted tree has no machinery for either:
no partial pooling for a 400-level categorical, no state transition across years.
That is why the current pipeline has to hand-build a shrinkage formula and then
bolt a tree on top of its residual.

This model states the structure directly:

    logit p_i = mu_t                     season level
              + theta(asof_i)            pitcher latent level, amortised
              + sum_k b[p_i, k] z_ik     pitcher-specific situation slopes
              + c[batter_i]              batter level
              + f(context_i)             shared situation response

`theta` is an amortised encoder rather than one free parameter per pitcher: it
reads the row's own as-of history (career rate and size, season-to-date rate and
size, recent-game form) and returns that pitcher's level. Trained across seasons
it learns the shrinkage rule that `season_weight = .15 + .30 * n/(n+80)` guesses
by hand, and because it is a function of the row's own columns it keeps working
in a season with no labels at all.

`b` holds the psychological hypotheses - platoon, count pressure, leverage,
runners in scoring position, home/away, recent-form direction. Each pitcher gets
their own slope, drawn toward zero by an L2 whose strength is the pooling
variance. A league-average slope is near zero because pitchers differ in sign;
an unpooled per-pitcher slope overfits. Partial pooling is the middle, and it is
fitted jointly with the levels so a slope cannot absorb level error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

# Situations whose effect is believed to vary by pitcher.
SITUATIONS = (
    "platoon_opposite", "two_strike", "three_ball", "high_leverage",
    "risp", "late_inning", "pitcher_home", "cold_form",
)

# Row-local inputs to the amortised pitcher-level encoder.
STATE_INPUTS = (
    "career_logit", "career_log_n", "season_logit", "season_log_n",
    "recent1_logit", "recent3_logit", "recent5_logit", "recent_std",
    "season_minus_career", "recent_minus_career",
)


def _logit(x, lo=1e-3):
    x = np.clip(np.asarray(x, dtype=np.float64), lo, 1 - lo)
    return np.log(x / (1 - x))


def build_state_inputs(raw: pd.DataFrame, season_rate, season_n, prior: float) -> np.ndarray:
    """Row-local history summary; nothing here touches another evaluation row."""
    num = lambda c, d=np.nan: pd.to_numeric(raw[c], errors="coerce").fillna(d).to_numpy(float)
    career_rate = num("asof_pitcher_success_rate", prior)
    career_n = np.clip(num("asof_pitcher_n", 0), 0, None)
    rec = [num(f"asof_pitcher_prev{k}_game_success_rate", prior) for k in (1, 3, 5)]
    season_rate = np.where(np.asarray(season_n) > 0, season_rate, prior)
    cols = {
        "career_logit": _logit(career_rate),
        "career_log_n": np.log1p(career_n),
        "season_logit": _logit(season_rate),
        "season_log_n": np.log1p(np.clip(season_n, 0, None)),
        "recent1_logit": _logit(rec[0]), "recent3_logit": _logit(rec[1]),
        "recent5_logit": _logit(rec[2]),
        "recent_std": np.nan_to_num(np.std(np.vstack(rec), axis=0), nan=.15),
        "season_minus_career": _logit(season_rate) - _logit(career_rate),
        "recent_minus_career": _logit(rec[1]) - _logit(career_rate),
    }
    return np.column_stack([cols[c] for c in STATE_INPUTS]).astype(np.float32)


def build_situations(raw: pd.DataFrame) -> np.ndarray:
    """Centred situation indicators; centring keeps slopes from moving the level."""
    num = lambda c, d=0.0: pd.to_numeric(raw[c], errors="coerce").fillna(d).to_numpy(float)
    balls, strikes = num("balls_before"), num("strikes_before")
    risp = (num("runner_on_2b") + num("runner_on_3b")) > 0
    career = num("asof_pitcher_success_rate", np.nan)
    prev1 = num("asof_pitcher_prev1_game_success_rate", np.nan)
    z = {
        "platoon_opposite": raw.pitcher_hand.astype(str).ne(raw.batter_hand.astype(str)).to_numpy(),
        "two_strike": strikes == 2,
        "three_ball": balls == 3,
        "high_leverage": num("li") >= 1.5,
        "risp": risp,
        "late_inning": num("inning") >= 7,
        "pitcher_home": raw.top_bottom.astype(str).eq("T").to_numpy(),
        "cold_form": np.nan_to_num(prev1 - career, nan=0.0) < -0.05,
    }
    out = np.column_stack([z[k].astype(np.float32) for k in SITUATIONS])
    return out - out.mean(axis=0, keepdims=True)


class HierarchicalState(nn.Module):
    def __init__(self, n_pitchers: int, n_batters: int, n_context: int,
                 state_dim: int = len(STATE_INPUTS), n_situations: int = len(SITUATIONS),
                 width: int = 64):
        super().__init__()
        self.level = nn.Sequential(
            nn.Linear(state_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.slopes = nn.Embedding(n_pitchers, n_situations)
        self.batter = nn.Embedding(n_batters, 1)
        self.context = nn.Sequential(
            nn.Linear(n_context, width), nn.SiLU(), nn.Linear(width, 1))
        self.season_level = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.slopes.weight)
        nn.init.zeros_(self.batter.weight)
        for m in (self.level[-1], self.context[-1]):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, state, situations, context, pitcher, batter):
        z = self.level(state).squeeze(-1)
        s = (self.slopes(pitcher) * situations).sum(-1)
        return self.season_level + z + s + self.batter(batter).squeeze(-1) + self.context(context).squeeze(-1)

    def penalty(self, pitcher, batter, slope_sd: float, batter_sd: float):
        """Partial pooling: an L2 whose strength is the pooling variance."""
        sl = self.slopes(pitcher)
        bt = self.batter(batter)
        return (sl.pow(2).sum(-1).mean() / (2 * slope_sd ** 2)
                + bt.pow(2).squeeze(-1).mean() / (2 * batter_sd ** 2))
