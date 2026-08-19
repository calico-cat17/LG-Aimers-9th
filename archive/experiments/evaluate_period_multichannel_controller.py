"""Fit period-specific nonnegative context controllers on 2023, audit once on 2024."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

R = Path(__file__).resolve().parent
z = np.load(R / "group_channels_crossseason.npz")
raw = pd.read_csv(R / "data/train.csv", usecols=["season", "game_month"])
m23 = raw.loc[raw.season.eq(2023), "game_month"].to_numpy()
m24 = raw.loc[raw.season.eq(2024), "game_month"].to_numpy()
names = z["names"]

def score(y, p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return 1e5 * (1 - np.mean((y - p) ** 2) / (y.mean() * (1 - y.mean())))

def periods(m):
    return [m <= 5, (m >= 6) & (m <= 8), m >= 9]

y23, p23, x23 = z["y23"], z["p23"], z["x23"]
y24, p24, x24 = z["y24"], z["p24"], z["x24"]
best = None
for ridge in (100, 300, 1000, 3000, 10000, 30000):
    weights = []
    pred23, pred24 = p23.copy(), p24.copy()
    for a, b in zip(periods(m23), periods(m24)):
        A = np.vstack([x23[a], np.sqrt(ridge) * np.eye(x23.shape[1])])
        target = np.r_[y23[a] - p23[a], np.zeros(x23.shape[1])]
        w = lsq_linear(A, target, bounds=(0, .5), lsmr_tol="auto").x
        pred23[a] += x23[a] @ w
        pred24[b] += x24[b] @ w
        weights.append(w)
    s23, s24 = score(y23, pred23), score(y24, pred24)
    print(ridge, round(s23, 3), round(s24, 3),
          [[(str(n), round(float(v), 3)) for n, v in zip(names, w) if v > .01]
           for w in weights])
    # Selection metric uses 2023 only. 2024 remains a one-time audit column.
    if best is None or s23 > best[0]:
        best = (s23, ridge, weights, pred24)

print("selected_on_2023", best[1], "audit_2024", round(score(y24, best[3]), 3))
np.savez_compressed(R / "period_multichannel_2024.npz", y=y24, p=best[3],
                    weights=np.stack(best[2]), names=names, ridge=best[1])
