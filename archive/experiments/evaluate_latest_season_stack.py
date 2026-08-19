from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "old" / "experiments"


def bss(y, p):
    bs = np.mean((np.clip(p, 1e-5, 1 - 1e-5) - y) ** 2)
    return 100000.0 * (1.0 - bs / (y.mean() * (1.0 - y.mean())))


files = sorted(EXP.glob("*.npz")) + sorted(ROOT.glob("*.npz"))
names, cols = [], []
y = month = None
for f in files:
    z = np.load(f)
    if y is None and "y" in z and z["y"].shape == (253507,):
        y = z["y"].astype(np.float64)
    if "month" in z and z["month"].shape == (253507,):
        month = z["month"].astype(int)
    for key in ("p", "main", "p2024"):
        if key in z and z[key].shape == (253507,):
            names.append(f.stem + ":" + key)
            cols.append(z[key].astype(np.float64))
    if "risks" in z and z["risks"].shape == (253507, 3):
        for j, tag in enumerate(("middle", "wild", "reverse")):
            names.append(f.stem + ":" + tag)
            cols.append(z["risks"][:, j].astype(np.float64))

assert y is not None and month is not None
X = np.column_stack(cols)
finite = np.all(np.isfinite(X), axis=0)
X, names = X[:, finite], [n for n, ok in zip(names, finite) if ok]
# Exact duplicate predictions add conditioning problems but no information.
keep = []
for j in range(X.shape[1]):
    if not any(np.max(np.abs(X[:, j] - X[:, k])) < 1e-10 for k in keep):
        keep.append(j)
X, names = X[:, keep], [names[j] for j in keep]

fit = month <= 5
tune = (month >= 6) & (month <= 7)
test = month >= 8

print("models", X.shape[1], "rows", fit.sum(), tune.sum(), test.sum())
for mask_name, mask in (("fit", fit), ("tune", tune), ("late", test), ("all", np.ones(len(y), bool))):
    ranked = sorted(((bss(y[mask], X[mask, j]), names[j]) for j in range(X.shape[1])), reverse=True)
    print(mask_name, "best", ranked[:5])


def ridge_predict(alpha, base_j, tr, va):
    # Learn only corrections around a strong anchor. This is much safer than free stacking.
    base = X[:, base_j]
    D = X - base[:, None]
    mu, sd = D[tr].mean(0), D[tr].std(0) + 1e-6
    Z = (D - mu) / sd
    A = np.column_stack([np.ones(tr.sum()), Z[tr]])
    reg = np.eye(A.shape[1]) * alpha
    reg[0, 0] = 0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ (y[tr] - base[tr]))
    return np.clip(base[va] + np.column_stack([np.ones(va.sum()), Z[va]]) @ coef, 1e-5, 1 - 1e-5), coef


# Select anchor only on fit, tune regularization only on tune, report late once.
base_j = max(range(X.shape[1]), key=lambda j: bss(y[fit], X[fit, j]))
print("anchor", names[base_j])
choices = []
for alpha in (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000):
    pred, _ = ridge_predict(alpha, base_j, fit, tune)
    score = bss(y[tune], pred)
    choices.append((score, alpha))
    print("ridge tune", alpha, score)
alpha = max(choices)[1]
pred_late, coef = ridge_predict(alpha, base_j, fit | tune, test)
print("ridge chosen", alpha, "late", bss(y[test], pred_late), "anchor late", bss(y[test], X[test, base_j]))

# Convex stack: nonnegative weights summing to one. Fit early, choose L2 on tune.
conv = []
for lam in (0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1):
    def obj(w):
        return np.mean((X[fit] @ w - y[fit]) ** 2) + lam * np.sum((w - 1 / X.shape[1]) ** 2)
    w0 = np.zeros(X.shape[1]); w0[base_j] = 1
    res = minimize(obj, w0, method="SLSQP", bounds=[(0, 1)] * X.shape[1],
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"maxiter": 300, "ftol": 1e-12})
    sc = bss(y[tune], X[tune] @ res.x)
    conv.append((sc, lam, res.x))
    print("convex tune", lam, sc, "active", (res.x > .005).sum())
_, lam, _ = max(conv, key=lambda t: t[0])
def obj2(w):
    return np.mean((X[fit | tune] @ w - y[fit | tune]) ** 2) + lam * np.sum((w - 1 / X.shape[1]) ** 2)
w0 = np.zeros(X.shape[1]); w0[base_j] = 1
res = minimize(obj2, w0, method="SLSQP", bounds=[(0, 1)] * X.shape[1],
               constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
               options={"maxiter": 500, "ftol": 1e-12})
print("convex chosen", lam, "late", bss(y[test], X[test] @ res.x), "active")
for j in np.argsort(res.x)[::-1][:12]:
    if res.x[j] > 1e-4: print(" ", round(res.x[j], 5), names[j])
