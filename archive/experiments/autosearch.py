"""Unattended hypothesis search under the three-season worst-case rule.

Every candidate is judged the same way: train on seasons < t, predict season t,
for t in 2022, 2023 and 2024, and keep it only if it beats the incumbent on the
worst of the three. Selecting on 2024 alone is what produced a year of local
gains that never reached the leaderboard, so that door is closed here.

Results append to `autosearch_log.json`. A hypothesis that fails is recorded as
dead and skipped on later runs, so the loop can be re-invoked as often as wanted
and will always spend its time on untried ground.

  ./trainenv311/bin/python autosearch.py --list
  ./trainenv311/bin/python autosearch.py --run all
  ./trainenv311/bin/python autosearch.py --run irm_hstate,trackman_v2
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "autosearch_log.json"
TARGETS = (2022, 2023, 2024)
# The incumbent single-model reference, measured under exactly this protocol.
INCUMBENT = {2022: 2394.0, 2023: -705.1, 2024: 834.4}


def bss(y, p):
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    y = np.asarray(y, float)
    v = y.mean() * (1 - y.mean())
    return 1e5 * (1 - ((p - y) ** 2).mean() / v)


def decompose(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    v = y.mean() * (1 - y.mean()); bias = p.mean() - y.mean()
    vp = p.var(); cov = ((p - p.mean()) * (y - y.mean())).mean()
    beta = cov / vp if vp > 0 else float("nan")
    return {"score": bss(y, p), "refine": 1e5 * cov * cov / (vp * v) if vp > 0 else float("nan"),
            "slope_loss": 1e5 * vp * (1 - beta) ** 2 / v if vp > 0 else float("nan"),
            "intercept_loss": 1e5 * bias * bias / v, "beta": beta, "bias": bias}


def load_log() -> dict:
    if LOG.exists():
        return json.loads(LOG.read_text(encoding="utf-8"))
    return {"runs": []}


def record(entry: dict) -> None:
    log = load_log()
    log["runs"].append(entry)
    LOG.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")


def already_done(name: str) -> dict | None:
    for run in reversed(load_log()["runs"]):
        if run["name"] == name and run.get("status") == "ok":
            return run
    return None


def evaluate(name: str, fn, force: bool = False) -> dict:
    prior = already_done(name)
    if prior and not force:
        print(f"{name:24s} (cached) worst={prior['worst']:.1f} {prior['verdict']}")
        return prior
    started = time.time()
    try:
        per = {}
        for t in TARGETS:
            y, p = fn(t)
            per[t] = decompose(y, p)
            print(f"  {name} {t}: score={per[t]['score']:9.1f} beta={per[t]['beta']:.3f}", flush=True)
        scores = {t: per[t]["score"] for t in TARGETS}
        worst_gain = min(scores[t] - INCUMBENT[t] for t in TARGETS)
        entry = {"name": name, "status": "ok", "seconds": round(time.time() - started, 1),
                 "per_season": {str(t): per[t] for t in TARGETS},
                 "worst": min(scores.values()), "mean": float(np.mean(list(scores.values()))),
                 "worst_gain_vs_incumbent": worst_gain,
                 "verdict": "ADOPT" if worst_gain > 0 else "DEAD"}
    except Exception:
        entry = {"name": name, "status": "error", "seconds": round(time.time() - started, 1),
                 "traceback": traceback.format_exc()[-2000:], "verdict": "ERROR"}
        print(f"{name:24s} ERROR\n{entry['traceback']}")
    record(entry)
    if entry["status"] == "ok":
        print(f"{name:24s} worst={entry['worst']:.1f} gain={entry['worst_gain_vs_incumbent']:+.1f} "
              f"{entry['verdict']}  [{entry['seconds']:.0f}s]", flush=True)
    return entry


def main() -> None:
    from hypotheses import REGISTRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="comma separated names, or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if already logged")
    args = ap.parse_args()

    if args.list or not args.run:
        log = load_log()
        seen = {r["name"]: r.get("verdict") for r in log["runs"]}
        print(f"{'HYPOTHESIS':24s} {'STATUS':10s} DESCRIPTION")
        for name, (fn, doc) in REGISTRY.items():
            print(f"{name:24s} {seen.get(name, 'untried'):10s} {doc}")
        return

    names = list(REGISTRY) if args.run == "all" else [n.strip() for n in args.run.split(",")]
    for name in names:
        if name not in REGISTRY:
            print(f"unknown hypothesis {name!r}"); continue
        evaluate(name, REGISTRY[name][0], force=args.force)

    log = load_log()
    ok = [r for r in log["runs"] if r.get("status") == "ok"]
    if ok:
        print(f"\n{'HYPOTHESIS':24s}{'worst':>10}{'mean':>10}{'gain':>10}  verdict")
        for r in sorted({r['name']: r for r in ok}.values(), key=lambda r: -r["worst_gain_vs_incumbent"]):
            print(f"{r['name']:24s}{r['worst']:>10.1f}{r['mean']:>10.1f}"
                  f"{r['worst_gain_vs_incumbent']:>+10.1f}  {r['verdict']}")


if __name__ == "__main__":
    main()
