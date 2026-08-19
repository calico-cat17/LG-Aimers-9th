"""Run the organiser's rule 3) check against a submission zip.

  "어떤 행의 예측값은 아래 두 경우 모두 동일해야 합니다.
     - test.csv에 해당 행 1개만 있는 경우
     - test.csv에 전체 평가 데이터가 함께 있는 경우"

The zip is unpacked and its own script.py is executed several times against
test.csv files holding different subsets of the same rows. If any row's
prediction moves, some other row influenced it.

  ./trainenv311/bin/python check_row_independence.py submissions/<name>.zip
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / "trainenv311/bin/python")


def make_eval_frame(n_rows: int, seed: int) -> pd.DataFrame:
    """A stand-in evaluation file: real 2024 rows relabelled as the 2025 season."""
    columns = pd.read_csv(ROOT / "data/test.csv", nrows=1).columns
    train = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    rows = train[train.season == 2024].sample(n_rows, random_state=seed).reset_index(drop=True)
    rows["season"] = 2025
    rows["row_id"] = [f"TEST_{i:06d}" for i in range(len(rows))]
    return rows[columns]


def predict(workdir: Path, frame: pd.DataFrame) -> pd.Series:
    frame.to_csv(workdir / "data/test.csv", index=False)
    done = subprocess.run([PYTHON, "script.py"], cwd=workdir, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"inference failed:\n{done.stdout}\n{done.stderr}")
    out = pd.read_csv(workdir / "output/submission.csv")
    return out.set_index("row_id").control_success


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--singles", type=int, default=6, help="rows also predicted completely alone")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5, help="above this a diff counts as real cross-row leakage")
    args = ap.parse_args()

    frame = make_eval_frame(args.rows, args.seed)
    workdir = Path(tempfile.mkdtemp(prefix="rowcheck-"))
    try:
        with zipfile.ZipFile(args.zip_path) as z:
            z.extractall(workdir)
        (workdir / "data").mkdir(exist_ok=True)

        full = predict(workdir, frame)

        checks: list[tuple[str, pd.Series, pd.Series]] = []
        # A different set of neighbours: only the first half of the rows.
        half = frame.iloc[: len(frame) // 2]
        checks.append(("half file", full.reindex(half.row_id), predict(workdir, half)))
        # Same rows, different order. Any order-dependent feature shows up here.
        shuffled = frame.sample(frac=1.0, random_state=args.seed + 1)
        checks.append(("shuffled file", full.reindex(shuffled.row_id), predict(workdir, shuffled)))
        # The strict form of the rule: one row, entirely alone.
        picks = frame.iloc[:: max(1, len(frame) // args.singles)].head(args.singles)
        alone = pd.concat([predict(workdir, frame.iloc[[i]]) for i in picks.index])
        checks.append((f"{len(picks)} rows alone", full.reindex(alone.index), alone))

        # float32 kernels accumulate in a different order once the batch size
        # changes, so a few 1e-7 wobbles are arithmetic, not information flow.
        # Any real cross-row aggregation moves predictions by 1e-3 or more.
        print(f"\n=== 행 독립성 검사: {Path(args.zip_path).name} ===")
        print(f"기준 실행: {len(frame)}행 일괄 예측  (누수 판정 임계 {args.tol:.0e})\n")
        worst = 0.0
        for name, reference, other in checks:
            gap = float(np.max(np.abs(reference.to_numpy(float) - other.to_numpy(float))))
            worst = max(worst, gap)
            if gap <= 1e-9:
                verdict = "동일"
            elif gap <= args.tol:
                verdict = "부동소수점 오차"
            else:
                verdict = "누수"
            print(f"  {name:<22} rows={len(other):5d}  max|diff| = {gap:.3e}   {verdict}")
        print()
        if worst <= args.tol:
            note = "" if worst <= 1e-9 else f" (최대 {worst:.1e}는 float32 누산 오차 범위)"
            print(f"PASS - 다른 행의 존재가 예측을 바꾸지 않습니다{note}")
            return 0
        print(f"FAIL - 최대 차이 {worst:.3e}. 다른 행이 추론에 영향을 주고 있습니다.")
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
