"""Submission wrapper for an official-score-calibrated JM/JY anti-blend."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_base(root: Path, data: Path) -> pd.DataFrame:
    link, output = root / "data", root / "output"
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink(): shutil.rmtree(link)
        else: link.unlink()
    link.symlink_to(data.resolve(), target_is_directory=True)
    if output.exists(): shutil.rmtree(output)
    output.mkdir()
    bootstrap = "import numpy as np,runpy; np.NaN=np.nan; np.Inf=np.inf; runpy.run_path('script.py',run_name='__main__')"
    completed = subprocess.run(
        [sys.executable, "-c", bootstrap], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600,
    )
    print(completed.stdout)
    if completed.returncode: raise RuntimeError(f"base package failed: {root}")
    return pd.read_csv(output / "submission.csv", encoding="utf-8-sig")


def aligned(ids: pd.Series, prediction: pd.DataFrame) -> np.ndarray:
    mapping = pd.Series(
        prediction["control_success"].to_numpy(float),
        index=prediction["row_id"].astype(str),
    )
    values = ids.astype(str).map(mapping)
    if values.isna().any(): raise ValueError(f"missing predictions={int(values.isna().sum())}")
    return values.to_numpy(float)


def main() -> None:
    root, model = Path.cwd(), Path.cwd() / "model"
    data = root / "data"
    if not data.exists():
        data.mkdir()
        for name in ("test.csv", "sample_submission.csv"): shutil.copy2(root / name, data / name)
    manifest = json.loads((model / "antiblend_manifest.json").read_text(encoding="utf-8"))
    jm = run_base(model / "jm_package", data)
    jy = run_base(model / "jy_package", data)
    test = pd.read_csv(data / "test.csv", encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(data / "sample_submission.csv", encoding="utf-8-sig")
    p_jm, p_jy = aligned(test["row_id"], jm), aligned(test["row_id"], jy)
    weight = float(manifest["jy_weight"])
    # Match the officially scored StrictStack direction exactly: F stays JM,
    # while only R moves along the JM/JY axis.
    is_r = test["game_type"].astype(str).eq("R").to_numpy()
    raw = p_jm.copy()
    raw[is_r] = p_jm[is_r] + weight * (p_jy[is_r] - p_jm[is_r])
    clipped = np.clip(raw, 1e-6, 1.0 - 1e-6)
    diagnostics = {
        "jy_weight": weight,
        "scope": "R only",
        "r_rows": int(is_r.sum()),
        "clip_fraction": float(np.mean(raw[is_r] != clipped[is_r])) if is_r.any() else 0.0,
        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
        "mean_abs_adjustment": float(np.mean(np.abs(clipped[is_r] - p_jm[is_r]))) if is_r.any() else 0.0,
        "max_abs_adjustment": float(np.max(np.abs(clipped[is_r] - p_jm[is_r]))) if is_r.any() else 0.0,
    }
    sample["control_success"] = clipped
    (root / "output").mkdir(exist_ok=True)
    sample.to_csv(root / "output" / "submission.csv", index=False, encoding="utf-8")
    print("ANTI_BLEND_DIAGNOSTIC=" + json.dumps(diagnostics, ensure_ascii=False))


if __name__ == "__main__": main()
