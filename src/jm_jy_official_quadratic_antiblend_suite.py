"""Build a JM/JY anti-blend from three official Brier leaderboard scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "2026-08-28-jm-jy-official-quadratic-antiblend-v1"


@dataclass(frozen=True)
class Config:
    jm_submission: Path
    jy_submission: Path
    data_dir: Path
    wrapper: Path
    output: Path
    jm_score: float
    jy_score: float
    half_score: float
    max_anti: float


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_root(source: Path, destination: Path) -> Path:
    """Extract one submission package without importing the OOF experiment stack."""
    if destination.exists(): shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(source) as archive: archive.extractall(destination)
    roots = [path.parent for path in destination.rglob("script.py") if (path.parent / "model").is_dir()]
    roots = [path for path in roots if "__MACOSX" not in path.parts]
    if len(roots) != 1: raise ValueError(f"submission roots={roots}")
    return roots[0]


def score_at(weight: np.ndarray | float, jm: float, jy: float, diversity: float):
    weight = np.asarray(weight, dtype=float)
    return (1.0 - weight) * jm + weight * jy + diversity * weight * (1.0 - weight)


def package(config: Config, weight: float, predicted_score: float) -> tuple[dict, dict]:
    stage, model = config.output / "stage", config.output / "stage" / "model"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(); model.mkdir()
    jm_root = normalize_root(config.jm_submission, config.output / "unpack_jm")
    jy_root = normalize_root(config.jy_submission, config.output / "unpack_jy")
    shutil.copytree(jm_root, model / "jm_package")
    shutil.copytree(jy_root, model / "jy_package")
    for root in (model / "jm_package", model / "jy_package"):
        for name in ("data", "output"):
            path = root / name
            if path.exists() or path.is_symlink():
                if path.is_dir() and not path.is_symlink(): shutil.rmtree(path)
                else: path.unlink()
    shutil.copy2(config.wrapper, stage / "script.py")
    (stage / "requirements.txt").write_text(
        "scikit-learn==1.8.0\njoblib==1.5.3\ncatboost==1.2.10\n"
        "xgboost==2.0.3\nlightgbm==4.3.0\n",
        encoding="utf-8",
    )
    write_json(model / "antiblend_manifest.json", {
        "version": VERSION, "jy_weight": weight,
        "formula": "clip(p_jm + jy_weight * (p_jy - p_jm))",
        "predicted_official_score_without_clipping": predicted_score,
        "official_inputs": {"jm": config.jm_score, "jy": config.jy_score, "half": config.half_score},
    })
    for name in ("test.csv", "sample_submission.csv"): shutil.copy2(config.data_dir / name, stage / name)
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=stage, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200,
    )
    (config.output / "submission_smoke.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode: raise RuntimeError(completed.stdout[-5000:])
    match = re.search(r"ANTI_BLEND_DIAGNOSTIC=(\{.*\})", completed.stdout)
    if not match: raise RuntimeError("anti-blend diagnostic missing from smoke output")
    diagnostics = json.loads(match.group(1))
    for name in ("test.csv", "sample_submission.csv"): (stage / name).unlink()
    shutil.rmtree(stage / "data"); shutil.rmtree(stage / "output")
    label = f"anti{abs(weight):.4f}".replace(".", "")
    output_zip = config.output / f"submit_JM_OfficialQuadratic_{label}.zip"
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in stage.rglob("*"):
            if path.is_file(): archive.write(path, path.relative_to(stage).as_posix())
    package_info = {
        "path": str(output_zip), "size_mib": output_zip.stat().st_size / 1024**2,
        "sha256": hashlib.sha256(output_zip.read_bytes()).hexdigest(),
    }
    return package_info, diagnostics


def run(config: Config) -> None:
    started = time.perf_counter(); config.output.mkdir(parents=True, exist_ok=True)
    diversity = 4.0 * (config.half_score - 0.5 * (config.jm_score + config.jy_score))
    if diversity <= 0: raise ValueError(f"non-positive diversity curvature: {diversity}")
    unconstrained = (diversity + config.jy_score - config.jm_score) / (2.0 * diversity)
    selected = float(np.clip(unconstrained, -abs(config.max_anti), 0.0))
    grid = np.unique(np.r_[np.linspace(-abs(config.max_anti), 0.10, 141), selected, unconstrained])
    curve = pd.DataFrame({"jy_weight": grid})
    curve["anti_strength"] = -curve.jy_weight
    curve["predicted_official_score_unclipped"] = score_at(
        curve.jy_weight.to_numpy(), config.jm_score, config.jy_score, diversity
    )
    curve["delta_vs_jm"] = curve.predicted_official_score_unclipped - config.jm_score
    curve.to_csv(config.output / "official_quadratic_curve.csv", index=False)
    selected_score = float(score_at(selected, config.jm_score, config.jy_score, diversity))
    package_info, diagnostics = package(config, selected, selected_score)
    decision = "GO" if selected_score > config.jm_score + 0.5 and diagnostics["clip_fraction"] <= 0.005 else "NO_GO"
    summary = {
        "version": VERSION, "status": "complete", "runtime_seconds": time.perf_counter() - started,
        "official_scores": {"jm": config.jm_score, "jy": config.jy_score, "half": config.half_score},
        "diversity_curvature": diversity, "unconstrained_jy_weight": unconstrained,
        "selected_jy_weight": selected, "selected_anti_strength": -selected,
        "predicted_score_without_clipping": selected_score,
        "predicted_delta_vs_jm_without_clipping": selected_score - config.jm_score,
        "diagnostics": diagnostics, "decision": decision, "package": package_info,
        "warning": "The quadratic prediction is exact before probability clipping; clipping diagnostics determine submit safety.",
    }
    write_json(config.output / "official_quadratic_antiblend_summary.json", summary)
    pd.DataFrame([summary | {"package": package_info["path"]}]).to_csv(
        config.output / "official_quadratic_candidate.csv", index=False
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jm-submission", type=Path, required=True)
    p.add_argument("--jy-submission", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--wrapper", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jm-score", type=float, default=1148.77306)
    p.add_argument("--jy-score", type=float, default=1130.3604943627)
    p.add_argument("--half-score", type=float, default=1142.1381933225)
    p.add_argument("--max-anti", type=float, default=0.50)
    args = p.parse_args()
    for path in (args.jm_submission, args.jy_submission, args.data_dir, args.wrapper):
        if not path.exists(): p.error(f"Missing input: {path}")
    run(Config(**vars(args)))


if __name__ == "__main__": main()
