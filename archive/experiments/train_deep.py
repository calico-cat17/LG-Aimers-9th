"""Train a GPU-friendly multi-task ensemble of failure-risk experts."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.deep_model import ControlMoE
from src.deep_preprocessing import TabularPreprocessor
from src.features import TARGET_COL
from src.label_recovery import recover_failure_labels


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); ps, risks, gates = [], [], []
    for batch in loader:
        numeric, categorical = batch[0], batch[1]
        out = model(numeric.to(device, non_blocking=True), categorical.to(device, non_blocking=True))
        ps.append(out["success"].cpu().numpy())
        risks.append(out["expert_risk"].cpu().numpy())
        gates.append(out["gate"].cpu().numpy())
    return np.concatenate(ps), np.concatenate(risks), np.concatenate(gates)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def fit_affine_calibrator(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit sigmoid(a*logit(p)+b) on the untouched temporal validation season."""
    x = torch.tensor(np.log(np.clip(prob, 1e-6, 1-1e-6) / np.clip(1-prob, 1e-6, 1)), dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.float64)
    ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.2, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(ab[0] * x + ab[1], target)
        loss.backward(); return loss
    opt.step(closure)
    return float(ab[0].detach()), float(ab[1].detach())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--config", default="config/deep_model.json")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--valid-season", type=int, default=2024)
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, torch={torch.__version__}")

    raw = pd.read_csv(args.data, encoding="utf-8-sig", low_memory=False)
    valid_mask = raw["season"].to_numpy() == args.valid_season
    if not valid_mask.any() or valid_mask.all(): raise ValueError("invalid temporal split")
    # Fit all statistics and vocabularies on pre-2024 rows only.
    prep = TabularPreprocessor().fit(raw.loc[~valid_mask])
    numeric, categorical = prep.transform(raw)
    y = raw[TARGET_COL].to_numpy(np.float32)
    subtype, subtype_mask = recover_failure_labels(raw)
    joblib.dump(prep, os.path.join(args.model_dir, "deep_preprocessor.joblib"), compress=3)
    del raw

    n_tensor = torch.from_numpy(numeric)
    c_tensor = torch.from_numpy(categorical)
    y_tensor = torch.from_numpy(y)
    subtype_tensor = torch.from_numpy(subtype)
    subtype_mask_tensor = torch.from_numpy(subtype_mask)
    valid_positions = np.flatnonzero(valid_mask)
    split = int(valid_positions[0])
    if not ((~valid_mask[:split]).all() and valid_mask[split:].all()):
        raise ValueError("rows must be season-sorted for memory-efficient temporal split")
    # Basic slices are zero-copy tensor views. Advanced boolean indexing duplicated
    # several GB on the 1.47M-row table and is intentionally avoided here.
    train_ds = TensorDataset(n_tensor[:split], c_tensor[:split], y_tensor[:split],
                             subtype_tensor[:split], subtype_mask_tensor[:split])
    valid_ds = TensorDataset(n_tensor[split:], c_tensor[split:], y_tensor[split:],
                             subtype_tensor[split:], subtype_mask_tensor[split:])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                              num_workers=0, pin_memory=device.type == "cuda")

    valid_members = []
    model_files = []
    for seed in cfg["seeds"]:
        seed_everything(seed)
        model_args = dict(
            n_numeric=numeric.shape[1], cardinalities=prep.cardinalities,
            width=cfg["width"], depth=cfg["depth"], dropout=cfg["dropout"],
            expert_prior=cfg["expert_prior"],
            numeric_importance=[
                1.5 if (i < len(prep.numeric_cols) and (
                    prep.numeric_cols[i].startswith("asof_pitcher_") or
                    prep.numeric_cols[i] in {"recent_vs_career", "recent_success_mean", "li"}
                )) else 1.0
                for i in range(numeric.shape[1])
            ],
        )
        model = ControlMoE(**model_args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                      weight_decay=cfg["weight_decay"])
        best, best_state, stale = math.inf, None, 0
        for epoch in range(1, cfg["epochs"] + 1):
            model.train(); total = 0.0; started = time.time()
            for nb, cb, yb, sb, mb in train_loader:
                nb, cb, yb = nb.to(device, non_blocking=True), cb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                sb, mb = sb.to(device, non_blocking=True), mb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(nb, cb)
                    primary = ((out["success"] - yb) ** 2).mean()
                    stable_bce = F.binary_cross_entropy(out["success"].float(), yb.float())
                    subtype_row_loss = F.binary_cross_entropy(
                        out["expert_risk"].float(), sb.float(), reduction="none"
                    ).mean(dim=1)
                    subtype_loss = (subtype_row_loss * mb).sum() / mb.sum().clamp_min(1.0)
                    balance = ((out["gate"].mean(0) - 1/3) ** 2).mean()
                    diversity = -out["expert_risk"].var(dim=1).mean()
                    loss = primary + cfg["bce_weight"] * stable_bce + cfg["subtype_weight"] * subtype_loss + cfg["load_balance_weight"] * balance + cfg["diversity_weight"] * diversity
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                total += float(loss.detach()) * len(yb)
            vp, _, _ = predict(model, valid_loader, device)
            vb = brier(y[valid_mask], vp)
            print(f"seed={seed} epoch={epoch} train_loss={total/len(train_ds):.6f} val_brier={vb:.7f} sec={time.time()-started:.1f}")
            if vb < best - 1e-6:
                best, stale = vb, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
                if stale >= cfg["patience"]: break
        model.load_state_dict(best_state)
        filename = f"deep_seed{seed}.pt"
        torch.save({"state_dict": best_state, "model_args": model_args}, os.path.join(args.model_dir, filename))
        model_files.append(filename)
        vp, risks, gates = predict(model, valid_loader, device)
        valid_members.append(vp)
        print("subtype risk means [middle,wild,reverse]:", risks.mean(0), "gate means:", gates.mean(0))

    ensemble = np.mean(valid_members, axis=0)
    a, b = fit_affine_calibrator(ensemble, y[valid_mask])
    calibrated = 1 / (1 + np.exp(-(a * np.log(np.clip(ensemble,1e-6,1-1e-6) / np.clip(1-ensemble,1e-6,1)) + b)))
    print(f"ensemble_brier={brier(y[valid_mask], ensemble):.7f}, calibrated_brier={brier(y[valid_mask], calibrated):.7f}")
    np.savez_compressed(os.path.join(args.model_dir, "validation_deep.npz"),
                        y=y[valid_mask], raw=ensemble, calibrated=calibrated)
    manifest = {"models": model_files, "calibration": {"scale": a, "bias": b},
                "valid_season": args.valid_season, "expert_names": list(ControlMoE.expert_names)}
    with open(os.path.join(args.model_dir, "deep_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
