"""Map anonymized batter IDs and build prior-only Trackman familiarity tables."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)
keys = ["season", "pitcher_id", "game_month", "game_dayofweek"]
main = pd.read_csv("data/train.csv", usecols=["batter_id", "batter_hand", "pitcher_id", "season", "game_month", "game_dayofweek"])
tm = pd.read_csv("data/trackman_history.csv", usecols=[
    "batter_trackman_id", "batter_hand", "pitcher_trackman_id", "season", "game_month", "game_dayofweek", "pitch_type_group",
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
])
for frame in (main, tm): frame["batter_hand"] = frame.batter_hand.replace({"Left": 1, "Right": 2})

# Convert high-confidence Trackman pitchers back to the main anonymous IDs.
pm = pd.read_csv(OUT / "pitcher_trackman_mapping.csv")[["pitcher_id", "pitcher_trackman_id"]]
tm = tm.merge(pm, on="pitcher_trackman_id", how="inner")
main = main.loc[main.pitcher_id.isin(pm.pitcher_id)].copy()

ga = main.groupby(["batter_id", *keys], observed=True).size().rename("n").reset_index()
gb = tm.groupby(["batter_trackman_id", *keys], observed=True).size().rename("n").reset_index()
vocab = pd.concat([ga[keys], gb[keys]], ignore_index=True).drop_duplicates().reset_index(drop=True)
vocab["_key"] = np.arange(len(vocab))
ga = ga.merge(vocab, on=keys); gb = gb.merge(vocab, on=keys)
ai = pd.Index(sorted(ga.batter_id.unique())); bi = pd.Index(sorted(gb.batter_trackman_id.unique()))
A = csr_matrix((ga.n, (ai.get_indexer(ga.batter_id), ga._key)), shape=(len(ai), len(vocab)), dtype=float)
B = csr_matrix((gb.n, (bi.get_indexer(gb.batter_trackman_id), gb._key)), shape=(len(bi), len(vocab)), dtype=float)
an = np.sqrt(A.multiply(A).sum(1)).A1; bn = np.sqrt(B.multiply(B).sum(1)).A1
sim = (A @ B.T).toarray() / np.maximum(an[:, None] * bn[None, :], 1e-12)
ah = main.groupby("batter_id").batter_hand.first().reindex(ai).to_numpy()
bh = tm.groupby("batter_trackman_id").batter_hand.first().reindex(bi).to_numpy()
sim[ah[:, None] != bh[None, :]] = -1
j = sim.argmax(1); rev = sim.argmax(0); ordered = np.sort(sim, axis=1)
confidence = sim[np.arange(len(ai)), j]; margin = ordered[:, -1] - ordered[:, -2]
mutual = rev[j] == np.arange(len(ai))
for conf_min, margin_min in ((0, 0), (.3, .01), (.5, .02), (.7, .02), (.8, .02), (.9, .03)):
    audit = mutual & (confidence >= conf_min) & (margin >= margin_min)
    print("threshold", conf_min, margin_min, "mapped", int(audit.sum()),
          "row coverage", float(main.batter_id.isin(ai[audit]).mean()), flush=True)
ok = mutual & (confidence >= .90) & (margin >= .03)
mapping = pd.DataFrame({
    "batter_id": ai[ok], "batter_trackman_id": bi[j[ok]],
    "mapping_similarity": confidence[ok], "mapping_margin": margin[ok],
})
mapping.to_csv(OUT / "batter_trackman_mapping.csv", index=False)
print("mapped", len(mapping), "row coverage", main.batter_id.isin(mapping.batter_id).mean(), flush=True)

# Batter familiarity, always using seasons strictly earlier than target season.
joined = tm.merge(mapping, on="batter_trackman_id", how="inner")
rows = []
metrics = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]
for rec in mapping.itertuples(index=False):
    player = joined.loc[joined.batter_id.eq(rec.batter_id)]
    for target in range(2019, 2026):
        hist = player.loc[player.season.lt(target)]
        row = {"batter_id": str(rec.batter_id), "season": target,
               "btm_n": len(hist), "btm_mapping_similarity": rec.mapping_similarity}
        rates = hist.pitch_type_group.value_counts(normalize=True)
        for pitch in ("fastball", "breaking", "offspeed", "other"):
            sub = hist.loc[hist.pitch_type_group.eq(pitch)]
            row[f"btm_{pitch}_exposure"] = rates.get(pitch, np.nan)
            for metric in metrics:
                row[f"btm_{pitch}_{metric}_mean"] = sub[metric].mean()
                row[f"btm_{pitch}_{metric}_std"] = sub[metric].std()
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "batter_trackman_familiarity.csv", index=False)
print("saved familiarity", len(rows), flush=True)
