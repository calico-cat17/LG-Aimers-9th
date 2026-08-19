#!/usr/bin/env bash
# Submission built from the seed-ensembled residual stack. The 36 seed models
# come from model_seed_ensemble; the gate, psych, FiLM and stable-expert assets
# are unchanged and come from model_hierarchical_stack.
set -euo pipefail
output_name="${1:-submissions/260815_seed_ensemble.zip}"
src_dir="model_seed_ensemble"
asset_dir="model_hierarchical_stack"

assets=(
  adaptive_gate.cbm
  pitcher_snapshots.pkl batter_snapshots.pkl pitchmix_snapshots.pkl
  trackman_prior_features.csv
  psych_latent_meta.npz psych_profile.pkl latent_pitch_context.csv
  psych_regime.pt psych_regime_assets.npz psych_regime_profile.pkl
  split_prior_tables.pkl
  stable_context_profile.pkl stable_context_ridge.npz stable_platoon.cbm
)

./trainenv311/bin/python -m py_compile script.py src/preprocessing_v2.py src/split_priors.py

staging_dir="$(mktemp -d /tmp/seed-submission.XXXXXX)"
trap 'rm -rf "$staging_dir"' EXIT
mkdir -p "$staging_dir/model" "$staging_dir/src" "$(dirname "$output_name")"

# seed list comes from the manifest so the archive can never disagree with it
seeds=$(./trainenv311/bin/python -c "import json;m=json.load(open('manifest_seed.json'));print(*m['seeds'])")
sub_seeds=$(./trainenv311/bin/python -c "import json;m=json.load(open('manifest_seed.json'));print(*m.get('subtype_seeds',m['seeds']))")
for s in $seeds; do
  for stem in v2_decay55 v3_decay55 v3_decay30; do
    cp "$src_dir/${stem}_seed${s}.cbm" "$staging_dir/model/"
  done
done
for s in $sub_seeds; do
  for n in middle wild reverse; do
    cp "$src_dir/subtype_${n}_seed${s}.cbm" "$staging_dir/model/"
  done
done
for f in "${assets[@]}"; do cp "$asset_dir/$f" "$staging_dir/model/$f"; done
cp manifest_seed.json "$staging_dir/model/manifest.json"
cp script.py requirements.txt "$staging_dir/"
cp src/__init__.py src/features.py src/season_delta_features.py \
   src/deep_preprocessing.py src/season_history_v3.py src/adaptive_gate.py \
   src/preprocessing_v2.py src/catboost_features.py src/psych_latent.py \
   src/psych_film_expert.py src/context_adjusted_psych.py src/split_priors.py \
   src/group_probe.py src/stable_experts.py src/context_pressure_features.py \
   "$staging_dir/src/"

rm -f "$output_name"
(cd "$staging_dir" && zip -q -r "$OLDPWD/$output_name" model src script.py requirements.txt)
unzip -t "$output_name" >/dev/null && echo "zip ok: $output_name"
du -h "$output_name"
