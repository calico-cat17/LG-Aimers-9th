#!/usr/bin/env bash
set -euo pipefail

output_name="${1:-submissions/260814_split_priors.zip}"
model_dir="model_hierarchical_stack"
keep_staging="${KEEP_STAGING:-}"

required_model_files=(
  manifest.json adaptive_gate.cbm
  v2_decay55.cbm v3_decay55.cbm v3_decay30.cbm
  subtype_middle.cbm subtype_wild.cbm subtype_reverse.cbm
  pitcher_snapshots.pkl batter_snapshots.pkl pitchmix_snapshots.pkl
  trackman_prior_features.csv
  psych_latent_meta.npz psych_profile.pkl latent_pitch_context.csv
  psych_regime.pt psych_regime_assets.npz psych_regime_profile.pkl
  split_prior_tables.pkl
  stable_context_profile.pkl stable_context_ridge.npz stable_platoon.cbm
)

for filename in "${required_model_files[@]}"; do
  if [[ ! -f "$model_dir/$filename" ]]; then
    echo "missing model file: $model_dir/$filename" >&2
    exit 1
  fi
done

./trainenv311/bin/python -m py_compile script.py src/adaptive_gate.py \
  src/preprocessing_v2.py src/split_priors.py src/group_probe.py src/stable_experts.py

staging_dir="${STAGING_DIR:-$(mktemp -d /tmp/adaptive-submission.XXXXXX)}"
if [[ -z "$keep_staging" ]]; then trap 'rm -rf "$staging_dir"' EXIT; fi
mkdir -p "$staging_dir/model" "$staging_dir/src" "$staging_dir/data" "$(dirname "$output_name")"

for filename in "${required_model_files[@]}"; do
  cp "$model_dir/$filename" "$staging_dir/model/$filename"
done
cp script.py requirements.txt "$staging_dir/"
cp src/__init__.py src/features.py src/season_delta_features.py \
   src/deep_preprocessing.py src/season_history_v3.py src/adaptive_gate.py \
   src/preprocessing_v2.py src/catboost_features.py "$staging_dir/src/"
cp src/psych_latent.py "$staging_dir/src/"
cp src/psych_film_expert.py src/context_adjusted_psych.py "$staging_dir/src/"
cp src/split_priors.py src/group_probe.py "$staging_dir/src/"
cp src/stable_experts.py src/context_pressure_features.py "$staging_dir/src/"

rm -f "$output_name"
(cd "$staging_dir" && zip -q -r "$OLDPWD/$output_name" model src script.py requirements.txt)
unzip -t "$output_name" >/dev/null && echo "zip ok: $output_name"
du -h "$output_name"
[[ -n "$keep_staging" ]] && echo "staging kept at $staging_dir"
exit 0
