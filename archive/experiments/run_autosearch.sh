#!/usr/bin/env bash
# One process per hypothesis: each starts with a clean heap, so a 4 GB feature
# frame from the previous candidate can never stack on top of the next one.
export PYTHONPATH="/home/joa/Desktop/LG Aimers"
for h in hstate irm_hstate_1 irm_hstate_10 irm_hstate_100 \
         baseline_tree tree_shrunk_90 tree_shrunk_75 tree_lr02 trackman_v2; do
  echo "########## $h  ($(date +%H:%M:%S))"
  timeout 3000 ./trainenv311/bin/python -u autosearch.py --run "$h" 2>&1 | tail -6
done
echo "########## SUMMARY"
./trainenv311/bin/python -u autosearch.py --list
