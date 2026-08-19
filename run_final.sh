#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT/run_final"

if [[ ! -f "$ROOT/data/test.csv" ]]; then
  echo "data/test.csv가 없습니다. 공식 대회 데이터를 저장소의 data/에 배치하세요." >&2
  exit 1
fi

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
unzip -q "$ROOT/final/260818_F_regime075.zip" -d "$RUN_DIR"
cp -a "$ROOT/data" "$RUN_DIR/data"

(
  cd "$RUN_DIR"
  python script.py
)

echo "완료: $RUN_DIR/output/submission.csv"
