"""Static guard for the organiser's row-independence and data-source rules."""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path


FORBIDDEN_SOURCE = {
    "test aggregation": r"\b(groupby|rolling|expanding|value_counts|rank)\s*\(",
    "test distribution": r"\b(test|df|raw)\s*\[[^\n]+\]\s*\.\s*(mean|median|std|quantile)\s*\(",
    "sample/group adjustment logic": r"\b(group_probe|row_probe|sample_probe)\b",
}
# The organiser explicitly allows leaderboard-based selection/interpolation of
# global hyperparameters and ensemble weights. Per-sample/group adjustments are
# different: those can encode evaluation information for selected rows.
FORBIDDEN_META = {"group_probe", "row_probe", "sample_probe"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    args = ap.parse_args()
    archive = Path(args.zip_path)
    findings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="submission-audit-") as tmp:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        root = Path(tmp)
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            # Training-only helper modules may contain groupby. The executable
            # inference script is the authoritative place for static blocking;
            # the dynamic checker catches indirect calls from helpers.
            if path.name != "script.py":
                continue
            for label, pattern in FORBIDDEN_SOURCE.items():
                if re.search(pattern, text, flags=re.I):
                    findings.append(f"{path.relative_to(root)}: {label}")
        manifest = root / "model/manifest.json"
        if manifest.exists():
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            for key in sorted(FORBIDDEN_META & set(meta)):
                findings.append(f"model/manifest.json: forbidden key {key!r}")

    print(f"=== 제출 규정 정적 감사: {archive.name} ===")
    if findings:
        for item in findings:
            print(f"FAIL  {item}")
        return 1
    print("PASS  금지된 test 집계·분포 보정·특정 샘플/그룹 조정 코드가 없습니다.")
    print("NEXT  check_row_independence.py로 실행 시 행 독립성도 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
