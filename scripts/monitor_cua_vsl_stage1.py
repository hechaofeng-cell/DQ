#!/usr/bin/env python3
"""Print a compact status snapshot for the CUA-VSL background stage."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/openimages_v7_cua_vsl_v1"


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def tail(path: Path, count: int = 8) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]


def main() -> None:
    fold_metrics = list((OUTPUT / "oof").glob("*/fold_*/metrics.json"))
    smoke_metrics = list((OUTPUT / "smoke_runs").glob("*/*/*/metrics.json"))
    status = {
        "sessions": {
            "cua_vsl_oof": session_exists("cua_vsl_oof"),
            "cua_vsl_stage1": session_exists("cua_vsl_stage1"),
            "cua_vsl_state_test": session_exists("cua_vsl_state_test"),
            "cua_vsl_smoke": session_exists("cua_vsl_smoke"),
        },
        "oof_folds_complete": len(fold_metrics),
        "oof_folds_planned": 6,
        "state_priority_ready": (OUTPUT / "state_priority/state_priority.csv").is_file(),
        "selection_ready": (OUTPUT / "selections/report.json").is_file(),
        "state_test_status": None,
        "smoke_runs_complete": len(smoke_metrics),
        "smoke_runs_planned": 3,
    }
    state_status = OUTPUT / "state_test_features_auto/pipeline_status.json"
    if state_status.is_file():
        status["state_test_status"] = json.loads(state_status.read_text(encoding="utf-8"))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    for label, path in (
        ("OOF log", OUTPUT / "background/oof_utility.log"),
        ("Follow-up log", OUTPUT / "background/stage1_followup.log"),
        ("Smoke waiter log", OUTPUT / "background/smoke_after_state_test.log"),
    ):
        lines = tail(path)
        if lines:
            print(f"\n{label}:")
            print("\n".join(lines))


if __name__ == "__main__":
    main()
