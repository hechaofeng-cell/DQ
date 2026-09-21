#!/usr/bin/env python3
"""Write a lightweight heartbeat for the existing four-model training process."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/openimages_v7_dog_gap_experiment_v1.json"
OUTPUT = ROOT / "artifacts/openimages_v7_four_model_experiment_v1"
RUNNER = str(ROOT / "scripts/run_openimages_v7_four_model_experiment.py")


def runner_pids() -> list[str]:
    found = subprocess.run(
        ["pgrep", "-f", f"^[^ ]*python[^ ]* -u {RUNNER}$"],
        capture_output=True, text=True, check=False,
    )
    return found.stdout.split() if found.returncode == 0 else []


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    ordered = [
        (model, condition, seed)
        for model in config["models"]
        for condition in config["conditions"]
        for seed in config["training_seeds"]
    ]
    path = OUTPUT / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as log:
        while True:
            completed = [
                (model, condition, seed)
                for model, condition, seed in ordered
                if (OUTPUT / "runs" / model / condition / str(seed) / "metrics.json").is_file()
            ]
            pending = next((run for run in ordered if run not in completed), None)
            pids = runner_pids()
            report_ready = (OUTPUT / "report.md").is_file()
            if report_ready:
                state = "report_ready"
            elif pids:
                state = "training"
            else:
                state = "runner_not_detected"
            current = "/".join(map(str, pending)) if pending else "none"
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            log.write(f"{stamp} state={state} completed={len(completed)}/{len(ordered)} "
                      f"next={current} runner_pid={pids[0] if pids else '-'}\n")
            if report_ready or not pids:
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
