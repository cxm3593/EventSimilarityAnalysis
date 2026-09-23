"""Supervise selected evaluations with durable logs and process provenance.

Launch this helper in a hidden background process. It does not implement a scheduler
or repeat runs: it starts exactly one sequential suite, waits for its exit code, and
records completion/failure. Existing launch directories are never overwritten.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATED_EVALUATIONS = ["spatial_offset", "spatial_scaling", "spatial_jitter", "temporal_offset"]


def save_manifest(directory: Path, record: dict):
    """Atomically persist process identity and exit status for subsequent checks."""
    temporary = directory / "manifest.tmp"
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(directory / "manifest.json")


def capture_sources(directory: Path) -> dict:
    """Keep the launched code/settings snapshot and hashes beside its logs."""
    source_dir = directory / "source"
    source_dir.mkdir()
    paths = ["experiments/perturbation.py", "experiments/perturbation_shared.py",
             "experiments/perturbation_methods.py", "experiments/common.py",
             "experiments/perturbation_cla.yaml", "experiments/overnight_runner.py", "config.yaml"]
    hashes = {}
    for relative in paths:
        path = PROJECT_ROOT / relative
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main():
    """Start one suite, record child/supervisor PIDs, and preserve both log streams."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", type=Path, required=True)
    parser.add_argument("--evaluations", nargs="+", default=UPDATED_EVALUATIONS,
                        choices=[*UPDATED_EVALUATIONS, "subsampling", "uniform_noise"])
    args = parser.parse_args()
    directory = args.launch_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "experiments/perturbation.py", "--cla",
               "experiments/perturbation_cla.yaml", "--evaluations", *args.evaluations]
    record = {"status": "starting", "started_at": datetime.now().isoformat(),
              "supervisor_pid": os.getpid(), "command": command, "cwd": str(PROJECT_ROOT),
              "stdout": str(directory / "stdout.log"), "stderr": str(directory / "stderr.log"),
              "source_sha256": capture_sources(directory)}
    save_manifest(directory, record)
    try:
        with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, \
                (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr)
            record.update(status="running", child_pid=process.pid)
            save_manifest(directory, record)
            exit_code = process.wait()
        record.update(status="completed" if exit_code == 0 else "failed", exit_code=exit_code,
                      finished_at=datetime.now().isoformat())
    except Exception:
        record.update(status="failed", error=traceback.format_exc(), finished_at=datetime.now().isoformat())
        raise
    finally:
        save_manifest(directory, record)


if __name__ == "__main__":
    main()
