"""Overnight driver: waits for test 1, then runs tests 2, 3a, 3b and 4 in order.

Each stage is its own process writing its own run folder, so a stage that fails
leaves every stage before it intact and the next one still gets attempted.

    python experiments/run_all.py <test1_run_folder> [window_us]
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent / ".venv" / "Scripts" / "python.exe"
LOGS = HERE.parent / "output" / "logs"
TESTS = HERE.parent / "output" / "tests" / "optical_chopper_data_f1"


def say(message: str):
    print(f"=== [{datetime.now():%H:%M:%S}] {message} ===", flush=True)


def newest(pattern: str):
    matches = sorted(TESTS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0]) if matches else None


def stage(name: str, arguments: list, log_name: str) -> int:
    say(f"{name}: starting")
    started = time.time()
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / log_name, "w") as handle:
        code = subprocess.call([str(PYTHON), "-u"] + arguments, cwd=str(HERE),
                               stdout=handle, stderr=subprocess.STDOUT)
    say(f"{name}: exit {code} after {time.time() - started:.0f}s")
    return code


def main():
    test1 = Path(sys.argv[1])
    window = sys.argv[2] if len(sys.argv) > 2 else "5000"

    say(f"waiting for {test1 / 'summary.csv'}")
    for _ in range(360):                       # up to three hours
        if (test1 / "summary.csv").exists():
            break
        time.sleep(30)
    floor = ["--floor-from", str(test1)] if (test1 / "summary.csv").exists() else []
    say("test 1 ready" if floor else "test 1 summary never appeared; no chance band")

    stage("test 2 ruler",
          ["test2_ruler.py", "--window-us", window] + floor,
          f"test2_w{window}.log")
    ruler = newest(f"test2_ruler/*w{window}us")

    stage("test 3a sweeps",
          ["test3_modifier.py", "--phase", "3a", "--window-us", window] + floor,
          f"test3a_w{window}.log")
    sweeps = newest(f"test3_modifier/*3a_w{window}us")

    stage("test 3b directed",
          ["test3_modifier.py", "--phase", "3b", "--window-us", window] + floor,
          f"test3b_w{window}.log")

    extra = []
    if ruler:
        extra += ["--ruler-from", ruler]
    if sweeps:
        extra += ["--sweeps-from", sweeps]
    stage("test 4 synthetic",
          ["test4_synthetic.py", "--window-us", window] + floor + extra,
          f"test4_w{window}.log")

    say("chain finished")


if __name__ == "__main__":
    main()
