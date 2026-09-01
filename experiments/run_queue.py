"""Overnight queue: the whole-period ruler on every trial, then the null test.

One stage at a time, one process each, so a stage that fails leaves the stages before
it intact and the next one still gets attempted. Every stage writes its own run folder
with its own run_config.yaml.

    python experiments/run_queue.py
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

TRIALS = [f"optical_chopper_data_f{i}" for i in range(1, 6)]


def say(message: str):
    print(f"=== [{datetime.now():%Y-%m-%d %H:%M:%S}] {message} ===", flush=True)


def newest(trial: str, test: str, suffix: str) -> str:
    """The most recent non-smoke run folder for a trial and test, or "" if none.

    Empty rather than missing so a stage still runs when the folder it wanted to point
    at is not there; the scripts treat an unusable --*-from as "draw it without that".
    """
    root = HERE.parent / "output" / "tests" / trial / test
    if not root.is_dir():
        return ""
    folders = [p for p in root.glob(f"*{suffix}")
               if p.is_dir() and "smoke" not in p.name]
    if not folders:
        return ""
    return str(max(folders, key=lambda p: p.stat().st_mtime))


def pointer(flag: str, folder: str) -> list:
    """[flag, folder], or nothing at all when there is no folder to point at."""
    return [flag, folder] if folder else []


def stage(name: str, arguments: list, log_name: str) -> int:
    say(f"{name}: starting")
    started = time.time()
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / log_name, "w", encoding="utf-8") as handle:
        code = subprocess.call([str(PYTHON), "-u"] + arguments, cwd=str(HERE),
                               stdout=handle, stderr=subprocess.STDOUT)
    say(f"{name}: exit {code} after {(time.time() - started) / 60:.1f} min")
    return code


def main():
    # Name stages on the command line to run only those, e.g.
    #     python experiments/run_queue.py test34
    # With no arguments the whole sequence runs in order.
    only = set(sys.argv[1:]) or {"test2", "test1", "test34"}

    # Test 2, whole-rotation span compared window by window, no event cap.
    # The sweep stops at one rotation rather than the spec's two, which keeps the
    # coarse step at the spec's 20 ms across the whole range instead of halving the
    # resolution to reach a second rotation that only repeats the first. One rotation
    # also ends exactly where the shifted span realigns with the reference.
    for trial in TRIALS if "test2" in only else []:
        stage(f"test 2 ruler, whole period, {trial}",
              ["test2_ruler.py", "--trial", trial,
               "--span-us", "period", "--window-us", "5000",
               "--n-baseline-windows", "1",
               "--far-step-us", "20000", "--max-shift-periods", "1.0"],
              f"test2_period_{trial}.log")

    # Test 1, uncapped, on every trial. Keep this as the full set so a corrected
    # implementation can be rerun consistently rather than silently reusing f1.
    for trial in TRIALS if "test1" in only else []:
        stage(f"test 1 null, {trial}",
              ["test1_null.py", "--trial", trial],
              f"test1_{trial}.log")

    # Tests 3 and 4, uncapped. These stay on short windows: "whole period" was about
    # the ruler's span, not about what a metric is handed. Test 4 is pointed at this
    # run's own test 1 and test 2 folders so its placement figures invert the right
    # curves rather than an older capped run's.
    for trial in TRIALS if "test34" in only else []:
        floor = pointer("--floor-from", newest(trial, "test1_null", ""))

        # One seed and one rotation. Periods only add averaging in test 3, and the
        # 40 windows spread across a rotation already provide it; the cut is what
        # makes five recordings fit a night.
        for phase in ("3a", "3b"):
            stage(f"test 3 {phase}, {trial}",
                  ["test3_modifier.py", "--trial", trial, "--phase", phase,
                   "--window-us", "5000", "--seeds", "0", "--periods", "1"] + floor,
                  f"test3{phase}_{trial}.log")

        # Resolved after 3a has run, so it finds the folder that stage just wrote.
        extra = (floor
                 + pointer("--ruler-from",
                           newest(trial, "test2_ruler", "period_w5000us"))
                 + pointer("--sweeps-from",
                           newest(trial, "test3_modifier", "3a_w5000us")))
        # Three rotations, set explicitly. The default is every whole rotation in the
        # recording, which is 7 on f1 but 37 on f5, and 4c's matrix is the periods
        # squared - that default would build a 74 x 74 matrix on the fastest wheel.
        stage(f"test 4 synthetic, {trial}",
              ["test4_synthetic.py", "--trial", trial, "--window-us", "5000",
               "--periods", "3", "--polarity", "drop"] + extra,
              f"test4_{trial}.log")

    say("queue finished")


if __name__ == "__main__":
    main()
