"""Generate V2CE events and convert them to the calibrated project event schema.

The upstream model resizes a 1280 x 720 video to roughly 462 x 260 in ``pano`` mode.
This script maps event pixel centres back to the original warped-video image plane,
applies the same calibrated ellipse mask used for real and v2e, and writes a
time-sorted HDF5 ``events`` dataset with the project's exact field types.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml


WORKSPACE = Path(r"C:/Users/cxm3593/Academic/Workspace")
PROJECT_ROOT = WORKSPACE / "EventSimilarityAnalysis"
DEFAULT_V2CE_ROOT = WORKSPACE / "V2CE-Toolbox"
DEFAULT_TRIALS_ROOT = WORKSPACE / "EventCamCalib/output/trials"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output/v2ce"
TRIAL_NAMES = tuple(f"optical_chopper_data_f{i}" for i in range(1, 6))

EVENT_DTYPE = np.dtype([
    ("x", "<u2"),
    ("y", "<u2"),
    ("p", "<i2"),
    ("t", "<i8"),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--trial", choices=TRIAL_NAMES)
    selected.add_argument("--all", action="store_true", help="generate F1 through F5")
    parser.add_argument("--v2ce-root", type=Path, default=DEFAULT_V2CE_ROOT)
    parser.add_argument("--trials-root", type=Path, default=DEFAULT_TRIALS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="limit frames for a smoke test; default uses the whole video")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stage2-batch-size", type=int, default=4)
    parser.add_argument("--skip-generation", action="store_true",
                        help="convert an already generated NPZ")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    record = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    if record["width"] <= 0 or record["height"] <= 0 or record["frame_count"] <= 1:
        raise RuntimeError(f"invalid video metadata: {record}")
    return record


def upstream_output_path(output_dir: Path, video_path: Path, fps: int,
                         suffix: str) -> Path:
    return output_dir / f"{video_path.stem}-ceil_10-fps_{fps}-{suffix}-events.npz"


def run_upstream(args: argparse.Namespace, trial: str, video_path: Path,
                 output_dir: Path, model_path: Path, metadata: dict) -> tuple[Path, list[str]]:
    fps = int(round(metadata["fps"]))
    if not np.isclose(metadata["fps"], fps, atol=1e-3):
        raise ValueError(f"V2CE accepts integer FPS, but video reports {metadata['fps']}")
    max_frames = args.max_frames or metadata["frame_count"]
    suffix = f"{trial}-pano"
    expected = upstream_output_path(output_dir, video_path, fps, suffix)
    if expected.exists() and not args.overwrite:
        print(f"using existing upstream output: {expected}")
        return expected, []

    command = [
        sys.executable,
        "v2ce.py",
        "--out_name_suffix", suffix,
        "--max_frame_num", str(max_frames),
        "--infer_type", "pano",
        "--input_video_path", str(video_path),
        "--out_folder", str(output_dir),
        "--model_path", str(model_path),
        "--fps", str(fps),
        "--batch_size", str(args.batch_size),
        "--stage2_batch_size", str(args.stage2_batch_size),
        "--write_event_frame_video", "false",
        "--log_level", "info",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "generation.log"
    print(f"running V2CE for {trial}: {max_frames} frames")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=args.v2ce_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    if not expected.exists():
        matches = sorted(output_dir.glob("*-events.npz"), key=lambda p: p.stat().st_mtime)
        if len(matches) != 1:
            raise FileNotFoundError(f"expected {expected}; candidates: {matches}")
        expected = matches[-1]
    return expected, command


def ellipse_record(trial_dir: Path) -> tuple[dict, float]:
    with (trial_dir / "result.yaml").open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    ellipse = result["spatial_fine_calibration"]["v2e_ellipse"]
    margin = float(result.get("masked_events", {}).get("mask_margin", 1.0))
    return ellipse, margin


def inside_ellipse(x: np.ndarray, y: np.ndarray, ellipse: dict,
                   margin: float) -> np.ndarray:
    cx, cy = map(float, ellipse["center"])
    d1, d2 = map(float, ellipse["axes"])
    theta = np.deg2rad(float(ellipse["angle_deg"]))
    dx = x.astype(np.float64) - cx
    dy = y.astype(np.float64) - cy
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    a = d1 * 0.5 * margin
    b = d2 * 0.5 * margin
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def convert(npz_path: Path, destination: Path, metadata: dict,
            ellipse: dict, margin: float) -> dict:
    with np.load(npz_path, allow_pickle=False) as archive:
        if "event_stream" not in archive:
            raise KeyError(f"event_stream missing from {npz_path}: {archive.files}")
        source = archive["event_stream"]

    required = {"timestamp", "x", "y", "polarity"}
    names = set(source.dtype.names or ())
    if not required.issubset(names):
        raise TypeError(f"unexpected V2CE event fields {source.dtype.names}")

    resized_height = 260
    resized_width = int(metadata["width"] / metadata["height"] * resized_height)
    x = ((source["x"].astype(np.float64) + 0.5)
         * metadata["width"] / resized_width - 0.5).astype(np.float32)
    y = ((source["y"].astype(np.float64) + 0.5)
         * metadata["height"] / resized_height - 0.5).astype(np.float32)
    t = source["timestamp"].astype(np.int64, copy=False)
    p = source["polarity"].astype(np.int8, copy=False)

    if len(t) and np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        x, y, p, t = x[order], y[order], p[order], t[order]
        was_sorted = False
    else:
        was_sorted = True

    valid = ((x >= 0) & (x < metadata["width"]) &
             (y >= 0) & (y < metadata["height"]) &
             inside_ellipse(x, y, ellipse, margin))
    converted = np.empty(int(valid.sum()), dtype=EVENT_DTYPE)
    # Match final_masked_real.h5/final_masked_v2e.h5 exactly. The inverse map is
    # evaluated at pixel centres and rounded only at this final schema boundary.
    converted["x"] = np.rint(x[valid]).astype(np.uint16)
    converted["y"] = np.rint(y[valid]).astype(np.uint16)
    converted["p"] = p[valid]
    converted["t"] = t[valid]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as handle:
        dataset = handle.create_dataset(
            "events", data=converted, compression="gzip", chunks=True,
        )
        dataset.attrs["coordinate_space"] = "frame_warped_1280x720"
        dataset.attrs["source_format"] = "V2CE event_stream NPZ"
        dataset.attrs["v2ce_resized_width"] = resized_width
        dataset.attrs["v2ce_resized_height"] = resized_height
        dataset.attrs["x_inverse_scale"] = metadata["width"] / resized_width
        dataset.attrs["y_inverse_scale"] = metadata["height"] / resized_height
    temporary.replace(destination)

    return {
        "events_before_mask": int(len(source)),
        "events_after_mask": int(len(converted)),
        "events_dropped": int(len(source) - len(converted)),
        "first_t_us": int(converted["t"][0]) if len(converted) else None,
        "last_t_us": int(converted["t"][-1]) if len(converted) else None,
        "duration_us": (int(converted["t"][-1] - converted["t"][0])
                        if len(converted) else None),
        "input_was_time_sorted": was_sorted,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "x_inverse_scale": metadata["width"] / resized_width,
        "y_inverse_scale": metadata["height"] / resized_height,
    }


def write_diagnostic(events_path: Path, video_path: Path, ellipse: dict,
                     output_path: Path, start_us: int = 500_000,
                     window_us: int = 33_333) -> None:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(start_us / 1e6 * fps))))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return
    with h5py.File(events_path, "r") as handle:
        events = handle["events"]
        times = events["t"][:]
        low, high = np.searchsorted(times, [start_us, start_us + window_us])
        sample = events[int(low):int(high)]
    if len(sample) > 100_000:
        sample = sample[np.linspace(0, len(sample) - 1, 100_000, dtype=int)]
    for polarity, colour in ((0, (255, 80, 40)), (1, (40, 60, 255))):
        block = sample[sample["p"] == polarity]
        xi = np.clip(np.rint(block["x"]).astype(int), 0, frame.shape[1] - 1)
        yi = np.clip(np.rint(block["y"]).astype(int), 0, frame.shape[0] - 1)
        frame[yi, xi] = colour
    center = tuple(int(round(v)) for v in ellipse["center"])
    axes = tuple(int(round(v / 2)) for v in ellipse["axes"])
    cv2.ellipse(frame, center, axes, float(ellipse["angle_deg"]), 0, 360,
                (0, 255, 255), 2)
    cv2.imwrite(str(output_path), frame)


def generate_trial(args: argparse.Namespace, trial: str) -> None:
    trial_dir = args.trials_root / trial
    video_path = trial_dir / "frame_warped.avi"
    model_path = args.model or args.v2ce_root / "weights/v2ce_3d.pt"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    metadata = video_metadata(video_path)
    output_dir = args.output_root / trial
    raw_dir = output_dir / "raw"
    ellipse, margin = ellipse_record(trial_dir)
    fps = int(round(metadata["fps"]))
    suffix = f"{trial}-pano"
    npz_path = upstream_output_path(raw_dir, video_path, fps, suffix)
    command: list[str] = []
    if not args.skip_generation:
        npz_path, command = run_upstream(
            args, trial, video_path, raw_dir, model_path, metadata,
        )
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    final_path = output_dir / "final_masked_v2ce.h5"
    if final_path.exists() and not args.overwrite:
        print(f"using existing converted output: {final_path}")
        conversion = {"status": "existing"}
    else:
        conversion = convert(npz_path, final_path, metadata, ellipse, margin)
        write_diagnostic(final_path, video_path, ellipse,
                         output_dir / "diagnostic_alignment.png")

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.v2ce_root, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "trial": trial,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_video": str(video_path),
        "video": metadata,
        "v2ce_root": str(args.v2ce_root),
        "v2ce_commit": commit,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "infer_type": "pano",
        "fps_argument": fps,
        "max_frames": args.max_frames or metadata["frame_count"],
        "command": command,
        "upstream_npz": str(npz_path),
        "output_h5": str(final_path),
        "output_schema": {"dataset": "events", "fields": ["x", "y", "p", "t"]},
        "ellipse_mask": {"ellipse": ellipse, "margin": margin},
        "conversion": conversion,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(json.loads(json.dumps(manifest)), handle, sort_keys=False)
    print(f"wrote {final_path}")
    print(f"wrote {output_dir / 'run_config.yaml'}")


def main() -> None:
    args = parse_args()
    if not args.v2ce_root.exists():
        raise FileNotFoundError(args.v2ce_root)
    trials = TRIAL_NAMES if args.all else (args.trial,)
    for trial in trials:
        print(f"\n=== {trial} ===")
        generate_trial(args, trial)


if __name__ == "__main__":
    main()
