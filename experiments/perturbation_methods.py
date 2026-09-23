"""One static-method class per perturbation; no instances or mutable class state.

Each class owns its physical definition and evaluation entry point. Shared output
and iteration live in perturbation_shared.py. apply() returns a new point array and
never modifies the reference. All lengths here are physical pixels or microseconds
before conversion into the feature space.
"""

from copy import copy

import numpy as np

if __package__:
    from .perturbation_shared import ReferenceWindow, evaluate_perturbation, read_points
else:
    from perturbation_shared import ReferenceWindow, evaluate_perturbation, read_points


class LocalPerturbation:
    """Shared defaults for transforms that require no additional recorded interval."""

    @staticmethod
    def validate_coverage(trial, real, windows, sweep):
        """Local transforms use the already validated reference windows."""
        return None

    @staticmethod
    def details(window, magnitude, modified):
        """Return optional per-comparison physical metadata."""
        return {}

    @staticmethod
    def example_magnitude(sweep):
        """Use the largest magnitude for the illustrative point cloud."""
        return max(sweep, key=abs)


class SpatialOffset(LocalPerturbation):
    """Translate all events by the same x and y offset, without clipping."""
    name = "spatial_offset"
    title = "Spatial translation"
    parameter = "offset_px"
    x_label = "Offset per axis (px): dx = dy"
    sweep_argument = "spatial_offset_sweep"
    description = "dx = dy = offset_px; total translation sqrt(2)*abs(offset_px); unchanged time/count"

    @staticmethod
    def apply(window: ReferenceWindow, magnitude: float, rng) -> np.ndarray:
        """Add the physical offset to both spatial axes; time is unchanged."""
        points = window.points.copy()
        points[:, 0] += magnitude / window.scales.get("x", 1)
        points[:, 1] += magnitude / window.scales.get("y", 1)
        return points

    @staticmethod
    def details(window, magnitude, modified):
        """Report Euclidean displacement as well as the per-axis sweep value."""
        return {"translation_length_px": float(np.sqrt(2) * abs(magnitude))}

    @staticmethod
    def evaluate(trial, real, args, config, metrics, output_dir):
        """Evaluate, save and plot the full spatial-offset sweep for one trial."""
        return evaluate_perturbation(SpatialOffset, trial, real, args, config, metrics, output_dir)


class SpatialScaling(LocalPerturbation):
    """Scale x and y around the calibrated chopper centre, preserving time/count."""
    name = "spatial_scaling"
    title = "Spatial scaling"
    parameter = "scale_factor"
    x_label = "Spatial scale factor (1 = unchanged)"
    sweep_argument = "spatial_scaling_sweep"
    description = "isotropic spatial scaling around trial.ellipse_centre; no clipping; unchanged time/count"

    @staticmethod
    def apply(window: ReferenceWindow, magnitude: float, rng) -> np.ndarray:
        """Scale in physical geometry, with the centre expressed in feature units."""
        points = window.points.copy()
        scales = np.array([window.scales.get("x", 1), window.scales.get("y", 1)])
        centre = np.asarray(window.trial.ellipse_centre) / scales
        points[:, :2] = centre + magnitude * (points[:, :2] - centre)
        return points

    @staticmethod
    def example_magnitude(sweep):
        """Choose the scale furthest from the identity factor one."""
        return max(sweep, key=lambda value: abs(value - 1))

    @staticmethod
    def evaluate(trial, real, args, config, metrics, output_dir):
        """Evaluate, save and plot the full scaling sweep for one trial."""
        return evaluate_perturbation(SpatialScaling, trial, real, args, config, metrics, output_dir)


class SpatialJitter(LocalPerturbation):
    """Independent bounded uniform x/y displacement per event, in floating-point pixels."""
    name = "spatial_jitter"
    title = "Spatial jitter"
    parameter = "jitter_px"
    x_label = "Maximum uniform jitter per spatial axis (px)"
    sweep_argument = "spatial_jitter_sweep"
    description = "independent per-event dx,dy ~ Uniform(-amplitude,+amplitude); no rounding/clipping; unchanged time/count"

    @staticmethod
    def apply(window: ReferenceWindow, magnitude: float, rng) -> np.ndarray:
        """Retain subpixel jitter and reuse random directions across magnitudes."""
        points = window.points.copy()
        scales = np.array([window.scales.get("x", 1), window.scales.get("y", 1)])
        points[:, :2] += rng.uniform(-1, 1, (len(points), 2)) * magnitude / scales
        return points

    @staticmethod
    def evaluate(trial, real, args, config, metrics, output_dir):
        """Evaluate, save and plot the spatial-jitter sweep for one trial."""
        return evaluate_perturbation(SpatialJitter, trial, real, args, config, metrics, output_dir)


class Subsampling(LocalPerturbation):
    """Randomly retain a fraction of reference events, without replacement."""
    name = "subsampling"
    title = "Subsampling"
    parameter = "retained_fraction"
    x_label = "Fraction of reference events retained"
    sweep_argument = "subsampling_sweep"
    description = "round(fraction*N) distinct reference events; original side remains full; no count equalization"

    @staticmethod
    def apply(window: ReferenceWindow, magnitude: float, rng) -> np.ndarray:
        """Use nested subsets across fractions and retain original event order."""
        count = int(round(magnitude * len(window.points)))
        indices = np.sort(rng.permutation(len(window.points))[:count])
        return window.points[indices].copy()

    @staticmethod
    def details(window, magnitude, modified):
        """Record the realized fraction after integer count rounding."""
        return {"actual_retained_fraction": len(modified) / len(window.points) if len(window.points) else float("nan")}

    @staticmethod
    def example_magnitude(sweep):
        """Show the smallest retained fraction rather than the identity fraction."""
        return min(sweep)

    @staticmethod
    def evaluate(trial, real, args, config, metrics, output_dir):
        """Evaluate, save and plot the subsampling sweep for one trial."""
        return evaluate_perturbation(Subsampling, trial, real, args, config, metrics, output_dir)


class UniformNoise(LocalPerturbation):
    """Add uniform events over the full sensor and the current temporal window."""
    name = "uniform_noise"
    title = "Added uniform noise"
    parameter = "added_noise_ratio"
    x_label = "Added noise events / original event count"
    sweep_argument = "uniform_noise_sweep"
    description = "add round(ratio*N) continuous uniform x/y/t events over full sensor/window; nominal final noise fraction ratio/(1+ratio), not a physical sensor-noise model"

    @staticmethod
    def apply(window: ReferenceWindow, magnitude: float, rng) -> np.ndarray:
        """Preserve original events; appended noise is not masked or count-normalized."""
        count = int(round(magnitude * len(window.points)))
        extent = np.array([window.sensor["width"] / window.scales.get("x", 1),
                           window.sensor["height"] / window.scales.get("y", 1),
                           (window.end_us - window.start_us) / window.scales.get("t", 1)])
        noise = rng.random((count, 3)) * extent
        return np.concatenate([window.points, noise])

    @staticmethod
    def details(window, magnitude, modified):
        """Distinguish added-to-original ratio from actual final mixture fraction."""
        count = len(modified) - len(window.points)
        return {"n_added_noise": count, "actual_noise_fraction": count / len(modified) if len(modified) else float("nan")}

    @staticmethod
    def evaluate(trial, real, args, config, metrics, output_dir):
        """Evaluate, save and plot the uniform-noise sweep for one trial."""
        return evaluate_perturbation(UniformNoise, trial, real, args, config, metrics, output_dir)


class TemporalOffset(LocalPerturbation):
    """Compare recorded windows at shifted chopper phases, without wrapping data."""
    name = "temporal_offset"
    title = "Temporal phase offset"
    parameter = "phase_degrees"
    x_label = "Chopper phase offset (degrees of a full rotation)"
    sweep_argument = "temporal_offset_sweep"
    description = "read [start+delta,end+delta), delta=round(degrees*rotation_period_us/360); re-zero each window separately; no timestamp translation or wrap; 360 degrees compares the next rotation"

    @staticmethod
    def shift_us(degrees: float, rotation_period_us: int) -> int:
        """Convert mechanical rotation degrees to the nearest integer microsecond."""
        return int(round(degrees * rotation_period_us / 360))

    @staticmethod
    def time_sweep_us(rotation_period_us: int, near_us: list[int], step_us: int) -> list[int]:
        """Resolve dense early offsets, regular later offsets and rotation anchors.

        The regular grid starts at the first multiple of step_us strictly beyond
        the largest near offset. All values are integer microseconds in [0,period].
        Quarter rotations are rounded once; the full rotation is always included.
        """
        first_far = (max(near_us) // step_us + 1) * step_us
        near = {value for value in near_us if value <= rotation_period_us}
        far = set(range(first_far, rotation_period_us + 1, step_us))
        anchors = {round(rotation_period_us * fraction) for fraction in (0, .25, .5, .75, 1)}
        return sorted(near | far | anchors)

    @staticmethod
    def resolved_args(trial, args):
        """Copy CLI settings and record the actual per-trial degree/time sweep."""
        resolved = copy(args)
        if getattr(args, "temporal_offset_grid", "degrees") == "time":
            offsets = TemporalOffset.time_sweep_us(
                trial.rotation_period_us, args.temporal_near_us, args.temporal_far_step_us)
            resolved.temporal_offset_sweep = [360 * offset / trial.rotation_period_us for offset in offsets]
        resolved.resolved_temporal_offsets_us = [
            TemporalOffset.shift_us(value, trial.rotation_period_us) for value in resolved.temporal_offset_sweep]
        return resolved

    @staticmethod
    def validate_coverage(trial, real, windows, sweep):
        """Require the same reference windows to fit every requested phase offset."""
        shifts = [TemporalOffset.shift_us(value, trial.rotation_period_us) for value in sweep]
        if windows.window_start_us.min() + min(shifts) < real.t_start:
            raise ValueError("Temporal sweep starts before the recording; increase first_period")
        if windows.window_end_us.max() + max(shifts) > real.t_end:
            raise ValueError("Temporal sweep exceeds recording coverage; reduce periods or phase offsets")

    @staticmethod
    def apply(window: ReferenceWindow, magnitude: float, rng) -> np.ndarray:
        """Read an actual shifted window and subtract its own start timestamp."""
        shift = TemporalOffset.shift_us(magnitude, window.trial.rotation_period_us)
        if shift == 0:
            return window.points.copy()
        return read_points(window.real, window.start_us + shift, window.end_us + shift, window.scales)

    @staticmethod
    def details(window, magnitude, modified):
        """Save integer time displacement and the corresponding realized angle."""
        shift = TemporalOffset.shift_us(magnitude, window.trial.rotation_period_us)
        return {"temporal_offset_us": shift,
                "realized_phase_degrees": 360 * shift / window.trial.rotation_period_us,
                "comparison_start_us": window.start_us + shift,
                "comparison_end_us": window.end_us + shift}

    @staticmethod
    def example_magnitude(sweep):
        """Prefer a visibly displaced 45-degree phase over a complete rotation."""
        nonzero = [value for value in sweep if value != 0]
        return min(nonzero, key=lambda value: abs(value - 45)) if nonzero else 0

    @staticmethod
    def evaluate(trial, real, args, config, metrics, output_dir):
        """Evaluate, save and plot phase-shifted-window responses for one trial."""
        resolved = TemporalOffset.resolved_args(trial, args)
        return evaluate_perturbation(TemporalOffset, trial, real, resolved, config, metrics, output_dir)


METHODS = {method.name: method for method in (
    SpatialOffset, SpatialScaling, SpatialJitter, Subsampling, UniformNoise, TemporalOffset)}
