"""Data modifiers applied to event windows before feature scaling.

Modifiers operate on structured event arrays in their original physical units
(``x``/``y`` in pixels, ``t`` in microseconds), after windowing but before
:func:`event_analysis_toolbox.feature_preprocessing.window_features`. This keeps
modifier parameters physically meaningful and comparable across windows.

Modifiers are registered by name so ``config.yaml`` can declare an ordered
sequence of steps, mirroring the metric registry in ``metrics.py``.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


POLARITY_FIELDS = {"p", "polarity", "pol"}
_DEFAULT_SPATIAL_FIELDS = ("x", "y", "t")


@dataclass
class ModifierContext:
    """Per-window context passed to each modifier's ``apply``."""

    rng: np.random.Generator
    window_start: int
    window_end: int
    sensor: dict | None = None


class BaseModifier(ABC):
    type_name: ClassVar[str]

    def __init__(self, **params: Any):
        self._params = params

    @abstractmethod
    def apply(self, events: np.ndarray, context: ModifierContext) -> np.ndarray:
        """Return a modified copy of ``events`` (never mutates the input)."""

    def describe(self) -> dict[str, Any]:
        return {"type": self.type_name, **self._params}


_MODIFIERS: dict[str, type[BaseModifier]] = {}


def register_modifier(modifier_cls: type[BaseModifier]) -> type[BaseModifier]:
    """Class decorator registering a modifier under its ``type_name``."""
    if modifier_cls.type_name in _MODIFIERS:
        raise ValueError(f"Modifier {modifier_cls.type_name!r} is already registered.")
    _MODIFIERS[modifier_cls.type_name] = modifier_cls
    return modifier_cls


def build_modifier(spec: dict[str, Any]) -> BaseModifier:
    """Instantiate a modifier from a ``{type: ..., **params}`` mapping."""
    spec = dict(spec)
    try:
        type_name = spec.pop("type")
    except KeyError as exc:
        raise ValueError("Each modifier step must define a 'type'.") from exc

    if type_name not in _MODIFIERS:
        raise ValueError(
            f"Unsupported modifier: {type_name!r}. Supported: {sorted(_MODIFIERS)}"
        )
    return _MODIFIERS[type_name](**spec)


@dataclass
class ModifierPipeline:
    """An ordered, named sequence of modifiers."""

    name: str
    modifiers: list[BaseModifier] = field(default_factory=list)

    def apply(self, events: np.ndarray, context: ModifierContext) -> np.ndarray:
        for modifier in self.modifiers:
            events = modifier.apply(events, context)
        return events

    def describe(self) -> list[dict[str, Any]]:
        return [modifier.describe() for modifier in self.modifiers]


def _expand_sweep(name: str, sweep_spec: dict[str, Any]) -> list[ModifierPipeline]:
    """Expand one modifier spec with list-valued params into separate curves.

    Each list-valued parameter is swept (cartesian product across multiple
    lists), producing one single-step pipeline per combination. Curve names
    encode the swept values so they read clearly in the legend.
    """
    spec = dict(sweep_spec)
    swept_keys = [key for key, value in spec.items() if key != "type" and isinstance(value, list)]
    if not swept_keys:
        return [ModifierPipeline(name=name, modifiers=[build_modifier(spec)])]

    pipelines: list[ModifierPipeline] = []
    for combo in itertools.product(*(spec[key] for key in swept_keys)):
        concrete = dict(spec)
        concrete.update(dict(zip(swept_keys, combo)))
        label = ", ".join(f"{key}={value}" for key, value in zip(swept_keys, combo))
        pipelines.append(
            ModifierPipeline(name=f"{name} ({label})", modifiers=[build_modifier(concrete)])
        )
    return pipelines


def build_pipelines(modifiers_config) -> list[ModifierPipeline]:
    """Build named pipelines from the ``modifiers`` config section.

    Each entry must define a ``name`` and either:
      * ``steps``: an ordered list of modifiers composed into a single curve, or
      * ``sweep``: a single modifier spec whose list-valued parameters are
        expanded into one separate curve per value.
    """
    pipelines: list[ModifierPipeline] = []
    seen: set[str] = set()
    for entry in modifiers_config or []:
        name = entry.get("name")
        if not name:
            raise ValueError("Each modifier pipeline must define a 'name'.")
        if "sweep" in entry and "steps" in entry:
            raise ValueError(f"Modifier {name!r} must define either 'steps' or 'sweep', not both.")

        if "sweep" in entry:
            built = _expand_sweep(name, entry["sweep"])
        else:
            steps = [build_modifier(step) for step in entry.get("steps", [])]
            built = [ModifierPipeline(name=name, modifiers=steps)]

        for pipeline in built:
            if pipeline.name in seen:
                raise ValueError(f"Duplicate modifier pipeline name: {pipeline.name!r}.")
            seen.add(pipeline.name)
            pipelines.append(pipeline)
    return pipelines


def _field_lookup(dtype, target: str) -> str | None:
    """Return the actual field name matching ``target`` case-insensitively."""
    if dtype.names is None:
        return None
    target = target.lower()
    for name in dtype.names:
        if name.lower() == target:
            return name
    return None


def _resolve_range_map(range_value, dtype) -> dict[str, float]:
    """Normalize a scalar or per-axis range into a ``{field: value}`` mapping."""
    if isinstance(range_value, dict):
        return {key: float(value) for key, value in range_value.items()}
    scalar = float(range_value)
    fields = [name for name in _DEFAULT_SPATIAL_FIELDS if _field_lookup(dtype, name)]
    return {name: scalar for name in fields}


@register_modifier
class AddNoise(BaseModifier):
    """Add uniformly distributed noise events to the window."""

    type_name = "add_noise"

    def __init__(self, count: int, distribution: str = "uniform"):
        super().__init__(count=int(count), distribution=distribution)
        self.count = int(count)
        self.distribution = distribution
        if self.count < 0:
            raise ValueError("add_noise count must be non-negative.")
        if distribution != "uniform":
            raise ValueError("add_noise currently supports distribution='uniform' only.")

    def apply(self, events: np.ndarray, context: ModifierContext) -> np.ndarray:
        if self.count == 0 or events.dtype.names is None:
            return np.array(events, copy=True)

        sensor = context.sensor or {}
        if "width" not in sensor or "height" not in sensor:
            raise ValueError(
                "add_noise requires sensor.width and sensor.height in config.yaml."
            )

        rng = context.rng
        noise = np.empty(self.count, dtype=events.dtype)
        for name in events.dtype.names:
            lowered = name.lower()
            if lowered == "x":
                noise[name] = rng.uniform(0, float(sensor["width"]), self.count)
            elif lowered == "y":
                noise[name] = rng.uniform(0, float(sensor["height"]), self.count)
            elif lowered == "t":
                noise[name] = rng.uniform(context.window_start, context.window_end, self.count)
            elif lowered in POLARITY_FIELDS:
                choices = np.unique(events[name]) if len(events) else np.array([1])
                noise[name] = rng.choice(choices, self.count)
            else:
                if len(events):
                    low, high = float(events[name].min()), float(events[name].max())
                else:
                    low, high = 0.0, 1.0
                noise[name] = rng.uniform(low, high, self.count)

        combined = np.concatenate([events, noise])
        if _field_lookup(combined.dtype, "t"):
            combined.sort(order=_field_lookup(combined.dtype, "t"))
        return combined


@register_modifier
class Subsample(BaseModifier):
    """Keep a fraction of events (``random`` or evenly spaced ``uniform``)."""

    type_name = "subsample"

    def __init__(self, ratio: float, method: str = "random"):
        super().__init__(ratio=float(ratio), method=method)
        self.ratio = float(ratio)
        self.method = method
        if not 0 < self.ratio <= 1:
            raise ValueError("subsample ratio must be in (0, 1].")
        if method not in {"random", "uniform"}:
            raise ValueError("subsample method must be 'random' or 'uniform'.")

    def apply(self, events: np.ndarray, context: ModifierContext) -> np.ndarray:
        n = len(events)
        if n == 0:
            return np.array(events, copy=True)

        keep = min(n, max(1, int(round(self.ratio * n))))
        if self.method == "random":
            indices = np.sort(context.rng.choice(n, size=keep, replace=False))
        else:
            indices = np.unique(np.linspace(0, n - 1, keep).astype(int))
        return np.array(events[indices], copy=True)


@register_modifier
class Jitter(BaseModifier):
    """Add uniform jitter within ``range`` to a fraction of selected events."""

    type_name = "jitter"

    def __init__(self, range, selection_ratio: float = 1.0):  # noqa: A002 - config key
        super().__init__(range=range, selection_ratio=float(selection_ratio))
        self.range = range
        self.selection_ratio = float(selection_ratio)
        if not 0 <= self.selection_ratio <= 1:
            raise ValueError("jitter selection_ratio must be in [0, 1].")

    def apply(self, events: np.ndarray, context: ModifierContext) -> np.ndarray:
        out = np.array(events, copy=True)
        n = len(out)
        if n == 0 or out.dtype.names is None:
            return out

        count = int(round(self.selection_ratio * n))
        if count == 0:
            return out

        rng = context.rng
        selected = rng.choice(n, size=count, replace=False)
        for axis, amplitude in _resolve_range_map(self.range, out.dtype).items():
            field_name = _field_lookup(out.dtype, axis)
            if field_name is None or amplitude == 0:
                continue
            values = out[field_name].astype(np.float64)
            values[selected] += rng.uniform(-amplitude, amplitude, size=count)
            out[field_name] = values
        return out


@register_modifier
class Transform(BaseModifier):
    """Add a constant per-axis offset to every event."""

    type_name = "transform"

    def __init__(self, offset: dict):
        super().__init__(offset=offset)
        self.offset = offset

    def apply(self, events: np.ndarray, context: ModifierContext) -> np.ndarray:
        out = np.array(events, copy=True)
        if out.dtype.names is None:
            return out
        for axis, value in self.offset.items():
            field_name = _field_lookup(out.dtype, axis)
            if field_name is None:
                continue
            out[field_name] = out[field_name] + value
        return out


@register_modifier
class Scaling(BaseModifier):
    """Multiply every event by a constant per-axis scale factor."""

    type_name = "scaling"

    def __init__(self, scale: dict):
        super().__init__(scale=scale)
        self.scale = scale

    def apply(self, events: np.ndarray, context: ModifierContext) -> np.ndarray:
        out = np.array(events, copy=True)
        if out.dtype.names is None:
            return out
        for axis, value in self.scale.items():
            field_name = _field_lookup(out.dtype, axis)
            if field_name is None:
                continue
            out[field_name] = out[field_name] * value
        return out
