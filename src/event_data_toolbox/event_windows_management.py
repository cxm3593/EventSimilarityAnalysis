"""Event time-window generation for comparison analyses."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


__all__ = ["EventWindow", "EventWindowsManager"]

StreamName = Literal["real", "v2e"]


class EventWindow:
    """One time window: metadata plus its event subset."""

    def __init__(
        self,
        source: StreamName,
        role: str,
        start: int,
        end: int,
        events,
    ):
        self.source = source
        self.role = role
        self.start = int(start)
        self.end = int(end)
        self.events = events
        self.n_events = len(events)

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "source": self.source,
            "role": self.role,
            "start": self.start,
            "end": self.end,
            "n_events": self.n_events,
        }
        if include_events:
            entry["events"] = self.events
        return entry


class _StreamIndexer:
    """Fast time slicing on a sorted event stream."""

    def __init__(self, data):
        self._data = data
        self._t = np.asarray(data["t"])

    def select(self, start: int, end: int):
        lo = int(np.searchsorted(self._t, start, side="left"))
        hi = int(np.searchsorted(self._t, end, side="left"))
        return self._data[lo:hi]


class EventWindowsManager:
    """Generate ``EventWindow`` instances from real and v2e event streams."""

    def __init__(self, real_data, v2e_data):
        self._real = _StreamIndexer(real_data)
        self._v2e = _StreamIndexer(v2e_data)

    def _select(self, source: StreamName, start: int, end: int):
        if source == "real":
            return self._real.select(start, end)
        return self._v2e.select(start, end)

    @staticmethod
    def _window_specs(start: int, width: int, count: int, stride: int) -> list[tuple[int, int]]:
        return [(start + i * stride, start + i * stride + width) for i in range(count)]

    def generate(
        self,
        baseline_start: int,
        baseline_end: int,
        n_real_windows: int,
        n_v2e_windows: int,
        stride: int | None = None,
        *,
        real_window_start: int | None = None,
        v2e_window_start: int | None = None,
    ) -> dict[str, Any]:
        """Build baseline + real + v2e windows for one scheme.

        Returns a dict with keys ``baseline`` (``EventWindow``), ``real`` and
        ``v2e`` (lists of comparison ``EventWindow``), and ``settings`` (layout
        metadata: width, stride, starts, counts).
        """
        if n_real_windows < 0 or n_v2e_windows < 0:
            raise ValueError("Window counts must be non-negative.")

        width = baseline_end - baseline_start
        if width <= 0:
            raise ValueError("baseline_end must be greater than baseline_start.")

        stride_value = width if stride is None else int(stride)
        if stride_value <= 0:
            raise ValueError("stride must be positive.")

        real_start = (
            baseline_start + stride_value
            if real_window_start is None
            else int(real_window_start)
        )
        v2e_start = baseline_start if v2e_window_start is None else int(v2e_window_start)

        settings = {
            "baseline_start": int(baseline_start),
            "baseline_end": int(baseline_end),
            "width": int(width),
            "stride": int(stride_value),
            "real_window_start": int(real_start),
            "v2e_window_start": int(v2e_start),
            "n_real_windows": int(n_real_windows),
            "n_v2e_windows": int(n_v2e_windows),
        }

        baseline = EventWindow(
            "real",
            "baseline",
            baseline_start,
            baseline_end,
            self._select("real", baseline_start, baseline_end),
        )

        real_windows = [
            EventWindow("real", "comparison", start, end, self._select("real", start, end))
            for start, end in self._window_specs(real_start, width, n_real_windows, stride_value)
        ]

        v2e_windows = [
            EventWindow("v2e", "comparison", start, end, self._select("v2e", start, end))
            for start, end in self._window_specs(v2e_start, width, n_v2e_windows, stride_value)
        ]

        return {
            "baseline": baseline,
            "real": real_windows,
            "v2e": v2e_windows,
            "settings": settings,
        }
