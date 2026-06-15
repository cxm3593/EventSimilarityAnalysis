"""Distance / similarity metric dispatch for event comparisons."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from .mmd import mmd_analysis


class MetricType(Enum):
    DISTANCE = "distance"
    SIMILARITY = "similarity"


@dataclass
class MetricResult:
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the historical dict shape expected by comparison strategies."""
        if self.metadata:
            return dict(self.metadata)
        return {"distance": self.value}


class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def type(self) -> MetricType:
        pass

    @abstractmethod
    def build_kwargs(self, config: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def compute(self, window_a, window_b, **kwargs) -> MetricResult:
        pass

    def describe_settings(self, metric_kwargs: dict[str, Any]) -> list[str]:
        """Optional human-readable lines printed before a comparison run."""
        return []

    def secondary_value(self, result: MetricResult) -> float | None:
        """Optional companion value (e.g. squared distance) for result tables."""
        return None


class MetricRegistry:
    _by_name: ClassVar[dict[str, BaseMetric]] = {}

    @classmethod
    def register(cls, metric: BaseMetric) -> BaseMetric:
        if metric.name in cls._by_name:
            raise ValueError(f"Metric {metric.name!r} is already registered.")
        cls._by_name[metric.name] = metric
        return metric

    @classmethod
    def get(cls, name: str) -> BaseMetric:
        try:
            return cls._by_name[name]
        except KeyError as exc:
            supported = sorted(cls._by_name)
            raise ValueError(
                f"Unsupported metric: {name!r}. Supported metrics: {supported}"
            ) from exc

    @classmethod
    def supported_names(cls) -> frozenset[str]:
        return frozenset(cls._by_name)


def register_metric(metric_cls: type[BaseMetric]) -> type[BaseMetric]:
    """Class decorator that registers a metric implementation."""
    MetricRegistry.register(metric_cls())
    return metric_cls


def get_metric(metric: str | BaseMetric) -> BaseMetric:
    if isinstance(metric, BaseMetric):
        return metric
    return MetricRegistry.get(metric)


def compare_events(
    events_a,
    events_b,
    metric: str | BaseMetric = "mmd",
    **kwargs,
) -> dict[str, Any]:
    """Compare two event subsets with the requested metric."""
    return get_metric(metric).compute(events_a, events_b, **kwargs).to_legacy_dict()


def _shared_feature_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": config.get("feature_names"),
        "feature_scales": config.get("feature_scales"),
    }


@register_metric
class MMDMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "mmd"

    @property
    def type(self) -> MetricType:
        return MetricType.DISTANCE

    def build_kwargs(self, config: dict[str, Any]) -> dict[str, Any]:
        mmd_config = config.get("mmd")
        if not mmd_config:
            raise ValueError("config.yaml must define an 'mmd' section when metric is 'mmd'.")
        return {
            **_shared_feature_kwargs(config),
            "chunk_size": mmd_config["chunk_size"],
            "rbf_kernel_max_distance": mmd_config["rbf_kernel_max_distance"],
            "rbf_kernel_target_similarity": mmd_config["rbf_kernel_target_similarity"],
            "backend": mmd_config["backend"],
        }

    def describe_settings(self, metric_kwargs: dict[str, Any]) -> list[str]:
        max_distance = metric_kwargs.get("rbf_kernel_max_distance")
        target_similarity = metric_kwargs.get("rbf_kernel_target_similarity")
        if max_distance is None:
            return []
        return [
            "RBF kernel max distance: "
            f"{max_distance} "
            f"(similarity <= {target_similarity} beyond this scaled distance)"
        ]

    def compute(self, window_a, window_b, **kwargs) -> MetricResult:
        raw = mmd_analysis(window_a, window_b, **kwargs)
        return MetricResult(value=float(raw["mmd"]), metadata=raw)

    def secondary_value(self, result: MetricResult) -> float | None:
        squared = result.metadata.get("mmd_squared")
        return float(squared) if squared is not None else None


# Populate the public alias after built-in metrics register themselves.
SUPPORTED_METRICS = MetricRegistry.supported_names()

# Register additional metrics in this module (or import them here) so they are
# available through get_metric() and compare_events().
