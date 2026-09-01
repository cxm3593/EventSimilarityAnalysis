"""Plot the three-source Test 4 comparison for real, v2e, and V2CE."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


METRICS = [
    ("mmd_rbf03", "MMD RBF-3"),
    ("mmd_rbf15", "MMD RBF-15"),
    ("mmd_rbf75", "MMD RBF-75"),
    ("swd", "SWD"),
    ("chamfer", "Chamfer"),
]
SOURCES = [
    ("real", "real-real", "#5C677D"),
    ("v2e", "real-v2e", "#C44E52"),
    ("v2ce", "real-V2CE", "#4C78A8"),
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=project_root / "output" / "v2ce" / "test4_comparison_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "output" / "v2ce" / "test4_real_v2e_v2ce.html",
    )
    return parser.parse_args()


def format_value(metric: str, value: float) -> str:
    return f"{value:.3f}" if metric.startswith("mmd_") else f"{value:.2f}"


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.summary)
    data["frequency"] = data["frequency"].str.upper()

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[label for _, label in METRICS],
        horizontal_spacing=0.10,
        vertical_spacing=0.13,
    )

    for metric_index, (metric, _) in enumerate(METRICS):
        row = metric_index // 2 + 1
        col = metric_index % 2 + 1
        metric_data = data[data["metric"] == metric]

        for source, label, colour in SOURCES:
            source_data = (
                metric_data[metric_data["source"] == source]
                .set_index("frequency")
                .reindex(["F1", "F2", "F3", "F4", "F5"])
                .reset_index()
            )
            values = source_data["mean_distance"].tolist()
            fig.add_trace(
                go.Bar(
                    x=source_data["frequency"],
                    y=values,
                    name=label,
                    legendgroup=source,
                    showlegend=metric_index == 0,
                    marker_color=colour,
                    width=0.20,
                    text=[format_value(metric, value) for value in values],
                    textposition="outside",
                    cliponaxis=False,
                    customdata=source_data[["sd_distance", "valid_comparisons"]],
                    hovertemplate=(
                        f"{label}<br>%{{x}}<br>mean=%{{y:.5g}}"
                        "<br>SD=%{customdata[0]:.5g}"
                        "<br>comparisons=%{customdata[1]:.0f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        panel_max = metric_data["mean_distance"].max()
        fig.update_yaxes(
            title_text="Mean distance",
            range=[0, panel_max * 1.22],
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Recording", row=row, col=col)

    fig.update_layout(
        title={
            "text": "Test 4: same-phase distance from real data",
            "x": 0.5,
            "xanchor": "center",
        },
        barmode="group",
        bargap=0.28,
        bargroupgap=0.06,
        height=1080,
        width=1180,
        margin={"l": 70, "r": 35, "t": 115, "b": 60},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.045,
            "xanchor": "center",
            "x": 0.5,
        },
        template="plotly_white",
        font={"size": 14},
    )
    fig.update_traces(textfont_size=11)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn", full_html=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
