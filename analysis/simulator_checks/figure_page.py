"""Build the simulator-comparison figure page from the per-window comparison results.

Floor = real rotations 1-2 against real rotation 0 at the same phase (rotation 0 against
itself is excluded). v2e and V2CE = rotations 0-2 against real rotation 0. Dots are medians,
whiskers the 5th to 95th percentile over windows.

The second figure converts each distance to the rotation that produces the same distance
between real windows in the temporal offset test (0-45 degrees). Quantiles survive that
monotonic conversion, so the whiskers are the converted 5th and 95th percentiles.

Writes simulator_comparison.html, light and dark, with a table view of every value.
"""

import html
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots

from _shared import RECORDINGS, TEMPORAL_RUN, comparison_runs, start, trial_name
from unit_conversion import curve_for, invert, simulator_distances

METRICS = [("swd", "SWD"), ("mmd_rbf75", "MMD, large kernel (75 px)"),
           ("mmd_rbf15", "MMD, medium kernel (15 px)"), ("mmd_rbf03", "MMD, small kernel (3 px)"),
           ("chamfer", "Chamfer (independent check)")]
SOURCES = [("floor", "Real, another rotation (floor)", -0.22), ("v2e", "v2e", 0.0),
           ("v2ce", "V2CE", 0.22)]
# Validated with the dataviz palette checker in both modes (all pairs).
THEME = {
    "light": {"surface": "#fcfcfb", "page": "#f9f9f7", "primary": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "floor": "#898781", "v2e": "#eb6834", "v2ce": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "primary": "#ffffff", "secondary": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "floor": "#898781", "v2e": "#d95926", "v2ce": "#3987e5"},
}
L = THEME["light"]
X_OF = {recording: i + 1 for i, recording in enumerate(RECORDINGS)}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def fmt(value):
    return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"


def summarise(windows, column):
    return (windows.groupby(["recording", "metric", "src"])[column]
            .agg(median="median", q05=lambda v: v.quantile(0.05), q95=lambda v: v.quantile(0.95),
                 n="count").reset_index())


def add_points(figure, table, row, col, legend, unit=""):
    for src, label, offset in SOURCES:
        block = table[table.src == src].set_index("recording").loc[RECORDINGS].reset_index()
        hover = []
        for _, r in block.iterrows():
            note = " (beyond the tested range)" if r.get("capped", False) else ""
            hover.append(f"<b>{label}</b> · {r.recording.upper()}<br>median {fmt(r['median'])}{unit}"
                         f"<br>5–95%: {fmt(r.q05)} – {fmt(r.q95)}{unit}{note}")
        figure.add_trace(go.Scatter(
            x=[X_OF[r] + offset for r in block.recording], y=block["median"], mode="markers",
            name=label, legendgroup=src, showlegend=legend, meta=src,
            marker=dict(size=9, color=L[src], line=dict(width=2, color=L["surface"])),
            error_y=dict(type="data", symmetric=False, thickness=2, width=0, color=L[src],
                         array=(block.q95 - block["median"]).clip(lower=0),
                         arrayminus=(block["median"] - block.q05).clip(lower=0)),
            text=hover, hovertemplate="%{text}<extra></extra>"), row=row, col=col)


def style(figure, height):
    figure.update_layout(
        height=height, margin=dict(l=64, r=96, t=72, b=48), paper_bgcolor=L["surface"],
        plot_bgcolor=L["surface"], font=dict(family=FONT, size=12, color=L["secondary"]),
        legend=dict(orientation="h", x=0, y=1.08, xanchor="left", yanchor="bottom",
                    font=dict(color=L["primary"])),
        hoverlabel=dict(font=dict(family=FONT)), hovermode="closest")
    figure.update_xaxes(tickvals=[1, 2, 3, 4, 5], ticktext=[r.upper() for r in RECORDINGS],
                        range=[0.5, 5.6], showgrid=False, linecolor=L["axis"], ticks="",
                        tickfont=dict(color=L["muted"]))
    figure.update_yaxes(gridcolor=L["grid"], gridwidth=1, zeroline=False, linecolor=L["axis"],
                        tickfont=dict(color=L["muted"]), rangemode="tozero")
    for annotation in figure.layout.annotations:
        annotation.font = dict(color=L["primary"], size=13)


def direct_labels(figure, table):
    for src, label, offset in SOURCES:
        r = table[(table.src == src) & (table.recording == "f5")].iloc[0]
        figure.add_annotation(x=5 + offset + 0.12, y=r["median"], xref="x", yref="y",
                              text=label.split(",")[0], showarrow=False, xanchor="left",
                              font=dict(size=11, color=L["secondary"]), name="direct-label")


def table_html(rows, headers):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
                   for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    output = start("figures", __file__,
                   {"quantiles": [0.05, 0.95], "rotation_curve_range_deg": [0, 45],
                    "temporal_run": TEMPORAL_RUN.name,
                    "comparison_runs": {k: v.name for k, v in comparison_runs().items()}})
    windows = simulator_distances()
    windows["src"] = windows.source.replace({"real": "floor"})
    stats = summarise(windows, "distance")
    events = summarise(windows[windows.metric == "swd"], "n_events_b")

    temporal = pd.read_csv(TEMPORAL_RUN / "summary.csv")
    rows = []
    for recording in RECORDINGS:
        for metric in ("swd", "mmd_rbf75"):
            curve = curve_for(temporal, recording, metric, "temporal_offset")
            for src, _, _ in SOURCES:
                s = stats[(stats.recording == recording) & (stats.metric == metric)
                          & (stats.src == src)].iloc[0]
                median, capped_m = invert(curve, s["median"])
                low, _ = invert(curve, s["q05"])
                high, capped_h = invert(curve, s["q95"])
                rows.append({"recording": recording, "metric": metric, "src": src, "median": median,
                             "q05": low, "q95": high, "capped": capped_m or capped_h})
    degrees = pd.DataFrame(rows)

    fig1 = make_subplots(rows=3, cols=2, vertical_spacing=0.11, horizontal_spacing=0.1,
                         subplot_titles=[m for _, m in METRICS] + ["Events per 5 ms window"])
    for k, (metric, _) in enumerate(METRICS):
        add_points(fig1, stats[stats.metric == metric], k // 2 + 1, k % 2 + 1, legend=(k == 0))
    add_points(fig1, events, 3, 2, legend=False)
    fig1.update_yaxes(type="log", row=3, col=2, rangemode="normal")
    style(fig1, 900)
    direct_labels(fig1, stats[stats.metric == "swd"])

    fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                         subplot_titles=["SWD as equivalent rotation (degrees)",
                                         "Large-kernel MMD as equivalent rotation (degrees)"])
    add_points(fig2, degrees[degrees.metric == "swd"], 1, 1, legend=True, unit="°")
    add_points(fig2, degrees[degrees.metric == "mmd_rbf75"], 1, 2, legend=False, unit="°")
    style(fig2, 420)
    fig2.update_yaxes(range=[0, 46])
    direct_labels(fig2, degrees[degrees.metric == "swd"])

    name = {s: label for s, label, _ in SOURCES}
    metric_name = dict(METRICS)
    t1 = [(r.recording.upper(), metric_name[r.metric], name[r.src], fmt(r["median"]), fmt(r.q05),
           fmt(r.q95), int(r.n)) for _, r in stats.sort_values(["metric", "recording", "src"]).iterrows()]
    t1 += [(r.recording.upper(), "Events per 5 ms window", name[r.src], f"{r['median']:.0f}",
            f"{r.q05:.0f}", f"{r.q95:.0f}", int(r.n))
           for _, r in events.sort_values(["recording", "src"]).iterrows()]
    t2 = [(r.recording.upper(), metric_name[r.metric], name[r.src], f"{r['median']:.1f}",
           f"{r.q05:.1f}", "≥45" if r.capped else f"{r.q95:.1f}")
          for _, r in degrees.sort_values(["metric", "recording", "src"]).iterrows()]

    dark = THEME["dark"]
    variables = lambda theme: "".join(f"--{k}:{v};" for k, v in theme.items())  # noqa: E731
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulator comparison</title>
<style>
:root{{color-scheme:light;{variables(L)}}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;{variables(dark)}}}}}
:root[data-theme="dark"]{{color-scheme:dark;{variables(dark)}}}
body{{margin:0;background:var(--page);color:var(--primary);font:15px/1.5 {FONT}}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:32px 0 6px}}
p{{color:var(--secondary);margin:6px 0 12px;max-width:80ch}}
.card{{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:8px;overflow:hidden}}
details{{margin-top:10px}} summary{{cursor:pointer;color:var(--secondary)}}
table{{border-collapse:collapse;font-size:13px;margin-top:8px;font-variant-numeric:tabular-nums}}
th,td{{padding:4px 10px;border-bottom:1px solid var(--grid);text-align:left}} th{{color:var(--secondary);font-weight:600}}
.scroll{{overflow-x:auto}}
</style><script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script></head><body><main>
<h1>Simulator comparison: v2e and V2CE against the real recordings</h1>
<p>Each real 5 ms window from a reference rotation is compared with the window at the same phase from another real rotation (the floor), from v2e, and from V2CE. Dots are medians over windows; whiskers span the 5th to 95th percentile. 40 phase-spread windows per rotation; three rotations per simulator and two for the floor.</p>
<h2>Distances by metric and recording</h2>
<div class="card"><div id="fig1"></div></div>
<p>MMD values are clamped at zero, so the floor sits at zero where the two real rotations cannot be told apart (the faster recordings at the smaller kernels). Event counts use a log scale.</p>
<details><summary>Table view</summary><div class="scroll">{table_html(t1, ["Recording", "Metric", "Source", "Median", "5%", "95%", "Windows"])}</div></details>
<h2>Distances as an equivalent chopper rotation</h2>
<div class="card"><div id="fig2"></div></div>
<p>Each distance is converted to the rotation that produces the same distance between real windows in the temporal offset test (0 to 45 degrees). Values beyond that range are drawn at 45 degrees.</p>
<details><summary>Table view</summary><div class="scroll">{table_html(t2, ["Recording", "Metric", "Source", "Median (°)", "5% (°)", "95% (°)"])}</div></details>
<p>Sources: output/simulator/comparison (runs of 26 August 2026) and output/perturbation/{TEMPORAL_RUN.name}.</p>
</main>
<script>
const FIGS = {{fig1: {fig1.to_json()}, fig2: {fig2.to_json()}}};
const THEME = {json.dumps(THEME)};
function themed(fig, t) {{
  const f = JSON.parse(JSON.stringify(fig));
  f.layout.paper_bgcolor = t.surface; f.layout.plot_bgcolor = t.surface;
  f.layout.font.color = t.secondary; f.layout.legend.font.color = t.primary;
  for (const k of Object.keys(f.layout)) {{
    if (/^[xy]axis\\d*$/.test(k)) {{
      f.layout[k].linecolor = t.axis; f.layout[k].tickfont = {{color: t.muted}};
      if (k.startsWith('y')) f.layout[k].gridcolor = t.grid;
    }}
  }}
  for (const a of f.layout.annotations || []) {{
    a.font = Object.assign({{}}, a.font, {{color: a.name === 'direct-label' ? t.secondary : t.primary}});
  }}
  for (const tr of f.data) {{
    tr.marker.color = t[tr.meta]; tr.marker.line.color = t.surface; tr.error_y.color = t[tr.meta];
  }}
  return f;
}}
function render() {{
  const forced = document.documentElement.getAttribute('data-theme');
  const dark = forced ? forced === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  for (const [id, fig] of Object.entries(FIGS)) {{
    const f = themed(fig, THEME[dark ? 'dark' : 'light']);
    Plotly.react(id, f.data, f.layout, {{responsive: true, displaylogo: false}});
  }}
}}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
render();
</script></body></html>"""
    (output / "simulator_comparison.html").write_text(page, encoding="utf-8")
    stats.to_csv(output / "distance_summary.csv", index=False)
    degrees.to_csv(output / "equivalent_rotation.csv", index=False)
    print(f"Wrote {output / 'simulator_comparison.html'}")
    print(stats[stats.metric == "swd"].pivot(index="recording", columns="src", values="median")
          .round(2).to_string())


if __name__ == "__main__":
    main()
