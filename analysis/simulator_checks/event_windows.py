"""Plot one 5 ms window from real, v2e and V2CE, for F1 and F5.

A qualitative view of where each source places its events: red for ON, blue for OFF.
Writes event_windows.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared import WINDOW_US, event_paths, read_span, start

RECORDINGS = ["f1", "f5"]
START_US = 1_100_000


def main():
    output = start("event_windows", __file__,
                   {"window_us": WINDOW_US, "start_us": START_US}, recordings=RECORDINGS)
    figure, axes = plt.subplots(len(RECORDINGS), 3, figsize=(15, 5 * len(RECORDINGS)))
    for row, recording in enumerate(RECORDINGS):
        for col, (name, path) in enumerate(event_paths(recording).items()):
            events = read_span(path, START_US, START_US + WINDOW_US)
            on = events["p"] > 0
            axis = axes[row, col]
            axis.scatter(events["x"][on], events["y"][on], s=0.3, c="#d1352b", alpha=0.5, lw=0)
            axis.scatter(events["x"][~on], events["y"][~on], s=0.3, c="#2a78d6", alpha=0.5, lw=0)
            axis.set_xlim(270, 715)
            axis.set_ylim(600, 158)
            axis.set_aspect("equal")
            axis.set_title(f"{recording.upper()} {name}: {len(events):,} events in 5 ms "
                           "(red ON, blue OFF)", fontsize=10)
            print(f"{recording.upper()} {name}: {len(events):,} events")
    plt.tight_layout()
    plt.savefig(output / "event_windows.png", dpi=90)
    print(f"\nWrote {output / 'event_windows.png'}")


if __name__ == "__main__":
    main()
