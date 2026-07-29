"""Draw final-day and selected-period flow comparisons for Node 47."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from plot_block_stitched_flow_comparisons import (
    BLOCK_DIR,
    LEGEND_LABELS,
    LEGEND_ORDER,
    METHODS,
    PLOT_STYLE,
    line_kwargs,
    shared_legend_handles,
    style_boxed_axis,
)
from plot_selected_abrupt_event_figures import (
    LOCAL_MARK_EVERY,
    load_node_csv,
    select_abrupt_event,
    set_plot_style,
)


NODE = 47
DEFAULT_SOURCE_CSV = BLOCK_DIR / f"node_{NODE:02d}_72h_block_stitched_values.csv"
DEFAULT_OUTPUT_DIR = BLOCK_DIR.parent / "node_47_last_day_zoom_pair"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw Node 47 final-day overview and abrupt-event zoom figures."
    )
    parser.add_argument("--source_csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def display_name(method):
    return LEGEND_LABELS.get(method, method)


def final_calendar_day_bounds(timestamps):
    """Return [lo, hi) covering the final calendar day in the series."""
    timestamps = np.asarray(timestamps, dtype=object)
    if timestamps.size == 0:
        raise ValueError("Cannot select the final day from an empty timestamp array.")
    final_date = timestamps[-1].date()
    indices = np.flatnonzero([timestamp.date() == final_date for timestamp in timestamps])
    if indices.size == 0:
        raise ValueError(f"No samples found for final date {final_date}.")
    return int(indices[0]), int(indices[-1] + 1)


def add_shared_legend(fig):
    fig.legend(
        handles=shared_legend_handles(),
        labels=[display_name(method) for method in LEGEND_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.79),
        ncol=4,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="#F4F7F8",
        edgecolor="#D1D9DE",
        borderpad=0.55,
        columnspacing=1.55,
        handlelength=2.7,
        fontsize=15,
    )


def style_emphasized_axis(ax):
    style_boxed_axis(ax)
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.3)
    ax.tick_params(
        axis="both",
        labelsize=13,
        direction="out",
        length=4.2,
        width=1.0,
        colors="#000000",
    )
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)


def plot_last_day_overview(
    timestamps,
    values,
    day_lo,
    day_hi,
    output_dir,
    dpi=400,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    day_slice = slice(day_lo, day_hi)
    fig, ax = plt.subplots(figsize=(12.0, 4.8), facecolor="white")

    ax.plot(
        timestamps[day_slice],
        values["Ground Truth"][day_slice],
        label="Ground Truth",
        **line_kwargs("Ground Truth"),
    )
    for method in METHODS:
        ax.plot(
            timestamps[day_slice],
            values[method][day_slice],
            label=display_name(method),
            **line_kwargs(method),
        )

    style_emphasized_axis(ax)
    ax.set_ylabel("Traffic flow")
    ax.set_xlabel("Date and time")
    ax.set_xlim(timestamps[day_lo], timestamps[day_hi - 1])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    fig.suptitle("Node 47: final-day flow comparison", y=0.985, fontsize=15)
    add_shared_legend(fig)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.76, bottom=0.18)

    output_path = output_dir / "node_47_last_day_flow_overview.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_selected_period_flow(
    timestamps,
    values,
    event_lo,
    event_hi,
    output_dir,
    dpi=400,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    event_slice = slice(event_lo, event_hi)
    fig, ax = plt.subplots(figsize=(10.8, 5.0), facecolor="white")

    for method in ("Ground Truth", "LTFP-STSR", "ST-LLM", "DeST-LLM"):
        style = PLOT_STYLE[method]
        ax.plot(
            timestamps[event_slice],
            values[method][event_slice],
            color=style["color"],
            linestyle="-",
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.9,
            markevery=LOCAL_MARK_EVERY[method],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=display_name(method),
        )

    style_emphasized_axis(ax)
    ax.set_ylabel("Traffic flow")
    ax.set_xlabel("Date and time")
    ax.set_xlim(timestamps[event_lo], timestamps[event_hi - 1])
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    interval = (
        f"{timestamps[event_lo]:%Y-%m-%d %H:%M}"
        f" to {timestamps[event_hi - 1]:%H:%M}"
    )
    fig.suptitle(
        f"Node 47: selected-period flow comparison ({interval})",
        y=0.985,
        fontsize=15,
    )
    add_shared_legend(fig)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.76, bottom=0.18)

    output_path = output_dir / "node_47_selected_period_flow_comparison.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    if not args.source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {args.source_csv}")

    set_plot_style()
    timestamps, values = load_node_csv(args.source_csv)
    day_lo, day_hi = final_calendar_day_bounds(timestamps)
    _, event_lo, event_hi, _, _ = select_abrupt_event(values)
    if not (day_lo <= event_lo < event_hi <= day_hi):
        raise ValueError(
            "Selected abrupt-event window is outside the final calendar day: "
            f"day=[{day_lo}, {day_hi}), event=[{event_lo}, {event_hi})."
        )

    overview_path = plot_last_day_overview(
        timestamps,
        values,
        day_lo,
        day_hi,
        args.output_dir,
        args.dpi,
    )
    selected_path = plot_selected_period_flow(
        timestamps,
        values,
        event_lo,
        event_hi,
        args.output_dir,
        args.dpi,
    )
    previous_zoom_path = args.output_dir / "node_47_abrupt_event_zoom.png"
    if previous_zoom_path.exists():
        previous_zoom_path.unlink()

    print(f"Saved final-day overview: {overview_path}")
    print(f"Saved selected-period comparison: {selected_path}")
    print(
        f"Overview: {timestamps[day_lo]} to {timestamps[day_hi - 1]}; "
        f"selected period: {timestamps[event_lo]} to {timestamps[event_hi - 1]}"
    )


if __name__ == "__main__":
    main()
