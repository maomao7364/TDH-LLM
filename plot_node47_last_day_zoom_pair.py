"""Draw the final-day Node 47 comparison with an abrupt-event inset."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator

from plot_block_stitched_flow_comparisons import (
    BLOCK_DIR,
    LEGEND_LABELS,
    LEGEND_ORDER,
    PLOT_STYLE,
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
DEFAULT_OUTPUT_DIR = BLOCK_DIR.parent / "node_47_last_day_inset_zoom"
FIGURE_STYLE = {
    method: dict(PLOT_STYLE[method])
    for method in LEGEND_ORDER
}
FIGURE_STYLE["Ground Truth"].update(
    color="#2F6F66",
    linewidth=1.15,
    alpha=1.0,
    zorder=6,
)
MAIN_MARK_EVERY = {
    "Ground Truth": (0, 24),
    "DeST-LLM": (6, 24),
    "ST-LLM": (12, 24),
    "LTFP-STSR": (18, 24),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw Node 47 final-day overview with an abrupt-event inset."
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
    handles = [
        Line2D(
            [0],
            [0],
            color=FIGURE_STYLE[method]["color"],
            linestyle="-",
            linewidth=FIGURE_STYLE[method]["linewidth"],
            marker=FIGURE_STYLE[method]["marker"],
            markersize=FIGURE_STYLE[method]["markersize"],
            markerfacecolor="white",
            markeredgecolor=FIGURE_STYLE[method]["color"],
            markeredgewidth=0.9,
        )
        for method in LEGEND_ORDER
    ]
    fig.legend(
        handles=handles,
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


def plot_series(ax, timestamps, values, data_slice, show_markers=False):
    order = ("Ground Truth", "LTFP-STSR", "ST-LLM", "DeST-LLM")
    for method in order:
        style = FIGURE_STYLE[method]
        if show_markers:
            ax.plot(
                timestamps[data_slice],
                values[method][data_slice],
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
            )
        else:
            ax.plot(
                timestamps[data_slice],
                values[method][data_slice],
                color=style["color"],
                linestyle="-",
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"] * 0.82,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=0.8,
                markevery=MAIN_MARK_EVERY[method],
                alpha=style["alpha"],
                zorder=style["zorder"],
            )


def event_y_limits(values, event_slice):
    event_values = np.concatenate(
        [values[method][event_slice] for method in LEGEND_ORDER]
    )
    y_min = float(np.min(event_values))
    y_max = float(np.max(event_values))
    padding = max((y_max - y_min) * 0.08, 2.0)
    return y_min - padding, y_max + padding


def style_inset_axis(ax):
    ax.set_facecolor("#FCFCFC")
    ax.grid(
        axis="both",
        color="#D8DEE3",
        linestyle="--",
        linewidth=0.55,
        alpha=0.60,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.0)
    ax.tick_params(
        axis="both",
        labelsize=10.5,
        direction="out",
        length=3.2,
        width=0.8,
        colors="#000000",
        pad=2.5,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def add_zoom_indicator(
    fig,
    main_ax,
    inset_ax,
    event_start,
    event_end,
    y_min,
    y_max,
):
    x_start = mdates.date2num(event_start)
    x_end = mdates.date2num(event_end)
    indicator_color = "#343A40"
    indicator_style = (0, (4, 2))

    main_ax.add_patch(
        Rectangle(
            (x_start, y_min),
            x_end - x_start,
            y_max - y_min,
            fill=False,
            edgecolor=indicator_color,
            linewidth=1.15,
            linestyle=indicator_style,
            zorder=8,
        )
    )

    for main_xy, inset_xy in (
        ((x_start, y_min), (0.0, 1.0)),
        ((x_end, y_min), (1.0, 1.0)),
    ):
        fig.add_artist(
            ConnectionPatch(
                xyA=main_xy,
                coordsA=main_ax.transData,
                xyB=inset_xy,
                coordsB=inset_ax.transAxes,
                color=indicator_color,
                linewidth=1.0,
                linestyle=indicator_style,
                alpha=0.88,
                zorder=7,
                clip_on=False,
            )
        )


def plot_last_day_with_inset(
    timestamps,
    values,
    day_lo,
    day_hi,
    event_lo,
    event_hi,
    output_dir,
    dpi=400,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    day_slice = slice(day_lo, day_hi)
    event_slice = slice(event_lo, event_hi)
    fig, ax = plt.subplots(figsize=(12.0, 5.8), facecolor="white")

    plot_series(ax, timestamps, values, day_slice)
    style_emphasized_axis(ax)
    ax.set_ylabel("Traffic flow")
    ax.set_xlabel("Date and time")
    ax.set_xlim(timestamps[day_lo], timestamps[day_hi - 1])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    inset_ax = ax.inset_axes([0.39, 0.08, 0.48, 0.42], zorder=10)
    plot_series(inset_ax, timestamps, values, event_slice, show_markers=True)
    inset_y_min, inset_y_max = event_y_limits(values, event_slice)
    inset_ax.set_xlim(timestamps[event_lo], timestamps[event_hi - 1])
    inset_ax.set_ylim(inset_y_min, inset_y_max)
    inset_ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    inset_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    style_inset_axis(inset_ax)
    inset_ax.set_title(
        f"{timestamps[event_lo]:%b %d, %H:%M}"
        f"-{timestamps[event_hi - 1]:%H:%M}",
        fontsize=11.5,
        pad=4,
    )

    add_zoom_indicator(
        fig,
        ax,
        inset_ax,
        timestamps[event_lo],
        timestamps[event_hi - 1],
        inset_y_min,
        inset_y_max,
    )

    fig.suptitle("Node 47: final-day flow comparison", y=0.985, fontsize=15)
    add_shared_legend(fig)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.76, bottom=0.15)

    output_path = output_dir / "node_47_last_day_flow_with_inset.png"
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

    output_path = plot_last_day_with_inset(
        timestamps,
        values,
        day_lo,
        day_hi,
        event_lo,
        event_hi,
        args.output_dir,
        args.dpi,
    )
    print(f"Saved final-day comparison with inset: {output_path}")
    print(
        f"Overview: {timestamps[day_lo]} to {timestamps[day_hi - 1]}; "
        f"selected period: {timestamps[event_lo]} to {timestamps[event_hi - 1]}"
    )


if __name__ == "__main__":
    main()
