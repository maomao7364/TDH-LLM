"""Draw two standalone Node 47 figures from saved block-stitched results."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from plot_selected_abrupt_event_figures import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    LOCAL_MARK_EVERY,
    METHODS,
    STYLE,
    load_node_csv,
    select_abrupt_event,
    set_plot_style,
    style_axis,
)


NODE = 47
DEFAULT_SOURCE_CSV = (
    DEFAULT_SOURCE_DIR / f"node_{NODE:02d}_72h_block_stitched_values.csv"
)
PLOT_STYLE = {name: dict(style) for name, style in STYLE.items()}
PLOT_STYLE["LTFP-STSR"]["color"] = "#7A68A6"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw standalone 72-h error and selected-window flow figures."
    )
    parser.add_argument("--source_csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def style_boxed_axis(ax):
    style_axis(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#2F3337")
        spine.set_linewidth(0.9)


def plot_72h_absolute_error(timestamps, values, lo, hi, output_dir, dpi):
    truth = values["Ground Truth"]
    fig, ax = plt.subplots(figsize=(11.4, 4.8), facecolor="white")

    mae_values = {}
    for method in ("LTFP-STSR", "ST-LLM", "DeST-LLM"):
        style = PLOT_STYLE[method]
        error = np.abs(values[method] - truth)
        mae_values[method] = float(np.mean(error))
        ax.plot(
            timestamps,
            error,
            color=style["color"],
            linestyle="-",
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=method,
        )

    interval_text = (
        f"Selected interval: {timestamps[lo]:%Y-%m-%d %H:%M} "
        f"to {timestamps[hi - 1]:%H:%M}"
    )
    fig.suptitle(
        "Node 47: 72-h point-wise absolute error",
        y=0.98,
        fontsize=15,
    )
    fig.text(
        0.5,
        0.90,
        interval_text,
        ha="center",
        va="center",
        fontsize=11.5,
        color="#46515A",
    )
    ax.set_ylabel("Absolute error")
    ax.set_xlabel("Date and time")
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    style_boxed_axis(ax)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.18)
    output_path = output_dir / "node_47_72h_absolute_error_curves.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path, mae_values


def plot_selected_window_flow(
    timestamps,
    values,
    lo,
    hi,
    output_dir,
    dpi,
):
    fig, ax = plt.subplots(figsize=(10.8, 5.0), facecolor="white")
    event_slice = slice(lo, hi)

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
            label=method,
        )

    fig.suptitle(
        "Node 47: flow comparison during the selected abrupt event",
        y=0.98,
        fontsize=15,
    )
    fig.text(
        0.5,
        0.90,
        f"{timestamps[lo]:%Y-%m-%d %H:%M} to {timestamps[hi - 1]:%H:%M}",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#46515A",
    )
    ax.set_ylabel("Traffic flow")
    ax.set_xlabel("Date and time")
    ax.set_xlim(timestamps[lo], timestamps[hi - 1])
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    style_boxed_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    legend_items = dict(zip(labels, handles))
    legend_order = ("Ground Truth", "DeST-LLM", "ST-LLM", "LTFP-STSR")
    ax.legend(
        handles=[legend_items[method] for method in legend_order],
        labels=list(legend_order),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=4,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="#F4F7F8",
        edgecolor="#D1D9DE",
        borderpad=0.55,
        columnspacing=1.55,
        handlelength=2.7,
    )

    fig.subplots_adjust(left=0.09, right=0.985, top=0.70, bottom=0.17)
    output_path = output_dir / "node_47_selected_2h_flow_comparison.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    if not args.source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {args.source_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()
    timestamps, values = load_node_csv(args.source_csv)
    _, lo, hi, _, _ = select_abrupt_event(values)

    error_path, mae_values = plot_72h_absolute_error(
        timestamps,
        values,
        lo,
        hi,
        args.output_dir,
        args.dpi,
    )
    flow_path = plot_selected_window_flow(
        timestamps,
        values,
        lo,
        hi,
        args.output_dir,
        args.dpi,
    )

    print(f"Saved 72-h error figure: {error_path}")
    print(f"Saved selected-window flow figure: {flow_path}")
    print(
        "72-h MAE: "
        + ", ".join(f"{method}={mae_values[method]:.3f}" for method in METHODS)
    )


if __name__ == "__main__":
    main()
