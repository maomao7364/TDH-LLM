"""Plot three publication-ready abrupt-event comparisons from saved results.

The script reads the existing 72-hour block-stitched CSV files only. It does
not run model inference or modify any saved prediction/log file.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = (
    PROJECT_DIR / "logs" / "2026-06-18-22-38-06-pems07_custom"
)
DEFAULT_SOURCE_DIR = (
    DEFAULT_LOG_DIR
    / "flow_sequence_comparison"
    / "all_nodes_72h_block_stitch"
)
DEFAULT_OUTPUT_DIR = DEFAULT_LOG_DIR / "selected_abrupt_event_nature"

SELECTED_NODES = (47, 22, 5)
METHODS = ("DeST-LLM", "ST-LLM", "LTFP-STSR")
OUTPUT_LEN = 24
WINDOW_RADIUS = 12

# High-contrast publication palette. Line style and marker provide secondary
# encodings so that overlapping trajectories remain distinguishable.
STYLE = {
    "Ground Truth": {
        "color": "#566573",
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "o",
        "markersize": 4.2,
        "alpha": 0.90,
        "zorder": 2,
    },
    "DeST-LLM": {
        "color": "#2F75B5",
        "linestyle": "--",
        "linewidth": 2.3,
        "marker": "s",
        "markersize": 4.5,
        "alpha": 1.0,
        "zorder": 5,
    },
    "ST-LLM": {
        "color": "#D8665B",
        "linestyle": "-.",
        "linewidth": 1.75,
        "marker": "^",
        "markersize": 4.0,
        "alpha": 0.95,
        "zorder": 4,
    },
    "LTFP-STSR": {
        "color": "#D9A441",
        "linestyle": ":",
        "linewidth": 1.75,
        "marker": "D",
        "markersize": 4.0,
        "alpha": 0.95,
        "zorder": 3,
    },
}

LOCAL_MARK_EVERY = {
    "Ground Truth": (0, 4),
    "DeST-LLM": (1, 4),
    "ST-LLM": (2, 4),
    "LTFP-STSR": (3, 4),
}

ERROR_MARK_EVERY = {
    "DeST-LLM": (0, 3),
    "ST-LLM": (1, 3),
    "LTFP-STSR": (2, 3),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot selected abrupt-event comparisons from saved CSV results."
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nodes", type=int, nargs="+", default=list(SELECTED_NODES))
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def set_plot_style():
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 12,
            "axes.linewidth": 0.85,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_node_csv(path):
    timestamps = []
    values = {
        "Ground Truth": [],
        "DeST-LLM": [],
        "ST-LLM": [],
        "LTFP-STSR": [],
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamps.append(datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S"))
            values["Ground Truth"].append(float(row["ground_truth"]))
            for method in METHODS:
                values[method].append(float(row[method]))

    if not timestamps:
        raise ValueError(f"No rows found in {path}")
    return np.asarray(timestamps, dtype=object), {
        name: np.asarray(series, dtype=np.float64)
        for name, series in values.items()
    }


def select_abrupt_event(values):
    truth = values["Ground Truth"]
    jumps = np.abs(np.diff(truth, prepend=truth[0]))
    abrupt_threshold = max(float(np.percentile(jumps[1:], 90)), 15.0)
    candidate_mask = jumps >= abrupt_threshold

    dest_error = np.abs(values["DeST-LLM"] - truth)
    best_baseline_error = np.minimum(
        np.abs(values["ST-LLM"] - truth),
        np.abs(values["LTFP-STSR"] - truth),
    )
    local_gain = best_baseline_error - dest_error
    candidate_scores = np.where(candidate_mask, local_gain, -np.inf)
    event_idx = int(np.argmax(candidate_scores))
    if not np.isfinite(candidate_scores[event_idx]):
        event_idx = int(np.argmax(jumps))

    lo = max(0, event_idx - WINDOW_RADIUS)
    hi = min(len(truth), event_idx + WINDOW_RADIUS + 1)
    return event_idx, lo, hi, jumps, local_gain


def window_mae(values, lo, hi):
    truth = values["Ground Truth"][lo:hi]
    return {
        method: float(np.mean(np.abs(values[method][lo:hi] - truth)))
        for method in METHODS
    }


def add_panel_label(ax, label):
    ax.text(
        -0.075,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


def style_axis(ax):
    ax.grid(axis="both", color="#D8DEE3", linestyle="--", linewidth=0.65, alpha=0.62)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors="#333333")


def plot_overview(ax, timestamps, values, lo, hi):
    for method in ("Ground Truth", "LTFP-STSR", "ST-LLM", "DeST-LLM"):
        style = STYLE[method]
        ax.plot(
            timestamps,
            values[method],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=method,
        )

    ax.axvspan(
        timestamps[lo],
        timestamps[hi - 1],
        facecolor="#E1E7EB",
        edgecolor="#6F7C86",
        linewidth=1.15,
        alpha=0.82,
        zorder=0.5,
    )
    ax.axvline(timestamps[lo], color="#6F7C86", linewidth=0.9, alpha=0.90, zorder=1)
    ax.axvline(timestamps[hi - 1], color="#6F7C86", linewidth=0.9, alpha=0.90, zorder=1)
    for idx in range(OUTPUT_LEN, len(timestamps), OUTPUT_LEN):
        ax.axvline(
            timestamps[idx],
            color="#C9D0D6",
            linewidth=0.45,
            linestyle=":",
            alpha=0.35,
            zorder=0,
        )

    region_midpoint = timestamps[lo] + (timestamps[hi - 1] - timestamps[lo]) / 2
    y_min, y_max = ax.get_ylim()
    ax.text(
        region_midpoint,
        y_max - 0.045 * (y_max - y_min),
        "Zoomed region (b)",
        ha="center",
        va="top",
        fontsize=8.8,
        color="#46515A",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "#AEB8BF",
            "linewidth": 0.65,
            "alpha": 0.94,
        },
        zorder=6,
    )

    ax.set_ylabel("Traffic flow")
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.set_title("72-h block-stitched trajectory", loc="left", pad=8)
    style_axis(ax)
    add_panel_label(ax, "a")


def plot_event_window(ax, timestamps, values, event_idx, lo, hi):
    event_slice = slice(lo, hi)
    for method in ("Ground Truth", "LTFP-STSR", "ST-LLM", "DeST-LLM"):
        style = STYLE[method]
        ax.plot(
            timestamps[event_slice],
            values[method][event_slice],
            color=style["color"],
            linestyle=style["linestyle"],
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

    ax.axvspan(
        timestamps[max(event_idx - 1, lo)],
        timestamps[event_idx],
        color="#DDE5EA",
        alpha=0.72,
        zorder=0,
    )
    event_truth = values["Ground Truth"][event_idx]
    previous_truth = values["Ground Truth"][max(event_idx - 1, 0)]
    jump = event_truth - previous_truth
    ax.annotate(
        f"Abrupt change: {jump:+.0f}",
        xy=(timestamps[event_idx], event_truth),
        xytext=(8, 22 if jump < 0 else -28),
        textcoords="offset points",
        ha="left",
        va="bottom" if jump < 0 else "top",
        fontsize=10.5,
        color="#3F4A52",
        arrowprops={
            "arrowstyle": "->",
            "color": "#66717A",
            "linewidth": 0.9,
            "shrinkA": 2,
            "shrinkB": 3,
        },
    )

    ax.set_ylabel("Traffic flow")
    ax.set_xlim(timestamps[lo], timestamps[hi - 1])
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(
        f"Zoomed view of shaded interval ({timestamps[lo]:%Y-%m-%d %H:%M}"
        f" to {timestamps[hi - 1]:%H:%M})",
        loc="left",
        pad=8,
    )
    style_axis(ax)
    add_panel_label(ax, "b")


def plot_local_error(ax, timestamps, values, event_idx, lo, hi):
    event_slice = slice(lo, hi)
    truth = values["Ground Truth"][event_slice]
    maes = window_mae(values, lo, hi)
    for method in METHODS:
        style = STYLE[method]
        error = np.abs(values[method][event_slice] - truth)
        ax.plot(
            timestamps[event_slice],
            error,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=max(style["linewidth"] - 0.15, 1.4),
            marker=style["marker"],
            markersize=max(style["markersize"] - 0.7, 3.2),
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.85,
            markevery=ERROR_MARK_EVERY[method],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )

    ax.axvspan(
        timestamps[max(event_idx - 1, lo)],
        timestamps[event_idx],
        color="#DDE5EA",
        alpha=0.72,
        zorder=0,
    )
    metric_text = (
        f"Window MAE  DeST-LLM {maes['DeST-LLM']:.2f}   "
        f"ST-LLM {maes['ST-LLM']:.2f}   "
        f"LTFP-STSR {maes['LTFP-STSR']:.2f}"
    )
    ax.text(
        0.995,
        0.94,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#3F4A52",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#D6DDE2",
            "linewidth": 0.75,
            "alpha": 0.92,
        },
    )

    ax.set_ylabel("Absolute error")
    ax.set_xlabel("Timestamp")
    ax.set_xlim(timestamps[lo], timestamps[hi - 1])
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title("Absolute error within the selected window", loc="left", pad=8)
    style_axis(ax)
    add_panel_label(ax, "c")


def plot_node(node, timestamps, values, output_dir, dpi):
    event_idx, lo, hi, jumps, local_gain = select_abrupt_event(values)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.4, 8.2),
        gridspec_kw={"height_ratios": [1.45, 1.35, 0.95], "hspace": 0.45},
        facecolor="white",
    )
    plot_overview(axes[0], timestamps, values, lo, hi)
    plot_event_window(axes[1], timestamps, values, event_idx, lo, hi)
    plot_local_error(axes[2], timestamps, values, event_idx, lo, hi)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=STYLE[name]["color"],
            linestyle=STYLE[name]["linestyle"],
            linewidth=STYLE[name]["linewidth"],
            marker=STYLE[name]["marker"],
            markersize=STYLE[name]["markersize"],
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=name,
        )
        for name in ("Ground Truth", "DeST-LLM", "ST-LLM", "LTFP-STSR")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="#F4F7F8",
        edgecolor="#D1D9DE",
        borderpad=0.55,
        columnspacing=1.8,
        handlelength=2.6,
    )

    event_label = timestamps[event_idx].strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"Node {node:02d}: abrupt traffic variation at {event_label}",
        x=0.5,
        y=0.985,
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.875, bottom=0.085)

    output_path = output_dir / f"node_{node:02d}_abrupt_event_nature.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "path": output_path,
        "event_time": event_label,
        "jump": float(jumps[event_idx]),
        "event_gain": float(local_gain[event_idx]),
    }


def main():
    args = parse_args()
    set_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for node in args.nodes:
        source_path = args.source_dir / f"node_{node:02d}_72h_block_stitched_values.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source CSV: {source_path}")
        timestamps, values = load_node_csv(source_path)
        summaries.append(plot_node(node, timestamps, values, args.output_dir, args.dpi))

    print(f"Saved {len(summaries)} figures to: {args.output_dir}")
    for summary in summaries:
        print(
            f"{summary['path'].name}: event={summary['event_time']}, "
            f"|change|={summary['jump']:.1f}, "
            f"local gain={summary['event_gain']:.2f}"
        )


if __name__ == "__main__":
    main()
