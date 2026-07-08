"""Draw 72-hour prediction comparisons for every station.

Each figure contains the 120-min-ahead continuous prediction curve and an
absolute-error panel, making it easier to choose representative stations for
paper figures.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from plot_flow_sequence_comparison import (
    CONTINUOUS_STYLE,
    DATA_TEST,
    OUT_DIR,
    PREDICTION_FILES,
    aligned_target_timestamps,
    align_last,
    choose_72h_start,
    load_prediction_npz,
    load_test_time_features,
    set_plot_style,
    test_raw_start_index,
)


ALL_NODE_DIR = OUT_DIR / 'all_nodes_72h'
METRIC_FILE = OUT_DIR / 'node_selection_metrics.csv'
HORIZON = 24
INPUT_LEN = 24
POINTS_72H = 72 * 12
METHODS = ('DeST-LLM', 'ST-LLM', 'LTFP-STSR')


def load_metric_rows(path):
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return {int(row['node']): row for row in rows}


def metric_float(row, key, default=np.nan):
    if not row or key not in row or row[key] == '':
        return default
    return float(row[key])


def node_title(node, metric_row):
    gain = metric_float(metric_row, 'dest_gain_h24_72h_mae')
    gain_pct = metric_float(metric_row, 'dest_gain_h24_72h_pct')
    diff_pct = metric_float(metric_row, 'dest_mae_percentile_all')
    dest_mae = metric_float(metric_row, 'dest_all_mae')
    if np.isfinite(gain):
        return (
            f'Node {node} continuous 72-h prediction (120 min ahead) | '
            f'72-h gain={gain:+.2f} MAE ({gain_pct:+.1f}%), '
            f'DeST MAE={dest_mae:.2f}, difficulty={diff_pct:.0f} pct'
        )
    return f'Node {node} continuous 72-h prediction (120 min ahead)'


def plot_node(node, aligned, timestamps, out_dir, metric_row):
    h_idx = HORIZON - 1
    start_idx = 0
    end_idx = len(timestamps)
    truth = aligned['DeST-LLM'][0][start_idx:end_idx, node, h_idx]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 5.8),
        sharex=True,
        gridspec_kw={'height_ratios': [3.0, 1.05], 'hspace': 0.08},
        facecolor='white',
    )
    ax, err_ax = axes

    ax.plot(timestamps, truth, label='Ground Truth', **CONTINUOUS_STYLE['Ground Truth'])
    for method in METHODS:
        pred = aligned[method][1][start_idx:end_idx, node, h_idx]
        ax.plot(timestamps, pred, label=method, **CONTINUOUS_STYLE[method])

        abs_err = np.abs(pred - truth)
        err_style = {
            'color': CONTINUOUS_STYLE[method]['color'],
            'linestyle': CONTINUOUS_STYLE[method]['linestyle'],
            'lw': 1.25,
            'alpha': 0.95,
        }
        err_ax.plot(timestamps, abs_err, label=method, **err_style)

    ax.set_title(node_title(node, metric_row), pad=12, fontsize=13)
    ax.set_ylabel('Traffic flow')
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.42)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=10,
        edgecolor='#444444',
        handlelength=2.5,
    )

    err_ax.set_ylabel('Abs. error')
    err_ax.set_xlabel('Timestamp')
    err_ax.set_xlim(timestamps[0], timestamps[-1])
    err_ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    err_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    err_ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.42)

    fig.subplots_adjust(left=0.08, right=0.84, top=0.90, bottom=0.13, hspace=0.10)
    out_path = out_dir / f'node_{node:02d}_72h_120min_ahead_flow_error_comparison.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def write_index(rows, path):
    fieldnames = [
        'rank_by_recommendation',
        'node',
        'figure',
        'recommendation_score',
        'middle_difficulty_20_80',
        'dest_mae_percentile_all',
        'dest_all_mae',
        'dest_gain_all_mae',
        'dest_gain_all_pct',
        'dest_gain_h24_mae',
        'dest_gain_h24_pct',
        'dest_gain_h24_72h_mae',
        'dest_gain_h24_72h_pct',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ALL_NODE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {method: load_prediction_npz(path) for method, path in PREDICTION_FILES.items()}
    aligned = align_last(arrays)
    common_len = next(iter(aligned.values()))[0].shape[0]
    dest_total_len = arrays['DeST-LLM'][0].shape[0]
    dest_aligned_offset = dest_total_len - common_len

    test_start_raw_idx = test_raw_start_index(INPUT_LEN)
    last_tod, _ = load_test_time_features()
    last_tod_aligned = last_tod[-common_len:]
    start_72h = choose_72h_start(last_tod_aligned, HORIZON, POINTS_72H)
    timestamps = aligned_target_timestamps(
        test_start_raw_idx,
        dest_aligned_offset,
        start_72h,
        POINTS_72H,
        HORIZON,
        INPUT_LEN,
    )

    metric_rows = load_metric_rows(METRIC_FILE)
    set_plot_style()

    index_rows = []
    num_nodes = aligned['DeST-LLM'][0].shape[1]
    for node in range(num_nodes):
        fig_path = plot_node(
            node,
            aligned={method: (true[start_72h:start_72h + POINTS_72H],
                              pred[start_72h:start_72h + POINTS_72H])
                     for method, (true, pred) in aligned.items()},
            timestamps=timestamps,
            out_dir=ALL_NODE_DIR,
            metric_row=metric_rows.get(node),
        )
        metric_row = metric_rows.get(node, {})
        index_rows.append({
            'rank_by_recommendation': '',
            'node': int(node),
            'figure': str(fig_path),
            'recommendation_score': metric_row.get('recommendation_score', ''),
            'middle_difficulty_20_80': metric_row.get('middle_difficulty_20_80', ''),
            'dest_mae_percentile_all': metric_row.get('dest_mae_percentile_all', ''),
            'dest_all_mae': metric_row.get('dest_all_mae', ''),
            'dest_gain_all_mae': metric_row.get('dest_gain_all_mae', ''),
            'dest_gain_all_pct': metric_row.get('dest_gain_all_pct', ''),
            'dest_gain_h24_mae': metric_row.get('dest_gain_h24_mae', ''),
            'dest_gain_h24_pct': metric_row.get('dest_gain_h24_pct', ''),
            'dest_gain_h24_72h_mae': metric_row.get('dest_gain_h24_72h_mae', ''),
            'dest_gain_h24_72h_pct': metric_row.get('dest_gain_h24_72h_pct', ''),
        })

    index_rows.sort(
        key=lambda row: float(row['recommendation_score']) if row['recommendation_score'] else -1e9,
        reverse=True,
    )
    for rank, row in enumerate(index_rows, start=1):
        row['rank_by_recommendation'] = rank

    index_path = ALL_NODE_DIR / 'all_node_figure_index.csv'
    write_index(index_rows, index_path)

    print(f'Saved {num_nodes} node figures to: {ALL_NODE_DIR}')
    print(f'Saved figure index to: {index_path}')
    print('Top nodes by recommendation score:')
    for row in index_rows[:10]:
        print(
            f"rank={row['rank_by_recommendation']:02d}, node={int(row['node']):02d}, "
            f"score={float(row['recommendation_score']):.2f}, "
            f"72h_gain={float(row['dest_gain_h24_72h_mae']):+.2f} "
            f"({float(row['dest_gain_h24_72h_pct']):+.1f}%)"
        )


if __name__ == '__main__':
    main()
