"""Draw paper-ready 72-hour block-stitched prediction curves.

The stitched curve uses non-overlapping 24-step forecast blocks. Each block is
produced from the true historical 24-step input window in the test set; no
predicted values are fed back as future inputs.
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
    OUT_DIR,
    PREDICTION_FILES,
    RAW_START_TIME,
    SAMPLE_MINUTES,
    align_last,
    load_prediction_npz,
    load_test_time_features,
    set_plot_style,
    target_tod_index,
    test_raw_start_index,
)


BLOCK_DIR = OUT_DIR / 'all_nodes_72h_block_stitch'
METRIC_FILE = OUT_DIR / 'node_selection_metrics.csv'
METHODS = ('DeST-LLM', 'ST-LLM', 'LTFP-STSR')
INPUT_LEN = 24
OUTPUT_LEN = 24
NUM_BLOCKS = 36
POINTS_72H = OUTPUT_LEN * NUM_BLOCKS


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


def find_block_start(last_tod_aligned, output_len, num_blocks):
    """Choose a start whose first forecast point is at 00:00."""
    first_target_idx = target_tod_index(last_tod_aligned, horizon=1)
    candidates = np.where(first_target_idx == 0)[0]
    needed = (num_blocks - 1) * output_len + 1
    for idx in candidates:
        if idx + needed <= len(last_tod_aligned):
            return int(idx)
    if needed <= len(last_tod_aligned):
        return 0
    raise ValueError(f'Need {needed} aligned samples, only have {len(last_tod_aligned)}.')


def target_timestamp(test_start_raw_idx, dest_sample_idx, horizon_idx):
    raw_idx = test_start_raw_idx + dest_sample_idx + INPUT_LEN + horizon_idx
    return RAW_START_TIME + np.timedelta64(raw_idx * SAMPLE_MINUTES, 'm').astype(object)


def build_block_series(aligned, node, start_idx, dest_offset, test_start_raw_idx):
    timestamps = []
    block_ids = []
    aligned_sample_ids = []
    horizon_steps = []
    truth_values = []
    pred_values = {method: [] for method in METHODS}

    for block_idx in range(NUM_BLOCKS):
        aligned_idx = start_idx + block_idx * OUTPUT_LEN
        dest_sample_idx = dest_offset + aligned_idx
        for h_idx in range(OUTPUT_LEN):
            timestamps.append(target_timestamp(test_start_raw_idx, dest_sample_idx, h_idx))
            block_ids.append(block_idx + 1)
            aligned_sample_ids.append(aligned_idx)
            horizon_steps.append(h_idx + 1)
            truth_values.append(aligned['DeST-LLM'][0][aligned_idx, node, h_idx])
            for method in METHODS:
                pred_values[method].append(aligned[method][1][aligned_idx, node, h_idx])

    truth = np.asarray(truth_values, dtype=np.float64)
    preds = {method: np.asarray(values, dtype=np.float64) for method, values in pred_values.items()}
    return {
        'timestamps': timestamps,
        'block_ids': np.asarray(block_ids, dtype=np.int32),
        'aligned_sample_ids': np.asarray(aligned_sample_ids, dtype=np.int32),
        'horizon_steps': np.asarray(horizon_steps, dtype=np.int32),
        'truth': truth,
        'preds': preds,
    }


def masked_mae(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs(y_pred[valid] - y_true[valid])))


def gain_against_best_baseline(series):
    dest_mae = masked_mae(series['truth'], series['preds']['DeST-LLM'])
    st_mae = masked_mae(series['truth'], series['preds']['ST-LLM'])
    ltfp_mae = masked_mae(series['truth'], series['preds']['LTFP-STSR'])
    best_baseline = min(st_mae, ltfp_mae)
    best_name = 'ST-LLM' if st_mae <= ltfp_mae else 'LTFP-STSR'
    gain = best_baseline - dest_mae
    gain_pct = gain / max(best_baseline, 1e-8) * 100.0
    return dest_mae, st_mae, ltfp_mae, best_name, gain, gain_pct


def node_title(node, metric_row, gain, gain_pct, dest_mae):
    diff_pct = metric_float(metric_row, 'dest_mae_percentile_all')
    if np.isfinite(diff_pct):
        return (
            f'Node {node} 72-h block-stitched prediction | '
            f'gain={gain:+.2f} MAE ({gain_pct:+.1f}%), '
            f'DeST MAE={dest_mae:.2f}, difficulty={diff_pct:.0f} pct'
        )
    return (
        f'Node {node} 72-h block-stitched prediction | '
        f'gain={gain:+.2f} MAE ({gain_pct:+.1f}%), DeST MAE={dest_mae:.2f}'
    )


def add_block_boundaries(ax, timestamps):
    for idx in range(OUTPUT_LEN, len(timestamps), OUTPUT_LEN):
        ax.axvline(timestamps[idx], color='#D6D6D6', lw=0.45, alpha=0.45, zorder=0)


def plot_node(node, series, out_dir, metric_row):
    dest_mae, st_mae, ltfp_mae, best_name, gain, gain_pct = gain_against_best_baseline(series)
    timestamps = series['timestamps']
    truth = series['truth']

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
        pred = series['preds'][method]
        ax.plot(timestamps, pred, label=method, **CONTINUOUS_STYLE[method])
        err_ax.plot(
            timestamps,
            np.abs(pred - truth),
            label=method,
            color=CONTINUOUS_STYLE[method]['color'],
            linestyle=CONTINUOUS_STYLE[method]['linestyle'],
            lw=1.25,
            alpha=0.95,
        )

    for panel in axes:
        add_block_boundaries(panel, timestamps)
        panel.grid(True, linestyle='--', linewidth=0.6, alpha=0.38)

    ax.set_title(node_title(node, metric_row, gain, gain_pct, dest_mae), pad=12, fontsize=13)
    ax.set_ylabel('Traffic flow')
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

    fig.subplots_adjust(left=0.08, right=0.84, top=0.90, bottom=0.13, hspace=0.10)
    out_path = out_dir / f'node_{node:02d}_72h_block_stitched_flow_error_comparison.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path, {
        'dest_block_mae': dest_mae,
        'st_llm_block_mae': st_mae,
        'ltfp_stsr_block_mae': ltfp_mae,
        'best_baseline_block': best_name,
        'dest_gain_block_mae': gain,
        'dest_gain_block_pct': gain_pct,
    }


def save_node_csv(node, series, out_dir):
    rows = []
    for idx, timestamp in enumerate(series['timestamps']):
        row = {
            'node': int(node),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'block_index': int(series['block_ids'][idx]),
            'aligned_sample_idx': int(series['aligned_sample_ids'][idx]),
            'horizon_step': int(series['horizon_steps'][idx]),
            'ground_truth': float(series['truth'][idx]),
        }
        for method in METHODS:
            row[method] = float(series['preds'][method][idx])
            row[f'{method}_abs_error'] = float(abs(series['preds'][method][idx] - series['truth'][idx]))
        rows.append(row)
    out_path = out_dir / f'node_{node:02d}_72h_block_stitched_values.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_index(rows, path):
    fieldnames = [
        'rank_by_block_gain',
        'node',
        'figure',
        'csv',
        'middle_difficulty_20_80',
        'dest_mae_percentile_all',
        'dest_block_mae',
        'st_llm_block_mae',
        'ltfp_stsr_block_mae',
        'best_baseline_block',
        'dest_gain_block_mae',
        'dest_gain_block_pct',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {method: load_prediction_npz(path) for method, path in PREDICTION_FILES.items()}
    aligned = align_last(arrays)
    common_len = next(iter(aligned.values()))[0].shape[0]
    dest_total_len = arrays['DeST-LLM'][0].shape[0]
    dest_offset = dest_total_len - common_len

    test_start_raw_idx = test_raw_start_index(INPUT_LEN)
    last_tod, _ = load_test_time_features()
    last_tod_aligned = last_tod[-common_len:]
    start_idx = find_block_start(last_tod_aligned, OUTPUT_LEN, NUM_BLOCKS)
    metric_rows = load_metric_rows(METRIC_FILE)

    set_plot_style()
    index_rows = []
    num_nodes = aligned['DeST-LLM'][0].shape[1]
    for node in range(num_nodes):
        series = build_block_series(aligned, node, start_idx, dest_offset, test_start_raw_idx)
        fig_path, metric_values = plot_node(node, series, BLOCK_DIR, metric_rows.get(node))
        csv_path = save_node_csv(node, series, BLOCK_DIR)
        row = {
            'rank_by_block_gain': '',
            'node': int(node),
            'figure': str(fig_path),
            'csv': str(csv_path),
            'middle_difficulty_20_80': metric_rows.get(node, {}).get('middle_difficulty_20_80', ''),
            'dest_mae_percentile_all': metric_rows.get(node, {}).get('dest_mae_percentile_all', ''),
        }
        row.update(metric_values)
        index_rows.append(row)

    index_rows.sort(key=lambda row: row['dest_gain_block_mae'], reverse=True)
    for rank, row in enumerate(index_rows, start=1):
        row['rank_by_block_gain'] = rank

    index_path = BLOCK_DIR / 'block_stitched_figure_index.csv'
    write_index(index_rows, index_path)

    first_series = build_block_series(aligned, 0, start_idx, dest_offset, test_start_raw_idx)
    print(f'Saved {num_nodes} block-stitched node figures to: {BLOCK_DIR}')
    print(f'Saved index to: {index_path}')
    print(
        f'Block stitching: {NUM_BLOCKS} blocks x {OUTPUT_LEN} steps = {POINTS_72H} points; '
        f'first={first_series["timestamps"][0]}, last={first_series["timestamps"][-1]}'
    )
    print('Top nodes by block-stitched MAE gain:')
    for row in index_rows[:10]:
        print(
            f"rank={row['rank_by_block_gain']:02d}, node={int(row['node']):02d}, "
            f"gain={row['dest_gain_block_mae']:+.2f} ({row['dest_gain_block_pct']:+.1f}%), "
            f"DeST={row['dest_block_mae']:.2f}, best_base={row['best_baseline_block']}"
        )


if __name__ == '__main__':
    main()
