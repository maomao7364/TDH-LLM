"""Plot traffic-flow prediction comparisons for selected stations.

The selected stations are determined by DeST-LLM station-wise overall MAE:
the largest-error and smallest-error stations. For each selected station, the
script can draw either a single future-120min sequence or a continuous
multi-hour h-step-ahead curve from the test split.
"""
import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'logs' / '2026-06-18-22-38-06-pems07_custom'
RADAR_DIR = LOG_DIR / 'method_radar_comparison_paper_style'
OUT_DIR = LOG_DIR / 'flow_sequence_comparison'
DATA_TEST = ROOT / 'data' / 'pems07_custom' / 'test.npz'
RAW_FLOW_FILE = ROOT / 'data' / 'pems 07-1' / 'flow_05_06.csv'

RAW_START_TIME = datetime(2021, 5, 1, 0, 0)
SAMPLE_MINUTES = 5
STEPS_PER_DAY = 288
WEEKLY_OFFSET = STEPS_PER_DAY * 7
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2

PREDICTION_FILES = {
    'DeST-LLM': RADAR_DIR / 'dest_predictions.npz',
    'ST-LLM': ROOT / 'other result' / 'st-llm-results' / 'st-llm_predictions.npz',
    'LTFP-STSR': ROOT / 'other result' / 'ltfp-stsr-results' / 'ltfp-stsr_predictions.npz',
}

LINE_STYLE = {
    'Ground Truth': dict(color='#5B5B5B', lw=2.2, marker='o', linestyle='-'),
    'DeST-LLM': dict(color='#4C78A8', lw=2.0, marker='s', linestyle='-'),
    'ST-LLM': dict(color='#72B7B2', lw=1.9, marker='^', linestyle='--'),
    'LTFP-STSR': dict(color='#C27C3A', lw=1.9, marker='D', linestyle='-.'),
}

CONTINUOUS_STYLE = {
    'Ground Truth': dict(color='#5B5B5B', lw=2.1, linestyle='-'),
    'DeST-LLM': dict(color='#4C78A8', lw=1.75, linestyle='-'),
    'ST-LLM': dict(color='#72B7B2', lw=1.65, linestyle='--'),
    'LTFP-STSR': dict(color='#C27C3A', lw=1.65, linestyle='-.'),
}


def set_plot_style():
    plt.rcParams.update({
        'figure.dpi': 140,
        'savefig.dpi': 600,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
        'font.size': 12,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.0,
        'legend.frameon': True,
        'axes.unicode_minus': False,
    })


def load_prediction_npz(path):
    data = np.load(path, allow_pickle=False)
    if {'y_true', 'y_pred'}.issubset(data.files):
        y_true = data['y_true']
        y_pred = data['y_pred']
    elif {'real', 'pred'}.issubset(data.files):
        y_true = data['real']
        y_pred = data['pred']
    else:
        raise ValueError(f'{path} must contain y_true/y_pred or real/pred; found {data.files}')
    if y_true.shape != y_pred.shape:
        raise ValueError(f'Shape mismatch in {path}: true={y_true.shape}, pred={y_pred.shape}')
    return y_true.astype(np.float64), y_pred.astype(np.float64)


def align_last(method_arrays):
    min_len = min(arrays[0].shape[0] for arrays in method_arrays.values())
    aligned = {
        method: (y_true[-min_len:], y_pred[-min_len:])
        for method, (y_true, y_pred) in method_arrays.items()
    }
    names = list(aligned)
    base = aligned[names[0]][0]
    for name in names[1:]:
        other = aligned[name][0]
        if not np.array_equal(base, other):
            diff = np.abs(base - other)
            raise ValueError(
                f'Aligned truth mismatch: {names[0]} vs {name}, '
                f'mean_abs={diff.mean():.6f}, max_abs={diff.max():.6f}'
            )
    return aligned


def masked_mae_by_node(y_true, y_pred):
    valid = y_true > 0
    abs_err = np.abs(y_pred - y_true)
    return np.sum(np.where(valid, abs_err, 0.0), axis=(0, 2)) / np.maximum(
        np.sum(valid, axis=(0, 2)), 1
    )


def sequence_mae(y_true, y_pred, node):
    true_node = y_true[:, node, :]
    pred_node = y_pred[:, node, :]
    valid = true_node > 0
    abs_err = np.abs(pred_node - true_node)
    return np.sum(np.where(valid, abs_err, 0.0), axis=1) / np.maximum(np.sum(valid, axis=1), 1)


def choose_representative_sample(y_true, y_pred, node):
    seq_mae = sequence_mae(y_true, y_pred, node)
    median_mae = np.nanmedian(seq_mae)
    sample_idx = int(np.nanargmin(np.abs(seq_mae - median_mae)))
    return sample_idx, float(seq_mae[sample_idx]), float(median_mae)


def plot_node_sequence(node, sample_idx, sample_mae, median_mae, aligned, out_dir, tag):
    minutes = np.arange(1, aligned['DeST-LLM'][0].shape[-1] + 1) * 5
    truth = aligned['DeST-LLM'][0][sample_idx, node, :]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor='white')
    ax.plot(
        minutes,
        truth,
        label='Ground Truth',
        markevery=3,
        markersize=4.5,
        **LINE_STYLE['Ground Truth'],
    )
    for method in ('DeST-LLM', 'ST-LLM', 'LTFP-STSR'):
        pred = aligned[method][1][sample_idx, node, :]
        ax.plot(
            minutes,
            pred,
            label=method,
            markevery=3,
            markersize=4.0,
            **LINE_STYLE[method],
        )

    ax.set_title(f'Node {node} future 120-min flow prediction')
    ax.set_xlabel('Prediction time (min)')
    ax.set_ylabel('Traffic flow')
    ax.set_xticks(minutes[::3])
    ax.grid(True, linestyle='--', linewidth=0.65, alpha=0.45)
    ax.legend(loc='best', fontsize=10, edgecolor='black')
    ax.text(
        0.01,
        0.97,
        f'DeST-LLM sequence MAE={sample_mae:.2f}; median={median_mae:.2f}',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox=dict(facecolor='white', edgecolor='#CCCCCC', alpha=0.85, pad=3),
    )
    fig.tight_layout()
    out_path = out_dir / f'node_{node}_{tag}_future120min_flow_comparison.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_sequence_csv(node, sample_idx, aligned, out_dir, tag):
    minutes = np.arange(1, aligned['DeST-LLM'][0].shape[-1] + 1) * 5
    rows = []
    for h_idx, minute in enumerate(minutes):
        row = {
            'node': int(node),
            'sample_idx_aligned': int(sample_idx),
            'step': int(h_idx + 1),
            'minutes': int(minute),
            'ground_truth': float(aligned['DeST-LLM'][0][sample_idx, node, h_idx]),
        }
        for method in ('DeST-LLM', 'ST-LLM', 'LTFP-STSR'):
            row[method] = float(aligned[method][1][sample_idx, node, h_idx])
        rows.append(row)
    out_path = out_dir / f'node_{node}_{tag}_future120min_flow_values.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def load_test_time_features():
    data = np.load(DATA_TEST, allow_pickle=False)
    x = data['x']
    # Last input timestamp for each test window. Values are normalized.
    tod = x[:, -1, 0, 1]
    dow = x[:, -1, 0, 2]
    return tod, dow


def target_tod_index(last_tod, horizon):
    last_idx = np.rint(last_tod * 288).astype(int) % 288
    return (last_idx + horizon) % 288


def choose_72h_start(last_tod_aligned, horizon, points):
    target_idx = target_tod_index(last_tod_aligned, horizon)
    candidates = np.where(target_idx == 0)[0]
    for idx in candidates:
        if idx + points <= len(last_tod_aligned):
            return int(idx)
    if points <= len(last_tod_aligned):
        return 0
    raise ValueError(f'Need {points} aligned samples, only have {len(last_tod_aligned)}.')


def count_raw_flow_steps():
    with open(RAW_FLOW_FILE, 'r', encoding='utf-8') as f:
        return max(sum(1 for _ in f) - 1, 0)


def test_raw_start_index(input_len):
    raw_steps = count_raw_flow_steps()
    t_new = raw_steps - WEEKLY_OFFSET
    val_end = int(t_new * (TRAIN_RATIO + VAL_RATIO))
    return WEEKLY_OFFSET + val_end - input_len


def target_timestamp(test_start_raw_idx, dest_sample_idx, horizon, input_len):
    raw_idx = test_start_raw_idx + dest_sample_idx + input_len + horizon - 1
    return RAW_START_TIME + timedelta(minutes=raw_idx * SAMPLE_MINUTES)


def aligned_target_timestamps(test_start_raw_idx, dest_offset, start_idx, points, horizon, input_len):
    return [
        target_timestamp(test_start_raw_idx, dest_offset + idx, horizon, input_len)
        for idx in range(start_idx, start_idx + points)
    ]


def plot_node_continuous(node, aligned, out_dir, tag, start_idx, points, horizon, timestamps):
    h_idx = horizon - 1
    end_idx = start_idx + points
    truth = aligned['DeST-LLM'][0][start_idx:end_idx, node, h_idx]

    fig, ax = plt.subplots(figsize=(12.0, 4.8), facecolor='white')
    ax.plot(timestamps, truth, label='Ground Truth', **CONTINUOUS_STYLE['Ground Truth'])
    for method in ('DeST-LLM', 'ST-LLM', 'LTFP-STSR'):
        pred = aligned[method][1][start_idx:end_idx, node, h_idx]
        ax.plot(timestamps, pred, label=method, **CONTINUOUS_STYLE[method])

    ax.set_title(f'Node {node} continuous 72-h flow prediction ({horizon * 5} min ahead)', pad=16)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Traffic flow')
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax.grid(True, linestyle='--', linewidth=0.65, alpha=0.45)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=10,
        edgecolor='black',
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    out_path = out_dir / f'node_{node}_{tag}_72h_{horizon * 5}min_ahead_flow_comparison.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_continuous_csv(
    node, aligned, out_dir, tag, start_idx, points, horizon, last_tod_aligned, timestamps
):
    h_idx = horizon - 1
    end_idx = start_idx + points
    target_idx = target_tod_index(last_tod_aligned[start_idx:end_idx], horizon)
    rows = []
    for local_idx, sample_idx in enumerate(range(start_idx, end_idx)):
        row = {
            'node': int(node),
            'aligned_sample_idx': int(sample_idx),
            'timestamp': timestamps[local_idx].strftime('%Y-%m-%d %H:%M:%S'),
            'relative_hour': float(local_idx * 5 / 60.0),
            'horizon': int(horizon),
            'ahead_minutes': int(horizon * 5),
            'target_tod_step': int(target_idx[local_idx]),
            'ground_truth': float(aligned['DeST-LLM'][0][sample_idx, node, h_idx]),
        }
        for method in ('DeST-LLM', 'ST-LLM', 'LTFP-STSR'):
            row[method] = float(aligned[method][1][sample_idx, node, h_idx])
        rows.append(row)
    out_path = out_dir / f'node_{node}_{tag}_72h_{horizon * 5}min_ahead_flow_values.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {method: load_prediction_npz(path) for method, path in PREDICTION_FILES.items()}
    aligned = align_last(arrays)
    common_len = next(iter(aligned.values()))[0].shape[0]
    dest_total_len = arrays['DeST-LLM'][0].shape[0]
    dest_aligned_offset = dest_total_len - common_len
    input_len = 24
    test_start_raw_idx = test_raw_start_index(input_len)
    last_tod, _ = load_test_time_features()
    last_tod_aligned = last_tod[-common_len:]

    dest_full_true, dest_full_pred = arrays['DeST-LLM']
    node_mae = masked_mae_by_node(dest_full_true, dest_full_pred)
    min_node = int(np.nanargmin(node_mae))
    max_node = int(np.nanargmax(node_mae))

    set_plot_style()
    summary_rows = []
    horizon = 24
    points_72h = 72 * 12
    start_72h = choose_72h_start(last_tod_aligned, horizon, points_72h)
    timestamps_72h = aligned_target_timestamps(
        test_start_raw_idx,
        dest_aligned_offset,
        start_72h,
        points_72h,
        horizon,
        input_len,
    )
    for node, tag in ((max_node, 'max_error'), (min_node, 'min_error')):
        sample_idx, sample_mae, median_mae = choose_representative_sample(
            aligned['DeST-LLM'][0], aligned['DeST-LLM'][1], node
        )
        fig_path = plot_node_sequence(node, sample_idx, sample_mae, median_mae, aligned, OUT_DIR, tag)
        csv_path = save_sequence_csv(node, sample_idx, aligned, OUT_DIR, tag)
        fig72_path = plot_node_continuous(
            node, aligned, OUT_DIR, tag, start_72h, points_72h, horizon, timestamps_72h
        )
        csv72_path = save_continuous_csv(
            node, aligned, OUT_DIR, tag, start_72h, points_72h, horizon,
            last_tod_aligned, timestamps_72h
        )
        summary_rows.append({
            'tag': tag,
            'node': int(node),
            'dest_overall_node_mae': float(node_mae[node]),
            'sample_idx_aligned': int(sample_idx),
            'dest_sequence_mae': float(sample_mae),
            'dest_node_median_sequence_mae': float(median_mae),
            'figure': str(fig_path),
            'csv': str(csv_path),
            'continuous_72h_start_aligned': int(start_72h),
            'continuous_72h_start_timestamp': timestamps_72h[0].strftime('%Y-%m-%d %H:%M:%S'),
            'continuous_72h_end_timestamp': timestamps_72h[-1].strftime('%Y-%m-%d %H:%M:%S'),
            'continuous_72h_horizon': int(horizon),
            'continuous_72h_figure': str(fig72_path),
            'continuous_72h_csv': str(csv72_path),
        })

    summary_path = OUT_DIR / 'selected_nodes_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f'Saved flow comparison figures to: {OUT_DIR}')
    for row in summary_rows:
        print(
            f"{row['tag']}: node={row['node']}, overall_MAE={row['dest_overall_node_mae']:.4f}, "
            f"sample={row['sample_idx_aligned']}, sequence_MAE={row['dest_sequence_mae']:.4f}"
        )


if __name__ == '__main__':
    main()
