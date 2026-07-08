"""Rank stations for qualitative prediction-curve visualization.

The script compares DeST-LLM with available baselines on the common aligned
test suffix and writes per-node metrics for choosing representative stations.
"""
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'logs' / '2026-06-18-22-38-06-pems07_custom'
RADAR_DIR = LOG_DIR / 'method_radar_comparison_paper_style'
OUT_DIR = LOG_DIR / 'flow_sequence_comparison'
DATA_TEST = ROOT / 'data' / 'pems07_custom' / 'test.npz'

PREDICTION_FILES = {
    'DeST-LLM': RADAR_DIR / 'dest_predictions.npz',
    'ST-LLM': ROOT / 'other result' / 'st-llm-results' / 'st-llm_predictions.npz',
    'LTFP-STSR': ROOT / 'other result' / 'ltfp-stsr-results' / 'ltfp-stsr_predictions.npz',
}

HORIZON = 24
POINTS_72H = 72 * 12
STEPS_PER_DAY = 288


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


def load_last_tod_aligned(common_len):
    x = np.load(DATA_TEST, allow_pickle=False)['x']
    return x[-common_len:, -1, 0, 1].astype(np.float64)


def target_tod_index(last_tod, horizon):
    last_idx = np.rint(last_tod * STEPS_PER_DAY).astype(int) % STEPS_PER_DAY
    return (last_idx + horizon) % STEPS_PER_DAY


def choose_72h_start(last_tod_aligned, horizon, points):
    target_idx = target_tod_index(last_tod_aligned, horizon)
    candidates = np.where(target_idx == 0)[0]
    for idx in candidates:
        if idx + points <= len(last_tod_aligned):
            return int(idx)
    if points <= len(last_tod_aligned):
        return 0
    raise ValueError(f'Need {points} aligned samples, only have {len(last_tod_aligned)}.')


def masked_metrics(y_true, y_pred, axes):
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    abs_err = np.abs(y_pred - y_true)
    sq_err = abs_err ** 2
    safe_true = np.where(valid, np.abs(y_true), 1.0)
    denom = np.maximum(np.sum(valid, axis=axes), 1)
    mae = np.sum(np.where(valid, abs_err, 0.0), axis=axes) / denom
    rmse = np.sqrt(np.sum(np.where(valid, sq_err, 0.0), axis=axes) / denom)
    mape = np.sum(np.where(valid, abs_err / safe_true, 0.0), axis=axes) / denom * 100.0
    return mae, rmse, mape


def per_node_metrics(y_true, y_pred):
    # y: [samples, nodes, horizons]
    return masked_metrics(y_true, y_pred, axes=(0, 2))


def per_node_horizon_metrics(y_true, y_pred, horizon_idx, sample_slice=None):
    if sample_slice is None:
        true_sel = y_true[:, :, horizon_idx]
        pred_sel = y_pred[:, :, horizon_idx]
    else:
        true_sel = y_true[sample_slice, :, horizon_idx]
        pred_sel = y_pred[sample_slice, :, horizon_idx]
    return masked_metrics(true_sel, pred_sel, axes=0)


def percentile_ranks(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return ranks / (len(values) - 1) * 100.0


def add_method_metrics(row, prefix, metrics_by_method):
    for method, safe_name in (
        ('DeST-LLM', 'dest'),
        ('ST-LLM', 'st_llm'),
        ('LTFP-STSR', 'ltfp_stsr'),
    ):
        mae, rmse, mape = metrics_by_method[method]
        row[f'{safe_name}_{prefix}_mae'] = float(mae)
        row[f'{safe_name}_{prefix}_rmse'] = float(rmse)
        row[f'{safe_name}_{prefix}_mape'] = float(mape)


def gain_values(dest_mae, st_mae, ltfp_mae):
    best_baseline = min(st_mae, ltfp_mae)
    best_name = 'ST-LLM' if st_mae <= ltfp_mae else 'LTFP-STSR'
    gain = best_baseline - dest_mae
    gain_pct = gain / max(best_baseline, 1e-8) * 100.0
    return best_baseline, best_name, gain, gain_pct


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {method: load_prediction_npz(path) for method, path in PREDICTION_FILES.items()}
    aligned = align_last(arrays)
    common_len = next(iter(aligned.values()))[0].shape[0]
    last_tod_aligned = load_last_tod_aligned(common_len)
    start_72h = choose_72h_start(last_tod_aligned, HORIZON, POINTS_72H)
    sample_72h = slice(start_72h, start_72h + POINTS_72H)
    h_idx = HORIZON - 1

    metrics_all = {}
    metrics_h24 = {}
    metrics_72h_h24 = {}
    for method, (y_true, y_pred) in aligned.items():
        metrics_all[method] = per_node_metrics(y_true, y_pred)
        metrics_h24[method] = per_node_horizon_metrics(y_true, y_pred, h_idx)
        metrics_72h_h24[method] = per_node_horizon_metrics(y_true, y_pred, h_idx, sample_72h)

    dest_all_mae = metrics_all['DeST-LLM'][0]
    difficulty_pct = percentile_ranks(dest_all_mae)
    rows = []
    for node in range(dest_all_mae.shape[0]):
        row = {
            'node': int(node),
            'dest_mae_percentile_all': float(difficulty_pct[node]),
            'middle_difficulty_20_80': bool(20.0 <= difficulty_pct[node] <= 80.0),
        }

        for prefix, pack in (
            ('all', metrics_all),
            ('h24', metrics_h24),
            ('h24_72h', metrics_72h_h24),
        ):
            node_metrics = {
                method: tuple(metric[node] for metric in pack[method])
                for method in pack
            }
            add_method_metrics(row, prefix, node_metrics)
            base, base_name, gain, gain_pct = gain_values(
                row[f'dest_{prefix}_mae'],
                row[f'st_llm_{prefix}_mae'],
                row[f'ltfp_stsr_{prefix}_mae'],
            )
            row[f'best_baseline_{prefix}'] = base_name
            row[f'best_baseline_{prefix}_mae'] = float(base)
            row[f'dest_gain_{prefix}_mae'] = float(gain)
            row[f'dest_gain_{prefix}_pct'] = float(gain_pct)

        row['recommendation_score'] = float(
            0.35 * row['dest_gain_all_pct']
            + 0.35 * row['dest_gain_h24_pct']
            + 0.30 * row['dest_gain_h24_72h_pct']
        )
        rows.append(row)

    fieldnames = list(rows[0])
    metrics_path = OUT_DIR / 'node_selection_metrics.csv'
    with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    candidates = [
        row for row in rows
        if row['middle_difficulty_20_80']
        and row['dest_gain_all_mae'] > 0
        and row['dest_gain_h24_mae'] > 0
        and row['dest_gain_h24_72h_mae'] > 0
    ]
    candidates.sort(key=lambda r: r['recommendation_score'], reverse=True)
    recommendation_path = OUT_DIR / 'node_selection_recommendations.csv'
    with open(recommendation_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    print(f'Aligned samples: {common_len}')
    print(f'72h aligned start: {start_72h}, points: {POINTS_72H}, horizon: {HORIZON}')
    print(f'Saved all node metrics: {metrics_path}')
    print(f'Saved recommended candidates: {recommendation_path}')
    print('Top recommended nodes:')
    for row in candidates[:10]:
        print(
            f"node={row['node']:02d}, score={row['recommendation_score']:.2f}, "
            f"dest_all={row['dest_all_mae']:.2f}, gain_all={row['dest_gain_all_mae']:.2f} "
            f"({row['dest_gain_all_pct']:.1f}%), "
            f"gain_h24={row['dest_gain_h24_mae']:.2f} "
            f"({row['dest_gain_h24_pct']:.1f}%), "
            f"gain_72h={row['dest_gain_h24_72h_mae']:.2f} "
            f"({row['dest_gain_h24_72h_pct']:.1f}%)"
        )


if __name__ == '__main__':
    main()
