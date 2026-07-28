"""Draw station-wise radar comparisons between methods.

This script is independent from visualize_forecast.py. It reads saved
prediction arrays, computes per-node mean MAE/MAPE/RMSE at selected horizons,
and exports one radar figure per metric-horizon pair.
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = (
    ROOT / 'logs' / '2026-06-18-22-38-06-pems07_custom'
    / 'method_radar_comparison_paper_style' / 'dest_predictions.npz'
)
DEFAULT_STLLM = ROOT / 'other result' / 'st-llm-results' / 'st-llm_predictions.npz'
DEFAULT_LTFP_STSR = (
    ROOT / 'other result' / 'ltfp-stsr-results' / 'ltfp-stsr_predictions.npz'
)
DEFAULT_OUT = (
    ROOT / 'logs' / '2026-06-18-22-38-06-pems07_custom'
    / 'method_radar_comparison_paper_style'
)


METHOD_STYLE = {
    'DeST-LLM': dict(color='#4C78A8', lw=1.65, alpha=0.12, marker='o'),
    'ST-LLM': dict(color='#72B7B2', lw=1.45, alpha=0.09, marker='s'),
    'LTFP-STSR': dict(color='#C27C3A', lw=1.45, alpha=0.09, marker='^'),
}
METHOD_LABELS = {'DeST-LLM': 'TDH-LLM'}


def parse_int_list(value):
    return [int(v.strip()) for v in value.split(',') if v.strip()]


def set_plot_style():
    plt.rcParams.update({
        'figure.dpi': 140,
        'savefig.dpi': 600,
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.frameon': True,
        'legend.fontsize': 10.5,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
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
        raise ValueError(
            f'{path} must contain either y_true/y_pred or real/pred. '
            f'Found keys: {data.files}'
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(f'Shape mismatch in {path}: true={y_true.shape}, pred={y_pred.shape}')
    if y_true.ndim != 3:
        raise ValueError(f'Expected [samples, nodes, horizons] in {path}, got {y_true.shape}')
    return y_true.astype(np.float64), y_pred.astype(np.float64)


def align_methods(method_arrays, mode='last'):
    if mode == 'independent':
        return method_arrays
    min_len = min(y_true.shape[0] for y_true, _ in method_arrays.values())
    aligned = {}
    for name, (y_true, y_pred) in method_arrays.items():
        if mode == 'last':
            y_true = y_true[-min_len:]
            y_pred = y_pred[-min_len:]
        elif mode == 'first':
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        else:
            raise ValueError(f'Unknown align mode: {mode}')
        aligned[name] = (y_true, y_pred)
    return aligned


def assert_same_truth(method_arrays):
    names = list(method_arrays)
    base_true = method_arrays[names[0]][0]
    for name in names[1:]:
        other_true = method_arrays[name][0]
        if base_true.shape != other_true.shape or not np.array_equal(base_true, other_true):
            diff = np.abs(base_true - other_true)
            raise ValueError(
                'Aligned ground-truth arrays are not identical. '
                f'{names[0]} vs {name}: mean_abs={diff.mean():.6f}, max_abs={diff.max():.6f}. '
                'Try --align last/first or verify the result files.'
            )


def masked_mean(values, valid, axis):
    numerator = np.sum(np.where(valid, values, 0.0), axis=axis)
    denominator = np.maximum(np.sum(valid, axis=axis), 1)
    return numerator / denominator


def compute_metrics_by_method(method_arrays, horizons):
    metrics = {}
    for method, (y_true, y_pred) in method_arrays.items():
        method_metrics = {}
        for horizon in horizons:
            h_idx = horizon - 1
            true_h = y_true[:, :, h_idx]
            pred_h = y_pred[:, :, h_idx]
            valid = true_h > 0
            err = pred_h - true_h
            abs_err = np.abs(err)
            sq_err = err ** 2
            safe_true = np.where(valid, np.abs(true_h), 1.0)
            method_metrics[horizon] = {
                'MAE': masked_mean(abs_err, valid, axis=0),
                'RMSE': np.sqrt(masked_mean(sq_err, valid, axis=0)),
                'MAPE': masked_mean(abs_err / safe_true, valid, axis=0) * 100.0,
            }
        metrics[method] = method_metrics
    return metrics


def radar_angles(num_nodes):
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    return angles, np.r_[angles, angles[0]]


def nice_limit(values):
    vmax = float(np.nanmax(values)) if np.size(values) else 0.0
    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    raw = vmax * 1.08
    if raw <= 10:
        step = 1
    elif raw <= 30:
        step = 5
    elif raw <= 80:
        step = 10
    else:
        step = 20
    return float(np.ceil(raw / step) * step)


def style_radar_axis(ax, angles, nodes, r_max):
    tick_labels = [str(n) for n in nodes]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor('#F6F8FA')
    ax.set_xticks(angles)
    ax.set_xticklabels(tick_labels, fontsize=8.8)
    ax.set_ylim(0, r_max)
    yticks = np.linspace(0, r_max, 4)[1:]
    ax.set_yticks(yticks)
    tick_fmt = '{:.0f}' if r_max >= 20 else '{:.1f}'
    ax.set_yticklabels([tick_fmt.format(t) for t in yticks], fontsize=8.2)
    ax.set_rlabel_position(235)
    ax.tick_params(axis='x', pad=-3.0)
    ax.tick_params(axis='y', pad=0.2)
    ax.grid(True, color='#B5BDC6', linewidth=0.42, alpha=0.78)
    ax.spines['polar'].set_color('#8E98A3')
    ax.spines['polar'].set_linewidth(0.65)


def plot_one_radar(metrics, metric_name, horizon, out_dir, formats):
    methods = list(metrics)
    first_method = methods[0]
    num_nodes = len(metrics[first_method][horizon][metric_name])
    nodes = np.arange(num_nodes)
    angles, closed_angles = radar_angles(num_nodes)
    all_values = np.concatenate([metrics[m][horizon][metric_name] for m in methods])
    r_max = nice_limit(all_values)
    minute = horizon * 5

    fig, ax = plt.subplots(figsize=(5.05, 4.75), facecolor='white',
                           subplot_kw={'projection': 'polar'})
    style_radar_axis(ax, angles, nodes, r_max)
    for method in methods:
        values = metrics[method][horizon][metric_name]
        closed = np.r_[values, values[0]]
        style = METHOD_STYLE.get(method, dict(color='#6B7280', lw=1.8, alpha=0.08))
        ax.plot(
            closed_angles,
            closed,
            color=style['color'],
            lw=style['lw'],
            marker=style['marker'],
            markersize=4.3,
            markeredgewidth=0.65,
            markeredgecolor='white',
            markerfacecolor=style['color'],
            label=METHOD_LABELS.get(method, method),
            zorder=3,
        )
        ax.fill(
            closed_angles,
            closed,
            color=style['color'],
            alpha=style['alpha'],
            zorder=2,
        )
    ax.set_xlabel(f'{metric_name} ({minute} min)', labelpad=10, fontsize=12)
    leg = ax.legend(
        loc='lower right',
        bbox_to_anchor=(1.48, -0.08),
        borderaxespad=0.0,
        handlelength=1.8,
        handletextpad=0.50,
        labelspacing=0.35,
    )
    leg.get_frame().set_facecolor('#FFFFFF')
    leg.get_frame().set_edgecolor('#C8CDD3')
    leg.get_frame().set_linewidth(0.65)
    leg.get_frame().set_alpha(0.96)
    fig.subplots_adjust(left=0.04, right=0.74, top=0.95, bottom=0.11)
    stem = f'radar_{metric_name}_{minute}min_method_comparison'
    for ext in formats:
        fig.savefig(out_dir / f'{stem}.{ext}', bbox_inches='tight')
    plt.close(fig)


def save_metrics_csv(metrics, out_dir, sample_counts):
    rows = []
    summary_rows = []
    for method, method_metrics in metrics.items():
        for horizon, horizon_metrics in method_metrics.items():
            minute = horizon * 5
            nodes = np.arange(len(next(iter(horizon_metrics.values()))))
            for metric_name, values in horizon_metrics.items():
                for node, value in zip(nodes, values):
                    rows.append({
                        'method': method,
                        'node': int(node),
                        'horizon': int(horizon),
                        'minutes': int(minute),
                        'metric': metric_name,
                        'value': float(value),
                    })
                summary_rows.append({
                    'method': method,
                    'samples': int(sample_counts[method]),
                    'horizon': int(horizon),
                    'minutes': int(minute),
                    'metric': metric_name,
                    'mean': float(np.nanmean(values)),
                    'median': float(np.nanmedian(values)),
                    'min': float(np.nanmin(values)),
                    'max': float(np.nanmax(values)),
                })

    with open(out_dir / 'station_metric_values.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / 'station_metric_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser(description='Draw method-comparison radar plots by station.')
    parser.add_argument('--dest_predictions', default=str(DEFAULT_DEST))
    parser.add_argument('--stllm_predictions', default=str(DEFAULT_STLLM))
    parser.add_argument('--ltfp_stsr_predictions', default=str(DEFAULT_LTFP_STSR))
    parser.add_argument('--out_dir', default=str(DEFAULT_OUT))
    parser.add_argument('--horizons', default='6,12,18,24',
                        help='One-based forecast horizons. Default: 6,12,18,24 = 30/60/90/120 min.')
    parser.add_argument('--align', default='independent', choices=['independent', 'last', 'first'],
                        help='independent uses each method full test samples, matching radar.py; '
                             'last/first aligns sample counts before comparison.')
    parser.add_argument('--formats', default='png')
    args = parser.parse_args()

    horizons = parse_int_list(args.horizons)
    formats = [fmt.strip().lower() for fmt in args.formats.split(',') if fmt.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_arrays = {
        'DeST-LLM': load_prediction_npz(Path(args.dest_predictions)),
        'ST-LLM': load_prediction_npz(Path(args.stllm_predictions)),
        'LTFP-STSR': load_prediction_npz(Path(args.ltfp_stsr_predictions)),
    }
    method_arrays = align_methods(method_arrays, mode=args.align)
    if args.align != 'independent':
        assert_same_truth(method_arrays)
    sample_counts = {method: arrays[0].shape[0] for method, arrays in method_arrays.items()}

    max_horizon = min(y_true.shape[-1] for y_true, _ in method_arrays.values())
    horizons = [h for h in horizons if 1 <= h <= max_horizon]
    if not horizons:
        raise ValueError(f'No valid horizons. Max available horizon is {max_horizon}.')

    set_plot_style()
    metrics = compute_metrics_by_method(method_arrays, horizons)
    save_metrics_csv(metrics, out_dir, sample_counts)
    for metric_name in ('MAE', 'MAPE', 'RMSE'):
        for horizon in horizons:
            plot_one_radar(metrics, metric_name, horizon, out_dir, formats)

    print(f'Saved method-comparison radar figures to: {out_dir}')
    sample_info = ', '.join(
        f'{method}={arrays[0].shape[0]}' for method, arrays in method_arrays.items()
    )
    print(f'Sample mode: {args.align}; {sample_info}')
    print('Generated one independent figure per metric-horizon pair.')


if __name__ == '__main__':
    main()
