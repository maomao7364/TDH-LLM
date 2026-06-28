"""Visualize node-level forecasts and errors from a trained checkpoint.

This script does not change training. It reloads best_model.pth, runs inference
on the test split, saves prediction arrays, and exports station-level figures.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_root_dir = os.path.dirname(os.path.abspath(__file__))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from dataset_checks import validate_dataset_shapes
from model.dest_llm import DeSTLLM
from util import build_correlation_adj, load_adj_from_npy, load_adj_from_pkl, load_dataset, topk_adj

try:
    from experiments.module_ablation_model import ModuleAblationDeSTLLM
except Exception:
    ModuleAblationDeSTLLM = None




def parse_int_list(value):
    if value is None or value == '':
        return []
    return [int(v.strip()) for v in value.split(',') if v.strip()]


def load_config(checkpoint, explicit_config=None):
    config_path = explicit_config
    ckpt_dir = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
    if config_path is None:
        auto_config = os.path.join(ckpt_dir, 'model_args.json')
        if os.path.exists(auto_config):
            config_path = auto_config
    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        print(f'Loaded config: {config_path}')
    return cfg, ckpt_dir


def resolve_data_dir(data):
    if os.path.isabs(data):
        return data
    return os.path.join(_root_dir, 'data', data)


def resolve_gpt2_path(path_value):
    if path_value and os.path.exists(path_value):
        return path_value
    local_snapshot = os.path.join(
        _root_dir, 'snapshots', '607a30d783dfa663caf39e06633721c8d4cfcd7e'
    )
    if os.path.exists(local_snapshot):
        return local_snapshot
    return path_value


def load_adjacency(args, data_dir, dataloader):
    if args.adj and os.path.exists(args.adj):
        adj = load_adj_from_npy(args.adj) if args.adj.endswith('.npy') else load_adj_from_pkl(args.adj)
        print(f'Loaded adjacency from {args.adj}, shape={adj.shape}')
    else:
        auto_adj_npy = os.path.join(data_dir, 'adj.npy')
        auto_adj_pkl = os.path.join(data_dir, 'adj.pkl')
        if os.path.exists(auto_adj_npy):
            adj = load_adj_from_npy(auto_adj_npy)
            print(f'Loaded adjacency from {auto_adj_npy}, shape={adj.shape}')
        elif os.path.exists(auto_adj_pkl):
            adj = load_adj_from_pkl(auto_adj_pkl)
            print(f'Loaded adjacency from {auto_adj_pkl}, shape={adj.shape}')
        else:
            adj = build_correlation_adj(dataloader['x_train'])
            print(f'Built correlation adjacency, shape={adj.shape}')
    if args.topk > 0:
        adj = topk_adj(adj, args.topk)
        print(f'Applied top-{args.topk} adjacency sparsification.')
    return adj


def apply_reference_mode(x, y, reference_mode):
    """x: [B,C,N,T], y: [B,C_y,N,H]. Return masked x and future refs."""
    x = x.clone()
    future_daily = y[:, 1, :, :] if y.size(1) > 1 else None
    future_weekly = y[:, 2, :, :] if y.size(1) > 2 else None
    if reference_mode in ('no_daily', 'no_refs') and x.size(1) > 3:
        x[:, 3, :, :] = 0
        future_daily = None
    if reference_mode in ('no_weekly', 'no_refs') and x.size(1) > 4:
        x[:, 4, :, :] = 0
        future_weekly = None
    return x, future_daily, future_weekly


def build_model(args, cfg, adj, device):
    common = dict(
        adj=adj,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        gpt_layers=args.gpt_layers,
        U=args.U,
        time_bins=args.time_bins,
        time_dim=args.time_dim,
        gpt_dim=args.gpt_dim,
        trend_kernel=args.trend_kernel,
        period_kernel=args.period_kernel,
        tcdg_layers=args.tcdg_layers,
        dropout=args.dropout,
        gpt2_path=args.gpt2_path,
    )
    module_variant = cfg.get('module_variant', args.module_variant)
    if module_variant:
        if ModuleAblationDeSTLLM is None:
            raise RuntimeError('ModuleAblationDeSTLLM is unavailable, cannot load module ablation checkpoint.')
        print(f'Building module ablation model: {module_variant}')
        return ModuleAblationDeSTLLM(module_variant=module_variant, **common).to(device)
    use_horizon_decoder = cfg.get('use_horizon_decoder', args.use_horizon_decoder)
    print(f'Building DeSTLLM model, use_horizon_decoder={use_horizon_decoder}')
    return DeSTLLM(use_horizon_decoder=use_horizon_decoder, **common).to(device)


def infer_predictions(args, cfg):
    device = torch.device(args.device)
    data_dir = resolve_data_dir(args.data)
    dataloader = load_dataset(
        data_dir, args.batch_size, args.batch_size, args.batch_size, num_workers=0
    )
    shape_info = validate_dataset_shapes(
        dataloader,
        input_len=args.input_len,
        output_len=args.output_len,
        input_dim=args.input_dim,
        num_nodes=args.num_nodes,
        target_dim=3 if args.input_dim >= 5 else 2,
    )
    print(f'Data shape: {shape_info}')
    scaler = dataloader['scaler']
    adj = load_adjacency(args, data_dir, dataloader)

    model = build_model(args, cfg, adj, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f'Checkpoint loaded: {args.checkpoint}')

    real_y = torch.as_tensor(dataloader['y_test'], dtype=torch.float32, device=device)
    real_y = real_y.transpose(1, 3)[:, 0, :, :]  # [S,N,H], original scale
    preds = []
    reference_mode = cfg.get('reference_mode', args.reference_mode)
    use_amp = (device.type == 'cuda') and (not args.no_amp)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for x, y in dataloader['test_loader']:
            test_x = x.float().to(device, non_blocking=True).transpose(1, 3)
            test_y = y.float().to(device, non_blocking=True).transpose(1, 3)
            test_x, future_daily, future_weekly = apply_reference_mode(
                test_x, test_y, reference_mode
            )
            out = model(
                test_x, future_daily=future_daily, future_weekly=future_weekly
            ).transpose(1, 3).squeeze(1)
            preds.append(out)

    y_hat = torch.cat(preds, dim=0)[:real_y.size(0), ...]
    y_pred = scaler.inverse_transform(y_hat).detach().cpu().numpy()
    y_true = real_y.detach().cpu().numpy()
    return y_true, y_pred


def masked_mean(values, mask, axis):
    return np.sum(np.where(mask, values, 0.0), axis=axis) / np.maximum(np.sum(mask, axis=axis), 1)


def compute_metrics(y_true, y_pred):
    valid = y_true > 0
    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2

    per_node_mae = masked_mean(abs_err, valid, axis=(0, 2))
    per_node_rmse = np.sqrt(masked_mean(sq_err, valid, axis=(0, 2)))
    per_node_wmape = np.sum(np.where(valid, abs_err, 0.0), axis=(0, 2)) / np.maximum(
        np.sum(np.where(valid, np.abs(y_true), 0.0), axis=(0, 2)), 1e-8
    )

    per_horizon_mae = masked_mean(abs_err, valid, axis=(0, 1))
    per_horizon_rmse = np.sqrt(masked_mean(sq_err, valid, axis=(0, 1)))
    per_horizon_wmape = np.sum(np.where(valid, abs_err, 0.0), axis=(0, 1)) / np.maximum(
        np.sum(np.where(valid, np.abs(y_true), 0.0), axis=(0, 1)), 1e-8
    )
    node_horizon_mae = masked_mean(abs_err, valid, axis=0)  # [N,H]
    return dict(
        abs_err=abs_err,
        err=err,
        per_node_mae=per_node_mae,
        per_node_rmse=per_node_rmse,
        per_node_wmape=per_node_wmape,
        per_horizon_mae=per_horizon_mae,
        per_horizon_rmse=per_horizon_rmse,
        per_horizon_wmape=per_horizon_wmape,
        node_horizon_mae=node_horizon_mae,
    )


def set_plot_style():
    plt.rcParams.update({
        'figure.dpi': 140,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
    })


def save_csvs(out_dir, metrics):
    node_df = pd.DataFrame({
        'node': np.arange(len(metrics['per_node_mae'])),
        'mae': metrics['per_node_mae'],
        'rmse': metrics['per_node_rmse'],
        'wmape': metrics['per_node_wmape'],
    })
    horizon_df = pd.DataFrame({
        'horizon': np.arange(1, len(metrics['per_horizon_mae']) + 1),
        'minutes': np.arange(1, len(metrics['per_horizon_mae']) + 1) * 5,
        'mae': metrics['per_horizon_mae'],
        'rmse': metrics['per_horizon_rmse'],
        'wmape': metrics['per_horizon_wmape'],
    })
    node_df.to_csv(os.path.join(out_dir, 'per_node_metrics.csv'), index=False)
    horizon_df.to_csv(os.path.join(out_dir, 'per_horizon_metrics.csv'), index=False)


def plot_forecast_grid(y_true, y_pred, nodes, horizons, max_points, out_dir):
    nrows = len(nodes)
    ncols = len(horizons)
    points = min(max_points, y_true.shape[0])
    x_axis = np.arange(points)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.4 * nrows), squeeze=False)
    for r, node in enumerate(nodes):
        for c, horizon in enumerate(horizons):
            h_idx = horizon - 1
            ax = axes[r][c]
            ax.plot(x_axis, y_true[:points, node, h_idx], color='#111827', lw=1.3, label='True')
            ax.plot(x_axis, y_pred[:points, node, h_idx], color='#2563EB', lw=1.2, label='Pred')
            ax.set_title(f'Node {node}, {horizon * 5} min')
            ax.set_xlabel('Test sample')
            ax.set_ylabel('Flow')
            if r == 0 and c == 0:
                ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'station_forecast_grid.png'), bbox_inches='tight')
    plt.close(fig)


def plot_error_grid(abs_err, nodes, horizons, max_points, out_dir):
    nrows = len(nodes)
    ncols = len(horizons)
    points = min(max_points, abs_err.shape[0])
    x_axis = np.arange(points)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.2 * nrows), squeeze=False)
    for r, node in enumerate(nodes):
        for c, horizon in enumerate(horizons):
            h_idx = horizon - 1
            ax = axes[r][c]
            ax.plot(x_axis, abs_err[:points, node, h_idx], color='#DC2626', lw=1.0)
            ax.fill_between(x_axis, abs_err[:points, node, h_idx], color='#FCA5A5', alpha=0.35)
            ax.set_title(f'Abs error: node {node}, {horizon * 5} min')
            ax.set_xlabel('Test sample')
            ax.set_ylabel('Abs error')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'station_error_grid.png'), bbox_inches='tight')
    plt.close(fig)


def plot_node_metrics(metrics, out_dir, topk):
    nodes = np.arange(len(metrics['per_node_mae']))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.bar(nodes, metrics['per_node_mae'], color='#3B82F6', width=0.78)
    ax.set_xlabel('Node')
    ax.set_ylabel('MAE')
    ax.set_title('Per-node MAE on test set')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'per_node_mae_all.png'), bbox_inches='tight')
    plt.close(fig)

    order = np.argsort(metrics['per_node_mae'])[::-1][:topk]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(np.arange(len(order)), metrics['per_node_mae'][order], color='#F97316')
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([str(i) for i in order], rotation=45, ha='right')
    ax.set_xlabel('Node')
    ax.set_ylabel('MAE')
    ax.set_title(f'Top-{len(order)} nodes by MAE')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'per_node_mae_top.png'), bbox_inches='tight')
    plt.close(fig)


def plot_node_radar(metrics, out_dir):
    nodes = np.arange(len(metrics['per_node_mae']))
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    mae = metrics['per_node_mae']
    closed_mae = np.r_[mae, mae[0]]
    fig, ax = plt.subplots(figsize=(9.2, 9.2), subplot_kw={'projection': 'polar'})
    ax.plot(closed_angles, closed_mae, color='#2563EB', lw=1.5, label='MAE')
    ax.fill(closed_angles, closed_mae, color='#93C5FD', alpha=0.28)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([str(n) for n in nodes], fontsize=6)
    ax.set_ylim(0, max(float(np.nanmax(mae)) * 1.12, 1e-6))
    ax.set_title('Per-node MAE radar', pad=24)
    ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.12))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'per_node_mae_radar.png'), bbox_inches='tight')
    plt.close(fig)

    series = [
        ('MAE', metrics['per_node_mae'], '#2563EB'),
        ('RMSE', metrics['per_node_rmse'], '#7C3AED'),
        ('WMAPE', metrics['per_node_wmape'], '#059669'),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 9.2), subplot_kw={'projection': 'polar'})
    for label, values, color in series:
        denom = max(float(np.nanmax(values)), 1e-8)
        scaled = values / denom
        ax.plot(closed_angles, np.r_[scaled, scaled[0]], lw=1.35, color=color, label=f'{label} / max')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([str(n) for n in nodes], fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_title('Normalized per-node error radar', pad=24)
    ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.14))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'per_node_error_radar_normalized.png'), bbox_inches='tight')
    plt.close(fig)


def plot_horizon_metrics(metrics, out_dir):
    horizons = np.arange(1, len(metrics['per_horizon_mae']) + 1)
    minutes = horizons * 5
    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    ax1.plot(minutes, metrics['per_horizon_mae'], marker='o', color='#2563EB', label='MAE')
    ax1.plot(minutes, metrics['per_horizon_rmse'], marker='s', color='#7C3AED', label='RMSE')
    ax1.set_xlabel('Forecast horizon (min)')
    ax1.set_ylabel('Error')
    ax2 = ax1.twinx()
    ax2.plot(minutes, metrics['per_horizon_wmape'], marker='^', color='#059669', label='WMAPE')
    ax2.set_ylabel('WMAPE')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    ax1.set_title('Error by forecast horizon')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'per_horizon_metrics.png'), bbox_inches='tight')
    plt.close(fig)


def plot_node_horizon_heatmap(metrics, out_dir):
    mat = metrics['node_horizon_mae']
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(mat, aspect='auto', cmap='viridis')
    ax.set_xlabel('Horizon step')
    ax.set_ylabel('Node')
    ax.set_title('Node x horizon MAE heatmap')
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(np.arange(1, mat.shape[1] + 1), fontsize=7)
    fig.colorbar(im, ax=ax, label='MAE')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'node_horizon_mae_heatmap.png'), bbox_inches='tight')
    plt.close(fig)


def plot_error_distribution(metrics, out_dir):
    values = metrics['err'].reshape(-1)
    values = values[np.isfinite(values)]
    if values.size > 300000:
        rng = np.random.default_rng(6666)
        values = rng.choice(values, size=300000, replace=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=80, color='#64748B', alpha=0.9)
    ax.axvline(0, color='#DC2626', lw=1.2)
    ax.set_xlabel('Prediction error (Pred - True)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction error distribution')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'error_distribution.png'), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize DeST-LLM forecasts by station.')
    parser.add_argument('--checkpoint', required=True, help='Path to best_model.pth')
    parser.add_argument('--config', default=None, help='Path to model_args.json; auto-detected by default')
    parser.add_argument('--out_dir', default=None, help='Output directory for figures')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data', default=None)
    parser.add_argument('--adj', default=None)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nodes', default='0,1,2', help='Comma-separated node ids for line plots')
    parser.add_argument('--horizons', default='3,6,12,24', help='Comma-separated horizon steps')
    parser.add_argument('--max_points', type=int, default=288)
    parser.add_argument('--top_error_nodes', type=int, default=20)
    parser.add_argument('--reference_mode', default='full_refs',
                        choices=['full_refs', 'no_daily', 'no_weekly', 'no_refs'])
    parser.add_argument('--module_variant', default=None)
    parser.add_argument('--use_horizon_decoder', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true')

    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--num_nodes', type=int, default=50)
    parser.add_argument('--input_len', type=int, default=24)
    parser.add_argument('--output_len', type=int, default=24)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--time_dim', type=int, default=32)
    parser.add_argument('--gpt_dim', type=int, default=768)
    parser.add_argument('--gpt_layers', type=int, default=4)
    parser.add_argument('--U', type=int, default=1)
    parser.add_argument('--time_bins', type=int, default=288)
    parser.add_argument('--tcdg_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.20)
    parser.add_argument('--trend_kernel', type=int, default=5)
    parser.add_argument('--period_kernel', type=int, default=7)
    parser.add_argument('--gpt2_path', default=None)
    args = parser.parse_args()

    cfg, ckpt_dir = load_config(args.checkpoint, args.config)
    for key, value in cfg.items():
        if hasattr(args, key) and key not in ('checkpoint', 'config', 'out_dir', 'device'):
            setattr(args, key, value)
    if args.data is None:
        args.data = cfg.get('data', 'pems07_custom')
    args.gpt2_path = resolve_gpt2_path(args.gpt2_path)

    out_dir = args.out_dir or os.path.join(ckpt_dir, 'visualizations')
    os.makedirs(out_dir, exist_ok=True)

    nodes = parse_int_list(args.nodes)
    horizons = parse_int_list(args.horizons)
    nodes = [n for n in nodes if 0 <= n < args.num_nodes]
    horizons = [h for h in horizons if 1 <= h <= args.output_len]
    if not nodes:
        nodes = [0]
    if not horizons:
        horizons = [args.output_len]

    set_plot_style()
    y_true, y_pred = infer_predictions(args, cfg)
    metrics = compute_metrics(y_true, y_pred)
    np.savez_compressed(
        os.path.join(out_dir, 'test_predictions.npz'),
        y_true=y_true,
        y_pred=y_pred,
        abs_error=metrics['abs_err'],
    )
    save_csvs(out_dir, metrics)
    plot_forecast_grid(y_true, y_pred, nodes, horizons, args.max_points, out_dir)
    plot_error_grid(metrics['abs_err'], nodes, horizons, args.max_points, out_dir)
    plot_node_metrics(metrics, out_dir, args.top_error_nodes)
    plot_node_radar(metrics, out_dir)
    plot_horizon_metrics(metrics, out_dir)
    plot_node_horizon_heatmap(metrics, out_dir)
    plot_error_distribution(metrics, out_dir)
    print(f'Figures and prediction arrays saved to: {out_dir}')


if __name__ == '__main__':
    main()
