"""
DeST-LLM publication-quality visualization.

Core claim: DeST-LLM captures spatially coherent traffic patterns across
50 freeway sensors with prediction error growing smoothly over 120 minutes.

Output (figures/):
  predictions.npz           — pred + real arrays
  map_flow.html             — folium heatmap layer of predicted 60min flow
  map_error.html            — folium heatmap layer of 60min MAE
  ts_comparison.svg/pdf     — pred vs real for 3 sensors, one day
  horizon_errors.svg/pdf    — MAE + RMSE vs horizon
  scatter_60min.svg/pdf     — predicted vs actual, 60min
  node_errors.svg/pdf       — per-sensor MAE, sorted
  node_horizon_heatmap.svg/pdf — sensor × horizon error matrix
"""

import torch, numpy as np, pandas as pd, os, sys, argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from util import load_dataset
from util import load_adj_from_npy
from model.dest_llm import DeSTLLM

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---- Nature-figure rcParams ----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.4,
    "figure.dpi": 300,
})

# Palette — restrained: one neutral blue, one warm signal, one accent
C_BLUE  = '#3274a1'
C_ORANGE = '#e37222'
C_RED   = '#be3c28'
C_GREY  = '#7f7f7f'


def save_pub(fig, stem):
    """Save figure as SVG + PDF + high-DPI PNG."""
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeST-LLM visualization')
parser.add_argument('--checkpoint', type=str,
                    default=os.path.join(_script_dir, 'logs',
                                         '2026-05-29-16-43-44-pems07_custom',
                                         'best_model.pth'))
parser.add_argument('--data', type=str, default='pems07_custom')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--output_dir', type=str,
                    default=os.path.join(_script_dir, 'figures'))
parser.add_argument('--stations', type=str,
                    default=os.path.join(_script_dir, 'data', 'pems 07-1',
                                         'station_lon_lat_50(1).csv'))
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Run inference, save predictions
# ---------------------------------------------------------------------------
def run_inference():
    data_dir = os.path.join(_script_dir, 'data', args.data)
    device = torch.device(args.device)
    dl = load_dataset(data_dir, args.batch_size, args.batch_size, args.batch_size,
                      num_workers=0)
    scaler = dl['scaler']
    stations = pd.read_csv(os.path.join(_script_dir, args.stations))
    adj = load_adj_from_npy(os.path.join(data_dir, 'adj.npy'))

    gpt2_path = os.path.join(_script_dir, 'snapshots',
                             '607a30d783dfa663caf39e06633721c8d4cfcd7e')
    model = DeSTLLM(
        adj=adj, input_dim=3, hidden_dim=256, num_nodes=50,
        input_len=24, output_len=24, gpt_layers=4, U=2,
        time_bins=288, time_dim=128, gpt_dim=768,
        tcdg_layers=2, dropout=0.15, trend_kernel=5, period_kernel=7,
        gpt2_path=gpt2_path,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    real_y = torch.Tensor(dl['y_test']).to(device).transpose(1, 3)[:, 0, :, :]
    preds = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for x, _ in dl['test_loader']:
            x = x.float().to(device).transpose(1, 3)
            preds.append(model(x).transpose(1, 3).squeeze(1))
    y_hat = torch.cat(preds, dim=0)[:real_y.size(0)]

    y_hat_np = np.zeros_like(y_hat.cpu().numpy(), dtype=np.float32)
    real_np = real_y.cpu().numpy().astype(np.float32)
    for j in range(24):
        y_hat_np[:, :, j] = scaler.inverse_transform(y_hat[:, :, j]).cpu().numpy()

    np.savez_compressed(os.path.join(args.output_dir, 'predictions.npz'),
                        pred=y_hat_np, real=real_np)
    return real_np, y_hat_np, stations


# ---------------------------------------------------------------------------
# Map 1 — Predicted flow heatmap layer (60min), folium
# ---------------------------------------------------------------------------
def plot_map_flow(y_hat_np, stations):
    from folium.plugins import HeatMap
    import folium

    node_flow = y_hat_np[:, :, 11].mean(axis=0)
    flow_max = float(node_flow.max())

    # HeatMap expects [[lat, lon, weight], ...]
    heat_data = []
    for ni, row in stations.iterrows():
        heat_data.append([float(row['Latitude']), float(row['Longitude']),
                          float(node_flow[ni])])

    m = folium.Map(
        location=[float(stations['Latitude'].mean()),
                  float(stations['Longitude'].mean())],
        zoom_start=11, tiles='CartoDB positron', control_scale=True)

    HeatMap(
        heat_data, radius=18, blur=12, max_zoom=13,
        min_opacity=0.25, max_val=flow_max,
        gradient={0.2: '#440154', 0.4: '#31688e',
                   0.6: '#35b779', 0.8: '#fde725', 1.0: '#fde725'},
    ).add_to(m)

    m.save(os.path.join(args.output_dir, 'map_flow.html'))
    print('[1/7] Flow heatmap → figures/map_flow.html')


# ---------------------------------------------------------------------------
# Map 2 — Prediction error heatmap layer (60min MAE), folium
# ---------------------------------------------------------------------------
def plot_map_error(real_np, y_hat_np, stations):
    from folium.plugins import HeatMap
    import folium

    node_err = np.abs(y_hat_np[:, :, 11] - real_np[:, :, 11]).mean(axis=0)
    err_max = float(node_err.max())

    heat_data = []
    for ni, row in stations.iterrows():
        heat_data.append([float(row['Latitude']), float(row['Longitude']),
                          float(node_err[ni])])

    m = folium.Map(
        location=[float(stations['Latitude'].mean()),
                  float(stations['Longitude'].mean())],
        zoom_start=11, tiles='CartoDB positron', control_scale=True)

    HeatMap(
        heat_data, radius=18, blur=12, max_zoom=13,
        min_opacity=0.25, max_val=err_max,
        gradient={0.2: '#1a9850', 0.4: '#ffffbf',
                   0.7: '#fc8d59', 1.0: '#d73027'},
    ).add_to(m)

    m.save(os.path.join(args.output_dir, 'map_error.html'))
    print('[2/7] Error heatmap → figures/map_error.html')


# ---------------------------------------------------------------------------
# Plot 3 — Timeseries, three sensors, one day
# ---------------------------------------------------------------------------
def plot_ts_comparison(real_np, y_hat_np):
    mf = real_np.mean(axis=(0, 2))
    nodes = [np.argmax(mf), np.argsort(mf)[len(mf)//2], np.argmin(mf)]
    labels = ['High-flow', 'Median-flow', 'Low-flow']
    day_len, day_start = 288, 0

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.5), sharex=True)
    for ax, ni, lb in zip(axes, nodes, labels):
        r = real_np[day_start:day_start+day_len, ni, 11]
        p = y_hat_np[day_start:day_start+day_len, ni, 11]
        xh = np.arange(day_len) / 12
        ax.plot(xh, r, color=C_BLUE, lw=0.9, label='Observed')
        ax.plot(xh, p, color=C_ORANGE, lw=0.9, label='Predicted (60 min)')
        ax.fill_between(xh, r, p, alpha=0.08, color=C_GREY)
        ax.set_ylabel('Flow (veh/5min)', fontsize=7)
        ax.text(0.01, 0.92, f'{lb}  |  MAE = {np.abs(r-p).mean():.0f}',
                transform=ax.transAxes, fontsize=6.5, fontweight='bold',
                color='#333')
        ax.legend(fontsize=6, loc='upper right')
    axes[-1].set_xlabel('Hour of day')
    fig.suptitle('DeST-LLM — 60-min ahead prediction', fontsize=8,
                 fontweight='bold', x=0.12, y=0.995)
    plt.tight_layout(pad=0.8)
    save_pub(fig, os.path.join(args.output_dir, 'ts_comparison'))
    plt.close(fig)
    print('[3/7] Timeseries → figures/ts_comparison.{svg,pdf,png}')


# ---------------------------------------------------------------------------
# Plot 4 — Horizon error curves
# ---------------------------------------------------------------------------
def plot_horizon_errors(real_np, y_hat_np):
    H = real_np.shape[2]
    maes, rmses = [], []
    for j in range(H):
        d = y_hat_np[:,:,j] - real_np[:,:,j]
        maes.append(np.mean(np.abs(d)))
        rmses.append(np.sqrt(np.mean(d**2)))

    xm = np.arange(1, H+1) * 5
    fig, ax = plt.subplots(figsize=(7.2, 2.6))

    ax.plot(xm, maes, color=C_BLUE, lw=1.5, marker='o', ms=3.5,
            markeredgewidth=0, label='MAE')
    ax.fill_between(xm, 0, maes, alpha=0.06, color=C_BLUE)

    # Annotate key horizons inline
    for h, lb in [(15,'15m'),(30,'30m'),(60,'60m'),(90,'90m'),(120,'120m')]:
        jj = h//5 - 1
        ax.annotate(f'{lb}\n{maes[jj]:.0f}', (h, maes[jj]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=6, ha='center', color=C_BLUE,
                    arrowprops=dict(arrowstyle='->', color=C_GREY, lw=0.5))

    ax.set_xlabel('Prediction horizon (minutes)')
    ax.set_ylabel('MAE (veh/5min)', color=C_BLUE)
    ax.tick_params(axis='y', labelcolor=C_BLUE)
    ax.set_xlim(0, 125)

    ax2 = ax.twinx()
    ax2.plot(xm, rmses, color=C_RED, lw=1.5, marker='s', ms=3.5,
             markeredgewidth=0, label='RMSE')
    ax2.set_ylabel('RMSE (veh/5min)', color=C_RED)
    ax2.tick_params(axis='y', labelcolor=C_RED)

    ax.set_title('Error growth over prediction horizon', fontsize=8,
                 fontweight='bold', loc='left')
    plt.tight_layout(pad=0.5)
    save_pub(fig, os.path.join(args.output_dir, 'horizon_errors'))
    plt.close(fig)
    print('[4/7] Horizon errors → figures/horizon_errors.{svg,pdf,png}')


# ---------------------------------------------------------------------------
# Plot 5 — Predicted vs actual scatter, 60min
# ---------------------------------------------------------------------------
def plot_scatter(real_np, y_hat_np):
    r = real_np[:,:,11].ravel()
    p = y_hat_np[:,:,11].ravel()
    idx = np.random.RandomState(42).choice(len(r), min(5000, len(r)), replace=False)
    rs, ps = r[idx], p[idx]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(rs, ps, alpha=0.18, s=4, c=C_BLUE, edgecolors='none',
               rasterized=True)
    mx = max(rs.max(), ps.max()) * 1.05
    ax.plot([0, mx], [0, mx], '--', color=C_RED, lw=0.9, alpha=0.5)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel('Observed flow'); ax.set_ylabel('Predicted flow')
    ax.set_title(f'60-min horizon (MAE = {np.abs(rs-ps).mean():.0f})',
                 fontsize=8, fontweight='bold', loc='left')
    ax.text(0.95, 0.05, f'n = {len(rs):,}', transform=ax.transAxes,
            fontsize=6, ha='right', color=C_GREY)
    plt.tight_layout(pad=0.3)
    save_pub(fig, os.path.join(args.output_dir, 'scatter_60min'))
    plt.close(fig)
    print('[5/7] Scatter → figures/scatter_60min.{svg,pdf,png}')


# ---------------------------------------------------------------------------
# Plot 6 — Per-sensor MAE bar chart, 60min
# ---------------------------------------------------------------------------
def plot_node_errors(real_np, y_hat_np, stations):
    mae = np.abs(y_hat_np[:,:,11] - real_np[:,:,11]).mean(axis=0)
    order = np.argsort(mae)
    ids = stations['ID'].values[order]

    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    norm = plt.Normalize(mae.min(), mae.max())
    colors = plt.cm.RdYlGn_r(norm(mae[order]))
    ax.bar(range(50), mae[order], color=colors, edgecolor='white', lw=0.3, width=0.8)
    ax.set_xticks(range(50))
    ax.set_xticklabels(ids, rotation=90, fontsize=5.5)
    ax.set_ylabel('MAE (60 min)')
    ax.set_xlabel('Sensor ID')
    ax.set_title(f'Per-sensor prediction error — mean MAE = {mae.mean():.0f}',
                 fontsize=8, fontweight='bold', loc='left')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.tight_layout(pad=0.5)
    save_pub(fig, os.path.join(args.output_dir, 'node_errors'))
    plt.close(fig)
    print('[6/7] Node errors → figures/node_errors.{svg,pdf,png}')


# ---------------------------------------------------------------------------
# Plot 7 — Sensor × Horizon error matrix
# ---------------------------------------------------------------------------
def plot_error_heatmap(real_np, y_hat_np):
    H, N = real_np.shape[2], real_np.shape[1]
    hm = np.zeros((N, H))
    for j in range(H):
        hm[:, j] = np.abs(y_hat_np[:,:,j] - real_np[:,:,j]).mean(axis=0)
    order = np.argsort(hm.mean(axis=1))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    im = ax.imshow(hm[order, :], aspect='auto', cmap='YlOrRd',
                   interpolation='bilinear', vmin=0)
    ax.set_xticks([0, 5, 11, 17, 23])
    ax.set_xticklabels(['5min', '30min', '60min', '90min', '120min'])
    ax.set_xlabel('Prediction horizon')
    ax.set_ylabel('Sensors (sorted by mean MAE)')
    ax.set_title('Sensor × horizon prediction error (MAE)', fontsize=8,
                 fontweight='bold', loc='left')
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015, aspect=30)
    cbar.set_label('MAE', fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    plt.tight_layout(pad=0.5)
    save_pub(fig, os.path.join(args.output_dir, 'node_horizon_heatmap'))
    plt.close(fig)
    print('[7/7] Heatmap → figures/node_horizon_heatmap.{svg,pdf,png}')


# ---------------------------------------------------------------------------
def main():
    print('Running inference ...')
    real_np, y_hat_np, stations = run_inference()
    print(f'Samples: {real_np.shape[0]}  |  Sensors: {real_np.shape[1]}  |  Horizons: {real_np.shape[2]}\n')

    plot_map_flow(y_hat_np, stations)
    plot_map_error(real_np, y_hat_np, stations)
    plot_ts_comparison(real_np, y_hat_np)
    plot_horizon_errors(real_np, y_hat_np)
    plot_scatter(real_np, y_hat_np)
    plot_node_errors(real_np, y_hat_np, stations)
    plot_error_heatmap(real_np, y_hat_np)
    print(f'\nDone → {args.output_dir}')


if __name__ == '__main__':
    main()
