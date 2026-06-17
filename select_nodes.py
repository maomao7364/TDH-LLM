"""
Select nodes with stable, periodic traffic patterns for better predictability.

Scoring strategy:
  - day-to-day CV:  how consistent the daily profile is across days (lower = better)
  - autocorr @24h:  strength of daily periodicity (higher = better)
  - Composite score ranks nodes by "predictability".

Usage:
    python select_nodes.py              # visualize + select 50 most stable nodes
    python select_nodes.py --n 30       # select 30 nodes
"""

import numpy as np
import os
import sys
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_script_dir, 'data', 'pems07', 'PEMS07.npz')
STEPS_PER_DAY = 288

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError:
    print('[ERROR] matplotlib not installed. Run: pip install matplotlib')
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_flow():
    flow = np.load(DATA_PATH)['data'][:, :, 0].astype(np.float32)  # (T, N)
    T, N = flow.shape
    ndays = T // STEPS_PER_DAY
    # Trim to exact days
    flow = flow[:ndays * STEPS_PER_DAY, :]
    print(f'Loaded: T={ndays * STEPS_PER_DAY} steps, N={N} nodes, {ndays} days')
    print(f'Flow range: [{flow.min():.0f}, {flow.max():.0f}]')
    return flow, ndays, N


# ---------------------------------------------------------------------------
# Stability scoring
# ---------------------------------------------------------------------------
def score_stability(flow, ndays, N):
    """
    Per-node stability score combining:
      1. Day-to-day CV: std across days / mean, averaged over time steps (lower better)
      2. Autocorrelation at lag=288 (24h periodicity, higher better)

    Returns sorted node indices (best first), plus the component scores.
    """
    # Reshape: (ndays, 288, N)
    daily = flow.reshape(ndays, STEPS_PER_DAY, N)

    # 1. Average daily profile per node: (288, N)
    mean_profile = daily.mean(axis=0)

    # Day-to-day std per node (averaged across time steps): (N,)
    daily_std = daily.std(axis=0).mean(axis=0)
    daily_mean = mean_profile.mean(axis=0)
    cv = daily_std / (daily_mean + 1e-8)  # coefficient of variation

    # 2. Autocorrelation at lag 288 for each node
    # Flatten back to (T, N) and compute lag-288 correlation
    flow_flat = daily.reshape(-1, N)  # (ndays*288, N)
    ac_24h = np.zeros(N, dtype=np.float32)
    for i in range(N):
        series = flow_flat[:, i]
        s_mean = series.mean()
        s_std = series.std()
        if s_std < 1e-8:
            ac_24h[i] = 0.0
            continue
        # Pearson corr between series[t] and series[t-288]
        x = series[STEPS_PER_DAY:] - s_mean
        y = series[:-STEPS_PER_DAY] - s_mean
        denom = s_std * s_std * len(x)
        if denom < 1e-8:
            ac_24h[i] = 0.0
        else:
            ac_24h[i] = (x * y).sum() / denom

    # 3. Composite score: normalize both to [0,1] and combine
    cv_norm = 1.0 - (cv - cv.min()) / (cv.max() - cv.min() + 1e-8)  # higher = more stable
    ac_norm = (ac_24h - ac_24h.min()) / (ac_24h.max() - ac_24h.min() + 1e-8)

    composite = 0.5 * cv_norm + 0.5 * ac_norm
    ranking = np.argsort(composite)[::-1]  # best first

    return ranking, cv, ac_24h, composite


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_score_histogram(composite, cv, ac_24h, selected, n_select):
    """Histogram of scores, with selected nodes highlighted."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(composite, bins=50, color='lightgray', edgecolor='white', alpha=0.8)
    axes[0].hist(composite[selected], bins=20, color='steelblue', edgecolor='white', alpha=0.9)
    axes[0].set_xlabel('Stability score')
    axes[0].set_ylabel('Count')
    axes[0].legend(['All nodes', f'Selected {n_select}'], fontsize=8)

    axes[1].hist(cv, bins=50, color='lightgray', edgecolor='white', alpha=0.8)
    axes[1].hist(cv[selected], bins=20, color='coral', edgecolor='white', alpha=0.9)
    axes[1].set_xlabel('Day-to-day CV')
    axes[1].legend(['All nodes', f'Selected {n_select}'], fontsize=8)

    axes[2].hist(ac_24h, bins=50, color='lightgray', edgecolor='white', alpha=0.8)
    axes[2].hist(ac_24h[selected], bins=20, color='darkgreen', edgecolor='white', alpha=0.9)
    axes[2].set_xlabel('Autocorrelation @24h')
    axes[2].legend(['All nodes', f'Selected {n_select}'], fontsize=8)

    fig.suptitle('Node stability score distribution')
    plt.tight_layout()
    return fig


def plot_heatmap(flow, node_ids, title, ndays=98):
    """Heatmap of selected nodes across multiple days."""
    n_days_show = min(7, ndays)
    day_start = STEPS_PER_DAY * 40
    seg = flow[day_start:day_start + n_days_show * STEPS_PER_DAY, :]

    # Extract only selected nodes, then sort them by mean flow
    sub_flow = seg[:, node_ids]  # (7*288, n_select)
    node_order = np.argsort(sub_flow.mean(axis=0))
    data = sub_flow[:, node_order].T

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd',
                   interpolation='bilinear', vmax=np.percentile(data, 95))
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Nodes ({len(node_ids)})')

    for d in range(1, n_days_show):
        ax.axvline(d * STEPS_PER_DAY, color='white', linewidth=0.8, alpha=0.6)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Flow')
    plt.tight_layout()
    return fig


def plot_multi_day_overlay(flow, node_ids, title, ndays=98, n_days=5):
    """Overlay multiple days of traffic for selected nodes to show stability."""
    day_start = STEPS_PER_DAY * 40
    n_show = min(8, len(node_ids))
    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows), squeeze=False)
    axes = axes.flatten()

    x = np.arange(STEPS_PER_DAY) / 12  # hours
    colors = plt.cm.viridis(np.linspace(0, 1, n_days))

    for i, node_id in enumerate(node_ids[:n_show]):
        for d in range(n_days):
            start = day_start + d * STEPS_PER_DAY
            seg = flow[start:start + STEPS_PER_DAY, node_id]
            axes[i].plot(x, seg, color=colors[d], linewidth=0.8, alpha=0.7,
                        label=f'Day {d + 1}' if i == 0 else '')
        # Mean profile
        all_days = flow[day_start:day_start + n_days * STEPS_PER_DAY, node_id]
        mean_5d = all_days.reshape(n_days, STEPS_PER_DAY).mean(axis=0)
        axes[i].plot(x, mean_5d, color='black', linewidth=1.5, linestyle='--')

        axes[i].set_title(f'Node {node_id}', fontsize=9)
        axes[i].set_xlabel('Hour')
        axes[i].set_ylabel('Flow')
        axes[i].set_xticks(range(0, 25, 6))
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    if n_show > 0:
        axes[0].legend(fontsize=6, loc='upper right')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_contrast(flow, best_nodes, worst_nodes, ndays=98):
    """Side-by-side: most stable vs least stable nodes (single day)."""
    return plot_multi_day_overlay(flow, best_nodes,
                                 'Most stable nodes — day-to-day consistency',
                                 ndays, n_days=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Select stable PeMS07 nodes')
    parser.add_argument('--n', type=int, default=50, help='number of nodes to select')
    args = parser.parse_args()

    flow, ndays, N = load_flow()
    n_select = args.n

    # ---- Score all nodes by stability ----
    print('Computing stability scores (CV + autocorrelation) ...')
    ranking, cv, ac_24h, composite = score_stability(flow, ndays, N)

    selected = sorted(ranking[:n_select].tolist())
    worst_8 = ranking[-8:][::-1].tolist()
    best_8 = ranking[:8].tolist()

    print(f'\nTop 10 nodes by stability:')
    for i, idx in enumerate(ranking[:10]):
        print(f'  #{i+1}: Node {idx:>4d}  |  CV={cv[idx]:.3f}  '
              f'AC24h={ac_24h[idx]:.3f}  Score={composite[idx]:.3f}')

    # ---- Figure 1: Score distribution ----
    fig1 = plot_score_histogram(composite, cv, ac_24h, selected, n_select)

    # ---- Figure 2: Most stable 8 nodes (multi-day overlay) ----
    fig2 = plot_multi_day_overlay(flow, best_8,
                                  f'Most predictable nodes ({n_select} selected)',
                                  ndays, n_days=5)

    # ---- Figure 3: Least stable 8 nodes (multi-day overlay) ----
    fig3 = plot_multi_day_overlay(flow, worst_8,
                                  'Least predictable nodes (for comparison)',
                                  ndays, n_days=5)

    # ---- Figure 4: Heatmap of selected nodes ----
    fig4 = plot_heatmap(flow, selected, f'{n_select} selected nodes — 7-day view', ndays)

    # ---- Summary ----
    means = flow.mean(axis=0)
    sel_means = means[selected]
    print(f'\n=== Summary ===')
    print(f'Selected {n_select} nodes:')
    print(f'  Flow range: [{sel_means.min():.0f}, {sel_means.max():.0f}]')
    print(f'  Flow mean:   {sel_means.mean():.0f}  (dataset mean: {means.mean():.0f})')
    print(f'  CV range:    [{cv[selected].min():.3f}, {cv[selected].max():.3f}]')
    print(f'  AC24h range: [{ac_24h[selected].min():.3f}, {ac_24h[selected].max():.3f}]')
    print(f'\nNode IDs: {selected}')

    # ---- Save ----
    out_path = os.path.join(_script_dir, 'data', 'pems07', f'nodes_{n_select}.npy')
    np.save(out_path, np.array(selected, dtype=np.int32))
    print(f'Saved to {out_path}')

    plt.show()


if __name__ == '__main__':
    main()
