"""Plot training curves and run comparisons from saved CSV logs."""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def find_run_dirs(root):
    run_dirs = []
    if os.path.isfile(os.path.join(root, 'train_log.csv')):
        return [root]
    for dirpath, _, filenames in os.walk(root):
        if 'train_log.csv' in filenames:
            run_dirs.append(dirpath)
    return sorted(run_dirs)


def run_label(run_dir, root):
    rel = os.path.relpath(run_dir, root)
    if rel == '.':
        return os.path.basename(os.path.abspath(run_dir))
    parts = rel.split(os.sep)
    return '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def save_loss_plot(train_df, out_dir, label):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(train_df['epoch'], train_df['train_loss'], color='#2563EB', lw=1.5, label='Train MAE')
    ax.plot(train_df['epoch'], train_df['valid_loss'], color='#DC2626', lw=1.5, label='Valid MAE')
    best_idx = train_df['valid_loss'].idxmin()
    best_row = train_df.loc[best_idx]
    ax.scatter([best_row['epoch']], [best_row['valid_loss']], color='#111827', zorder=5)
    ax.annotate(
        f"best {best_row['valid_loss']:.3f}",
        xy=(best_row['epoch'], best_row['valid_loss']),
        xytext=(8, 8),
        textcoords='offset points',
        fontsize=9,
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title(f'Training and validation loss: {label}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'training_loss.png'), bbox_inches='tight')
    plt.close(fig)


def save_metric_plot(train_df, out_dir, label):
    pairs = [
        ('rmse', 'RMSE', '#7C3AED'),
        ('mape', 'MAPE', '#059669'),
        ('wmape', 'WMAPE', '#F97316'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for ax, (metric, title, color) in zip(axes, pairs):
        train_col = f'train_{metric}'
        valid_col = f'valid_{metric}'
        if train_col not in train_df.columns or valid_col not in train_df.columns:
            ax.axis('off')
            continue
        ax.plot(train_df['epoch'], train_df[train_col], color=color, lw=1.4, label=f'Train {title}')
        ax.plot(train_df['epoch'], train_df[valid_col], color='#111827', lw=1.2, label=f'Valid {title}')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
    fig.suptitle(f'Training metrics: {label}', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'training_metrics.png'), bbox_inches='tight')
    plt.close(fig)


def save_test_plot(test_df, out_dir, label):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(test_df['epoch'], test_df['test_mae'], marker='o', color='#2563EB', lw=1.4)
    axes[0].set_title('Test MAE')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE')
    if 'test_rmse' in test_df.columns:
        axes[1].plot(test_df['epoch'], test_df['test_rmse'], marker='s', color='#7C3AED', lw=1.4, label='RMSE')
    if 'test_wmape' in test_df.columns:
        ax2 = axes[1].twinx()
        ax2.plot(test_df['epoch'], test_df['test_wmape'], marker='^', color='#059669', lw=1.2, label='WMAPE')
        ax2.set_ylabel('WMAPE')
    axes[1].set_title('Test RMSE / WMAPE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    fig.suptitle(f'Test evaluation during training: {label}', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'test_metrics.png'), bbox_inches='tight')
    plt.close(fig)


def plot_single_run(run_dir, out_root, root):
    label = run_label(run_dir, root)
    safe_label = label.replace('/', '__').replace('\\', '__')
    out_dir = os.path.join(out_root, safe_label)
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(run_dir, 'train_log.csv'))
    save_loss_plot(train_df, out_dir, label)
    save_metric_plot(train_df, out_dir, label)

    test_path = os.path.join(run_dir, 'test_log.csv')
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        if not test_df.empty:
            save_test_plot(test_df, out_dir, label)

    best_row = train_df.loc[train_df['valid_loss'].idxmin()].to_dict()
    summary = {
        'label': label,
        'run_dir': run_dir,
        'best_epoch': int(best_row['epoch']),
        'best_valid_mae': float(best_row['valid_loss']),
        'best_train_mae': float(best_row['train_loss']),
    }
    final_path = os.path.join(run_dir, 'final_test.csv')
    if os.path.exists(final_path):
        final_df = pd.read_csv(final_path)
        if not final_df.empty:
            for col in final_df.columns:
                summary[f'final_{col}'] = final_df.iloc[0][col]
    return summary


def save_comparison(run_summaries, root, out_root):
    if len(run_summaries) <= 1:
        return
    summary_df = pd.DataFrame(run_summaries)
    summary_df.to_csv(os.path.join(out_root, 'runs_summary.csv'), index=False)

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(summary_df)), 4.8))
    order = np.argsort(summary_df['best_valid_mae'].to_numpy())
    labels = summary_df['label'].to_numpy()[order]
    values = summary_df['best_valid_mae'].to_numpy()[order]
    ax.bar(np.arange(len(values)), values, color='#3B82F6')
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Best valid MAE')
    ax.set_title('Best validation MAE comparison')
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, 'compare_best_valid_mae.png'), bbox_inches='tight')
    plt.close(fig)

    test_col = None
    for candidate in ('final_avg_mae', 'final_test_mae', 'final_mae'):
        if candidate in summary_df.columns:
            test_col = candidate
            break
    if test_col is not None:
        valid = summary_df.dropna(subset=[test_col]).copy()
        if not valid.empty:
            order = np.argsort(valid[test_col].to_numpy())
            fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(valid)), 4.8))
            ax.bar(np.arange(len(valid)), valid[test_col].to_numpy()[order], color='#F97316')
            ax.set_xticks(np.arange(len(valid)))
            ax.set_xticklabels(valid['label'].to_numpy()[order], rotation=45, ha='right')
            ax.set_ylabel(test_col)
            ax.set_title('Final test MAE comparison')
            fig.tight_layout()
            fig.savefig(os.path.join(out_root, 'compare_final_test_mae.png'), bbox_inches='tight')
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for run_dir in find_run_dirs(root):
        label = run_label(run_dir, root)
        train_df = pd.read_csv(os.path.join(run_dir, 'train_log.csv'))
        ax.plot(train_df['epoch'], train_df['valid_loss'], lw=1.2, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Valid MAE')
    ax.set_title('Validation curves comparison')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, 'compare_valid_curves.png'), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot DeST-LLM training logs.')
    parser.add_argument('--log_dir', required=True,
                        help='A run directory with train_log.csv, or a root directory containing many runs.')
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    set_plot_style()
    root = os.path.abspath(args.log_dir)
    out_root = args.out_dir or os.path.join(root, 'figures')
    os.makedirs(out_root, exist_ok=True)

    run_dirs = find_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(f'No train_log.csv found under {root}')

    summaries = [plot_single_run(run_dir, out_root, root) for run_dir in run_dirs]
    save_comparison(summaries, root, out_root)
    print(f'Found {len(run_dirs)} run(s). Figures saved to: {out_root}')


if __name__ == '__main__':
    main()
