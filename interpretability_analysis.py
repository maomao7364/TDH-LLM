"""Interpretability figures for a trained DeST-LLM checkpoint.

The script is inference-only. It reloads a checkpoint, runs the test split,
captures selected intermediate tensors, and writes paper-oriented figures to
``figures/interpretability`` or a user-provided output directory.
"""
import argparse
import os
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PALETTE = {
    "black": "#111827",
    "blue": "#4C78A8",
    "sky": "#8FB8DE",
    "green": "#8DAA91",
    "orange": "#C9895B",
    "red": "#B56576",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "light_gray": "#E5E7EB",
    "residual_hist": "#9DBAD5",
    "residual_kde": "#3F6E9F",
    "residual_mean": "#4B5563",
    "gate_reference": "#7C97B8",
    "gate_trend": "#8DAA91",
    "gate_periodic": "#7EA6B8",
    "gate_residual": "#D2A66A",
}
TIME_PERIODS = ["Night", "Morning peak", "Off-peak", "Evening peak"]


def key_horizon_indices(output_len):
    """Return zero-based indices for 30/60/90/120 min when using 5-min data."""
    wanted_steps = [6, 12, 18, 24]
    return [step - 1 for step in wanted_steps if step <= output_len]


def residual_summary(y_true, y_pred, mask_value=0.0):
    """Summarise pred-true residuals after masking invalid ground truth values."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > mask_value)
    residual = (y_pred - y_true)[valid]
    if residual.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan}
    return {
        "count": int(residual.size),
        "mean": float(np.mean(residual)),
        "std": float(np.std(residual)),
    }


def assign_time_periods(tod):
    """Map normalised time-of-day values to four interpretable traffic periods."""
    tod = np.asarray(tod, dtype=np.float64) % 1.0
    hour = tod * 24.0
    labels = np.full(tod.shape, "Night", dtype=object)
    labels[(hour >= 6.0) & (hour < 10.0)] = "Morning peak"
    labels[(hour >= 10.0) & (hour < 16.0)] = "Off-peak"
    labels[(hour >= 16.0) & (hour < 20.0)] = "Evening peak"
    return labels


def cosine_similarity_matrix(tokens, eps=1e-8):
    """Compute an NxN cosine-similarity matrix from node tokens [N, D]."""
    tokens = np.asarray(tokens, dtype=np.float64)
    denom = np.linalg.norm(tokens, axis=-1, keepdims=True)
    normalised = tokens / np.maximum(denom, eps)
    return normalised @ normalised.T


def budgeted_keep_count(batch_size, remaining):
    """Return how many samples to keep; negative remaining means no limit."""
    if remaining < 0:
        return batch_size
    return min(batch_size, remaining)


def averaged_cosine_similarity(tokens):
    """Average per-sample node cosine similarities from tokens [S, N, D]."""
    tokens = np.asarray(tokens, dtype=np.float64)
    if tokens.ndim == 2:
        return cosine_similarity_matrix(tokens)
    sims = [cosine_similarity_matrix(sample) for sample in tokens]
    return np.mean(sims, axis=0)


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def kde_curve(values, grid_size=256, max_points=50000):
    """Small dependency-free Gaussian KDE for residual plots."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.array([]), np.array([])
    if values.size > max_points:
        rng = np.random.default_rng(6666)
        values = rng.choice(values, size=max_points, replace=False)
    std = np.std(values)
    if std < 1e-12:
        return np.array([]), np.array([])
    q01, q99 = np.percentile(values, [1, 99])
    pad = max((q99 - q01) * 0.15, std * 0.25)
    x_grid = np.linspace(q01 - pad, q99 + pad, grid_size)
    bandwidth = 1.06 * std * (values.size ** (-1 / 5))
    bandwidth = max(bandwidth, 1e-6)
    chunks = []
    for start in range(0, values.size, 5000):
        chunk = values[start:start + 5000]
        u = (x_grid[:, None] - chunk[None, :]) / bandwidth
        chunks.append(np.exp(-0.5 * u * u).sum(axis=1))
    density = np.sum(chunks, axis=0) / (values.size * bandwidth * np.sqrt(2 * np.pi))
    return x_grid, density


def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.dpi": 140,
        "savefig.dpi": 600,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "legend.frameon": False,
    })


def save_figure(fig, out_dir, stem, formats=None):
    if formats is None:
        formats = ["png", "svg", "pdf"]
    for fmt in formats:
        fig.savefig(os.path.join(out_dir, f"{stem}.{fmt}"), bbox_inches="tight")
    plt.close(fig)


class IntermediateCollector:
    """Capture interpretable tensors without changing training code."""

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.last = {}
        self._original_gmsf_forward = None
        self._original_gpt_forward = None

    def __enter__(self):
        if hasattr(self.model, "mtd"):
            self.handles.append(self.model.mtd.register_forward_hook(self._capture_mtd))
        if hasattr(self.model, "periodic_gate"):
            self.handles.append(
                self.model.periodic_gate.register_forward_hook(self._capture_periodic_gate)
            )
        if hasattr(self.model, "gmsf"):
            self._patch_gmsf()
        if hasattr(self.model, "gpt"):
            self._patch_gpt()
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        if self._original_gmsf_forward is not None:
            self.model.gmsf.forward = self._original_gmsf_forward
        if self._original_gpt_forward is not None:
            self.model.gpt.forward = self._original_gpt_forward
        return False

    def _capture_mtd(self, module, inputs, output):
        trend, periodic, residual = output
        self.last["mtd"] = tuple(t.detach().float().cpu() for t in (trend, periodic, residual))

    def _capture_periodic_gate(self, module, inputs, output):
        self.last["periodic_gate"] = output.detach().float().cpu()

    def _patch_gmsf(self):
        import torch
        import torch.nn.functional as F

        gmsf = self.model.gmsf
        self._original_gmsf_forward = gmsf.forward
        collector = self

        def forward_with_capture(this, components, t_embed):
            t_feat = this.time_proj(t_embed)
            concat = torch.cat(components + [t_feat], dim=-1)
            gates = F.softmax(this.gate_net(concat), dim=-1)
            projected = [
                proj(comp)
                for proj, comp in zip(this.component_proj, components)
            ]
            fused = sum(
                gates[..., k:k + 1] * projected[k]
                for k in range(len(components))
            )
            out = this.out_proj(fused)
            collector.last["gmsf_gates"] = gates.detach().float().cpu()
            collector.last["fused_tokens"] = out.detach().float().cpu()
            return out

        gmsf.forward = types.MethodType(forward_with_capture, gmsf)

    def _patch_gpt(self):
        gpt = self.model.gpt
        self._original_gpt_forward = gpt.forward
        collector = self

        def forward_with_capture(this, x):
            collector.last["pre_gpt_tokens"] = x.detach().float().cpu()
            out = collector._original_gpt_forward(x)
            collector.last["post_gpt_tokens"] = out.detach().float().cpu()
            return out

        gpt.forward = types.MethodType(forward_with_capture, gpt)


def build_runtime(args, cfg):
    import torch

    from dataset_checks import validate_dataset_shapes
    from util import load_dataset
    from visualize_forecast import build_model, load_adjacency, resolve_data_dir

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
    print(f"Data shape: {shape_info}")
    adj = load_adjacency(args, data_dir, dataloader)
    model = build_model(args, cfg, adj, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint loaded: {args.checkpoint}")
    return device, dataloader, dataloader["scaler"], adj, model


def run_inference_with_intermediates(args, cfg):
    import torch
    from visualize_forecast import apply_reference_mode

    device, dataloader, scaler, adj, model = build_runtime(args, cfg)
    real_y = torch.as_tensor(dataloader["y_test"], dtype=torch.float32, device=device)
    real_y = real_y.transpose(1, 3)[:, 0, :, :]

    preds = []
    tod_values = []
    ref_dependency = []
    gmsf_weights = []
    pre_tokens = []
    post_tokens = []
    mtd_example = None
    reference_mode = cfg.get("reference_mode", args.reference_mode)
    use_amp = (device.type == "cuda") and (not args.no_amp)
    sample_offset = 0
    token_budget = args.max_token_samples
    gate_budget = args.max_gate_samples

    with IntermediateCollector(model) as collector:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for x, y in dataloader["test_loader"]:
                batch_size = x.size(0)
                x_original = x.float().to(device, non_blocking=True)
                y_original = y.float().to(device, non_blocking=True)
                test_x = x_original.transpose(1, 3)
                test_y = y_original.transpose(1, 3)
                test_x, future_daily, future_weekly = apply_reference_mode(
                    test_x, test_y, reference_mode
                )
                collector.last = {}
                out = model(
                    test_x, future_daily=future_daily, future_weekly=future_weekly
                ).transpose(1, 3).squeeze(1)
                preds.append(out.detach())

                keep = budgeted_keep_count(batch_size, gate_budget)
                if keep > 0:
                    tod_values.append(x_original[:keep, -1, :, 1].detach().cpu().numpy())
                    if "periodic_gate" in collector.last:
                        gate = collector.last["periodic_gate"][:keep].numpy()
                        ref_dependency.append(1.0 - gate.mean(axis=-1))
                    if "gmsf_gates" in collector.last:
                        gmsf_weights.append(collector.last["gmsf_gates"][:keep].numpy())
                    if gate_budget >= 0:
                        gate_budget -= keep

                keep = budgeted_keep_count(batch_size, token_budget)
                if keep > 0 and "pre_gpt_tokens" in collector.last:
                    pre_tokens.append(collector.last["pre_gpt_tokens"][:keep].numpy())
                    post_tokens.append(collector.last["post_gpt_tokens"][:keep].numpy())
                    if token_budget >= 0:
                        token_budget -= keep

                if mtd_example is None:
                    start = sample_offset
                    end = sample_offset + batch_size
                    if start <= args.sample_index < end and "mtd" in collector.last:
                        local_idx = args.sample_index - start
                        node = min(max(args.node, 0), args.num_nodes - 1)
                        trend, periodic, residual = collector.last["mtd"]
                        flow_norm = x_original[local_idx, :, node, 0].detach().cpu()
                        flow = scaler.inverse_transform(flow_norm).numpy()
                        mtd_example = {
                            "node": node,
                            "sample_index": args.sample_index,
                            "flow": flow,
                            "trend": np.linalg.norm(
                                trend[local_idx, :, node, :].numpy(), axis=-1
                            ),
                            "periodic": np.linalg.norm(
                                periodic[local_idx, :, node, :].numpy(), axis=-1
                            ),
                            "residual": np.linalg.norm(
                                residual[local_idx, :, node, :].numpy(), axis=-1
                            ),
                        }

                sample_offset += batch_size

    y_hat = torch.cat(preds, dim=0)[:real_y.size(0), ...]
    y_pred = scaler.inverse_transform(y_hat).detach().cpu().numpy()
    y_true = real_y.detach().cpu().numpy()
    captured = {
        "tod": np.concatenate(tod_values, axis=0) if tod_values else None,
        "ref_dependency": (
            np.concatenate(ref_dependency, axis=0) if ref_dependency else None
        ),
        "gmsf_weights": np.concatenate(gmsf_weights, axis=0) if gmsf_weights else None,
        "pre_tokens": np.concatenate(pre_tokens, axis=0) if pre_tokens else None,
        "post_tokens": np.concatenate(post_tokens, axis=0) if post_tokens else None,
        "mtd_example": mtd_example,
        "adj": adj,
    }
    return y_true, y_pred, captured


def sample_residual_for_plot(residual, max_points=200000):
    if residual.size > max_points:
        rng = np.random.default_rng(6666)
        return rng.choice(residual, size=max_points, replace=False)
    return residual


def draw_residual_distribution(ax, residual, summary, minutes):
    residual = sample_residual_for_plot(residual)
    ax.hist(
        residual,
        bins=70,
        density=True,
        color=PALETTE["residual_hist"],
        alpha=0.66,
        edgecolor="white",
        linewidth=0.3,
    )
    x_grid, density = kde_curve(residual)
    if x_grid.size:
        ax.plot(x_grid, density, color=PALETTE["residual_kde"], lw=1.55)
    ax.axvline(0, color=PALETTE["red"], lw=1.0, ls="--")
    ax.axvline(summary["mean"], color=PALETTE["residual_mean"], lw=1.0)
    ax.text(
        0.03,
        0.94,
        f"\u03bc={summary['mean']:.2f}\n\u03c3={summary['std']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.8),
    )
    ax.set_title(f"{minutes} min residual")
    ax.set_xlabel("Prediction residual")
    ax.set_ylabel("Density")


def plot_residual_distributions(y_true, y_pred, out_dir):
    horizon_indices = key_horizon_indices(y_true.shape[-1])
    if not horizon_indices:
        horizon_indices = [y_true.shape[-1] - 1]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8), squeeze=False)
    summaries = []
    residual_items = []
    for ax, h_idx in zip(axes.ravel(), horizon_indices):
        valid = y_true[:, :, h_idx] > 0
        residual = (y_pred[:, :, h_idx] - y_true[:, :, h_idx])[valid]
        summary = residual_summary(y_true[:, :, h_idx], y_pred[:, :, h_idx])
        minutes = (h_idx + 1) * 5
        summaries.append({
            "horizon": h_idx + 1,
            "minutes": minutes,
            **summary,
        })
        residual_items.append((h_idx, minutes, residual, summary))
        draw_residual_distribution(ax, residual, summary, minutes)
    for ax in axes.ravel()[len(horizon_indices):]:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, out_dir, "residual_distribution_by_horizon")

    for h_idx, minutes, residual, summary in residual_items:
        single_fig, single_ax = plt.subplots(figsize=(4.6, 3.8))
        draw_residual_distribution(single_ax, residual, summary, minutes)
        single_fig.tight_layout()
        save_figure(
            single_fig,
            out_dir,
            f"residual_distribution_{minutes}min",
        )

    pd.DataFrame(summaries).to_csv(
        os.path.join(out_dir, "residual_distribution_summary.csv"), index=False
    )


def plot_mtd_example(example, out_dir):
    if example is None:
        print("MTD example unavailable; skipped MTD response figure.")
        return
    steps = np.arange(1, len(example["flow"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True)
    axes[0].plot(steps, example["flow"], color=PALETTE["black"], lw=1.4)
    axes[0].set_ylabel("Flow")
    axes[0].set_title(
        f"Input flow, node {example['node']}, sample {example['sample_index']}"
    )
    axes[1].plot(steps, minmax(example["trend"]), color=PALETTE["green"], lw=1.4, label="Trend")
    axes[1].plot(
        steps, minmax(example["periodic"]), color=PALETTE["blue"], lw=1.4, label="Periodic"
    )
    axes[1].plot(
        steps, minmax(example["residual"]), color=PALETTE["orange"], lw=1.4, label="Residual"
    )
    axes[1].set_xlabel("Historical time step")
    axes[1].set_ylabel("Normalised hidden response")
    axes[1].set_title("MTD component response strength")
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.23))
    fig.tight_layout()
    save_figure(fig, out_dir, "mtd_component_response")

    flow_fig, flow_ax = plt.subplots(figsize=(5.0, 3.2))
    flow_ax.plot(steps, example["flow"], color=PALETTE["black"], lw=1.5)
    flow_ax.set_xlabel("Historical time step")
    flow_ax.set_ylabel("Flow")
    flow_ax.set_title(f"Input flow, node {example['node']}")
    flow_fig.tight_layout()
    save_figure(flow_fig, out_dir, "mtd_input_flow")

    comp_fig, comp_ax = plt.subplots(figsize=(5.2, 3.5))
    comp_ax.plot(steps, minmax(example["trend"]), color=PALETTE["gate_trend"], lw=1.5, label="Trend")
    comp_ax.plot(
        steps, minmax(example["periodic"]), color=PALETTE["gate_periodic"], lw=1.5, label="Periodic"
    )
    comp_ax.plot(
        steps, minmax(example["residual"]), color=PALETTE["gate_residual"], lw=1.5, label="Residual"
    )
    comp_ax.set_xlabel("Historical time step")
    comp_ax.set_ylabel("Normalised hidden response")
    comp_ax.set_title("MTD component response strength")
    comp_ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    comp_fig.tight_layout()
    save_figure(comp_fig, out_dir, "mtd_component_response_strength")


def aggregate_gates_by_period(tod, ref_dependency, gmsf_weights):
    labels = assign_time_periods(tod)
    rows = []
    for period in TIME_PERIODS:
        mask = labels == period
        row = {"period": period, "count": int(mask.sum())}
        if ref_dependency is not None and mask.any():
            row["reference_dependency"] = float(np.mean(ref_dependency[mask]))
        else:
            row["reference_dependency"] = np.nan
        if gmsf_weights is not None and mask.any():
            values = gmsf_weights[mask]
            row["trend_weight"] = float(np.mean(values[:, 0]))
            row["periodic_weight"] = float(np.mean(values[:, 1]))
            row["residual_weight"] = float(np.mean(values[:, 2]))
        else:
            row["trend_weight"] = np.nan
            row["periodic_weight"] = np.nan
            row["residual_weight"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_gate_analysis(tod, ref_dependency, gmsf_weights, out_dir):
    if tod is None or (ref_dependency is None and gmsf_weights is None):
        print("Gate tensors unavailable; skipped gate analysis.")
        return
    df = aggregate_gates_by_period(tod, ref_dependency, gmsf_weights)
    df.to_csv(os.path.join(out_dir, "gate_weights_by_time_period.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    x = np.arange(len(df))
    if "reference_dependency" in df:
        axes[0].bar(x, df["reference_dependency"], color=PALETTE["gate_reference"], width=0.68)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["period"], rotation=25, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Mean 1 - gate")
    axes[0].set_title("Periodic reference dependency")

    bottoms = np.zeros(len(df))
    series = [
        ("Trend", "trend_weight", PALETTE["gate_trend"]),
        ("Periodic", "periodic_weight", PALETTE["gate_periodic"]),
        ("Residual", "residual_weight", PALETTE["gate_residual"]),
    ]
    for label, column, color in series:
        values = df[column].to_numpy(dtype=float)
        axes[1].bar(x, values, bottom=bottoms, color=color, width=0.68, label=label)
        bottoms += np.nan_to_num(values)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["period"], rotation=25, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean GMSF weight")
    axes[1].set_title("Component fusion weights")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.tight_layout()
    save_figure(fig, out_dir, "gate_weights_by_time_period")

    ref_fig, ref_ax = plt.subplots(figsize=(5.0, 3.8))
    if "reference_dependency" in df:
        ref_ax.bar(x, df["reference_dependency"], color=PALETTE["gate_reference"], width=0.62)
    ref_ax.set_xticks(x)
    ref_ax.set_xticklabels(df["period"], rotation=25, ha="right")
    ref_ax.set_ylim(0, 1)
    ref_ax.set_ylabel("Mean 1 - gate")
    ref_ax.set_title("Periodic reference dependency")
    ref_fig.tight_layout()
    save_figure(ref_fig, out_dir, "gate_reference_dependency")

    gmsf_fig, gmsf_ax = plt.subplots(figsize=(5.3, 3.8))
    bottoms = np.zeros(len(df))
    for label, column, color in series:
        values = df[column].to_numpy(dtype=float)
        gmsf_ax.bar(x, values, bottom=bottoms, color=color, width=0.62, label=label)
        bottoms += np.nan_to_num(values)
    gmsf_ax.set_xticks(x)
    gmsf_ax.set_xticklabels(df["period"], rotation=25, ha="right")
    gmsf_ax.set_ylim(0, 1)
    gmsf_ax.set_ylabel("Mean GMSF weight")
    gmsf_ax.set_title("Component fusion weights")
    gmsf_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    gmsf_fig.tight_layout()
    save_figure(gmsf_fig, out_dir, "gate_gmsf_component_weights")


def plot_similarity_heatmaps(adj, pre_tokens, post_tokens, out_dir):
    if pre_tokens is None or post_tokens is None:
        print("GPT token tensors unavailable; skipped similarity heatmaps.")
        return
    adj = np.asarray(adj, dtype=np.float64)
    pre_sim = averaged_cosine_similarity(pre_tokens)
    post_sim = averaged_cosine_similarity(post_tokens)
    delta = post_sim - pre_sim
    np.savez_compressed(
        os.path.join(out_dir, "node_similarity_matrices.npz"),
        adjacency=adj,
        pre_gpt_similarity=pre_sim,
        post_gpt_similarity=post_sim,
        delta_similarity=delta,
    )

    mats = [
        ("adjacency", "Static adjacency", adj, "viridis", None, None),
        ("pre_gpt_similarity", "Before GPT/PFA", pre_sim, "coolwarm", -1, 1),
        ("post_gpt_similarity", "After GPT/PFA", post_sim, "coolwarm", -1, 1),
        ("delta_similarity", "After - before", delta, "coolwarm", -0.4, 0.4),
    ]
    for stem, title, mat, cmap, vmin, vmax in mats:
        fig, ax = plt.subplots(figsize=(4.4, 4.0), dpi=600)
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")
        ax.tick_params(labelsize=8.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8.5)
        plt.tight_layout()
        save_figure(fig, out_dir, stem, formats=["png"])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DeST-LLM interpretability figures.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--config", default=None, help="Path to model_args.json; auto-detected")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data", default=None)
    parser.add_argument("--adj", default=None)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--reference_mode", default="full_refs",
                        choices=["full_refs", "no_daily", "no_weekly", "no_refs"])
    parser.add_argument("--module_variant", default=None)
    parser.add_argument("--use_horizon_decoder", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--max_gate_samples", type=int, default=-1,
                        help="Number of test samples for gate aggregation; -1 uses all.")
    parser.add_argument("--max_token_samples", type=int, default=128)

    parser.add_argument("--input_dim", type=int, default=5)
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--input_len", type=int, default=24)
    parser.add_argument("--output_len", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--gpt_dim", type=int, default=768)
    parser.add_argument("--gpt_layers", type=int, default=4)
    parser.add_argument("--U", type=int, default=1)
    parser.add_argument("--time_bins", type=int, default=288)
    parser.add_argument("--tcdg_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--trend_kernel", type=int, default=5)
    parser.add_argument("--period_kernel", type=int, default=7)
    parser.add_argument("--gpt2_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from visualize_forecast import load_config, resolve_gpt2_path

    cfg, ckpt_dir = load_config(args.checkpoint, args.config)
    for key, value in cfg.items():
        if hasattr(args, key) and key not in ("checkpoint", "config", "out_dir", "device"):
            setattr(args, key, value)
    if args.data is None:
        args.data = cfg.get("data", "pems07_custom")
    args.gpt2_path = resolve_gpt2_path(args.gpt2_path)
    out_dir = args.out_dir or os.path.join(ROOT_DIR, "figures", "interpretability")
    os.makedirs(out_dir, exist_ok=True)

    set_plot_style()
    y_true, y_pred, captured = run_inference_with_intermediates(args, cfg)
    np.savez_compressed(
        os.path.join(out_dir, "interpretability_predictions.npz"),
        y_true=y_true,
        y_pred=y_pred,
    )
    plot_residual_distributions(y_true, y_pred, out_dir)
    plot_mtd_example(captured["mtd_example"], out_dir)
    plot_gate_analysis(
        captured["tod"],
        captured["ref_dependency"],
        captured["gmsf_weights"],
        out_dir,
    )
    plot_similarity_heatmaps(
        captured["adj"],
        captured["pre_tokens"],
        captured["post_tokens"],
        out_dir,
    )
    print(f"Interpretability figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
