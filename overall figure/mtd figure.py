import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Style helper — matches input.py
# ============================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


def apply_style(fig, ax):
    """Apply input.py-style spines, grid, ticks, labels."""
    # transparent background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # all four spines visible, black border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    # grid
    ax.grid(True, which="major", axis="both", linestyle="-",
            linewidth=0.6, color="#D6D6D6", alpha=0.9)
    ax.grid(True, which="minor", axis="x", linestyle=":",
            linewidth=0.5, color="#E6E6E6", alpha=0.9)

    # ticks
    ax.tick_params(axis="both", labelsize=8.8, colors="#333333",
                   width=1.0, length=4.0)

    # labels
    ax.set_xlabel("Historical step", fontsize=11, color="#333333", fontweight="bold")
    ax.set_ylabel("Magnitude", fontsize=11, color="#333333", fontweight="bold")

    ax.set_xticks([1, 6, 12, 18, 24])


# ============================================================
# 2. Generate synthetic MTD components
# ============================================================
def generate_mtd_components(num_points=24, seed=2026):
    rng = np.random.default_rng(seed)
    x = np.arange(1, num_points + 1)
    t = np.linspace(0, 1, num_points)

    # Trend
    trend = (
        46
        + 7.5 * np.exp(-0.5 * ((t - 0.35) / 0.16) ** 2)
        + 10.0 * np.exp(-0.5 * ((t - 0.78) / 0.14) ** 2)
        - 3.0 * np.exp(-0.5 * ((t - 0.08) / 0.10) ** 2)
    )
    trend += 1.2 * np.sin(2 * np.pi * (t - 0.15))

    # Periodic
    amplitude = 0.85 + 0.25 * np.exp(-0.5 * ((t - 0.55) / 0.22) ** 2)
    periodic = amplitude * (
        4.8 * np.sin(2 * np.pi * 3.2 * t + 0.25)
        + 1.6 * np.sin(2 * np.pi * 6.4 * t + 1.00)
    )

    # Residual
    white = rng.normal(0, 0.9, num_points)
    residual = np.zeros(num_points)
    for i in range(1, num_points):
        residual[i] = 0.42 * residual[i - 1] + white[i]
    residual += 0.35 * np.sin(2 * np.pi * 10.5 * t + 0.8)
    spikes = {5: 1.1, 11: -1.5, 17: 1.3, 21: -0.9}
    for idx, amp in spikes.items():
        if 0 <= idx < num_points:
            residual[idx] += amp

    signal = trend + periodic + residual
    return x, trend, periodic, residual, signal


# ============================================================
# 3. Save helper
# ============================================================
def save_figure(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(out_dir, f"{stem}.{fmt}"),
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )
    plt.close(fig)


# ============================================================
# 4. Plot components
# ============================================================
def plot_trend(x, trend, out_dir="mtd_figures_transparent"):
    fig, ax = plt.subplots(figsize=(5.4, 2.0), dpi=600)
    color = "#E68613"
    ax.plot(x, trend, color=color, lw=2.1, solid_capstyle="round", zorder=3)
    apply_style(fig, ax)
    ymin, ymax = np.min(trend), np.max(trend)
    yr = ymax - ymin
    ax.set_ylim(ymin - 0.12 * yr, ymax + 0.12 * yr)
    save_figure(fig, out_dir, "mtd_trend")


def plot_periodic(x, periodic, out_dir="mtd_figures_transparent"):
    fig, ax = plt.subplots(figsize=(5.4, 2.0), dpi=600)
    color = "#4C9A2A"
    ax.plot(x, periodic, color=color, lw=1.8, solid_capstyle="round", zorder=3)
    ax.axhline(0, color="black", lw=0.75, alpha=0.70, zorder=1)
    apply_style(fig, ax)
    lim = 1.15 * np.max(np.abs(periodic))
    ax.set_ylim(-lim, lim)
    save_figure(fig, out_dir, "mtd_periodic")


def plot_residual(x, residual, out_dir="mtd_figures_transparent"):
    fig, ax = plt.subplots(figsize=(5.4, 2.0), dpi=600)
    color = "#2F6FD6"
    ax.plot(x, residual, color=color, lw=1.35, solid_capstyle="round", zorder=3)
    ax.axhline(0, color="black", lw=0.75, alpha=0.70, zorder=1)
    apply_style(fig, ax)
    lim = 1.18 * np.max(np.abs(residual))
    ax.set_ylim(-lim, lim)
    save_figure(fig, out_dir, "mtd_residual")


# ============================================================
# 5. Main
# ============================================================
def main():
    out_dir = "mtd_figures_transparent"
    x, trend, periodic, residual, _ = generate_mtd_components(
        num_points=24, seed=2026)

    plot_trend(x, trend, out_dir)
    plot_periodic(x, periodic, out_dir)
    plot_residual(x, residual, out_dir)

    print(f"Saved figures to: {os.path.abspath(out_dir)}")
    print("  - mtd_trend.png / pdf / svg")
    print("  - mtd_periodic.png / pdf / svg")
    print("  - mtd_residual.png / pdf / svg")


if __name__ == "__main__":
    main()
