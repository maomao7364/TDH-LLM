import torch
import torch.nn as nn


class MultiScaleDecomposition(nn.Module):
    """
    Decompose temporal signal into trend, periodic, and residual components
    using learnable filters. Replaces ST-LLM's TemporalEmbedding with explicit
    signal decomposition.

    Input:  [B, T, N, D]
    Output: trend [B, T, N, D], periodic [B, T, N, D], residual [B, T, N, D]
    """

    def __init__(self, hidden_dim, trend_kernel=3, period_kernel=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Trend extractor: low-pass filter initialized as moving average
        self.trend_conv = nn.Conv1d(
            hidden_dim, hidden_dim, trend_kernel,
            padding=trend_kernel // 2, bias=False, groups=hidden_dim
        )
        ma_weight = torch.ones(hidden_dim, 1, trend_kernel) / trend_kernel
        self.trend_conv.weight.data.copy_(ma_weight)

        # Periodic extractor: band-pass initialized as edge detector
        self.period_conv = nn.Conv1d(
            hidden_dim, hidden_dim, period_kernel,
            padding=period_kernel // 2, bias=False, groups=hidden_dim
        )
        hp_weight = torch.zeros(hidden_dim, 1, period_kernel)
        hp_weight[:, :, period_kernel // 2] = 1.0
        hp_weight[:, :, :period_kernel // 2] = -0.5 / (period_kernel // 2)
        hp_weight[:, :, period_kernel // 2 + 1:] = -0.5 / (period_kernel // 2)
        self.period_conv.weight.data.copy_(hp_weight)

    def forward(self, x):
        # x: [B, T, N, D]
        B, T, N, D = x.shape

        # Reshape for Conv1d: [B*N, D, T]
        x_t = x.permute(0, 2, 3, 1).reshape(B * N, D, T)

        trend = self.trend_conv(x_t)
        detrended = x_t - trend
        periodic = self.period_conv(detrended)
        residual = detrended - periodic

        # Reshape back to [B, T, N, D]
        trend = trend.reshape(B, N, D, T).permute(0, 3, 1, 2)
        periodic = periodic.reshape(B, N, D, T).permute(0, 3, 1, 2)
        residual = residual.reshape(B, N, D, T).permute(0, 3, 1, 2)

        return trend, periodic, residual
