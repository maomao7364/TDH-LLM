import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightTimeEmbedding(nn.Module):
    """
    Provides positional time context derived from time-of-day and day-of-week features.
    Lighter than ST-LLM's TemporalEmbedding — this only provides positional info,
    not signal patterns (which are handled by MTD).
    """

    def __init__(self, time_bins, hidden_dim):
        super().__init__()
        self.time_bins = time_bins
        self.time_day = nn.Parameter(torch.empty(time_bins, hidden_dim))
        self.time_week = nn.Parameter(torch.empty(7, hidden_dim))
        nn.init.xavier_uniform_(self.time_day)
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        # x: [B, T, N, C] — raw input with normalised time features at channels 1,2
        # time-of-day in [0,1), day-of-week in [0,1) — scale to index space
        day_val = x[..., 1]   # [B, T, N]
        week_val = x[..., 2]  # [B, T, N]

        day_idx = (day_val[:, -1, :] * self.time_bins).long().clamp(0, self.time_bins - 1)
        week_idx = (week_val[:, -1, :] * 7).long().clamp(0, 6)

        d_emb = self.time_day[day_idx]    # [B, N, D]
        w_emb = self.time_week[week_idx]  # [B, N, D]
        return (d_emb + w_emb).unsqueeze(1)  # [B, 1, N, D]


class GatedMultiSourceFusion(nn.Module):
    """
    Fuses multiple feature sources (trend, periodic, residual) via learned gating.
    The gate is conditioned on time context so the model can learn to weight
    different components differently depending on temporal position.

    Input:  components: list of 3 tensors [B, N, H]
            t_embed: [B, N, D_t]
    Output: [B, N, gpt_dim]
    """

    def __init__(self, hidden_dim, gpt_dim=768, num_components=3, time_dim=64, dropout=0.15):
        super().__init__()

        self.component_proj = nn.ModuleList([
            nn.Linear(hidden_dim, gpt_dim)
            for _ in range(num_components)
        ])

        self.time_proj = nn.Linear(time_dim, hidden_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * (num_components + 1), hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_components),
        )

        self.out_proj = nn.Linear(gpt_dim, gpt_dim)

    def forward(self, components, t_embed):
        # components: list of [B, N, H]
        # t_embed: [B, N, D_t]
        t_feat = self.time_proj(t_embed)  # [B, N, H]

        # Compute gates from all sources
        concat = torch.cat(components + [t_feat], dim=-1)  # [B, N, H*(K+1)]
        gates = F.softmax(self.gate_net(concat), dim=-1)  # [B, N, K]

        # Project components to GPT space and fuse
        projected = [proj(comp) for proj, comp in zip(self.component_proj, components)]

        fused = sum(
            gates[..., k:k + 1] * projected[k]
            for k in range(len(components))
        )  # [B, N, gpt_dim]

        return self.out_proj(fused)
