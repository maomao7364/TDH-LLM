import torch
import torch.nn as nn


class HorizonAwareDecoder(nn.Module):
    """
    Horizon-aware decoder: each future step predicted with explicit conditioning on
      - z_i:      node i's GPT-2 representation
      - e_h:      horizon step embedding (which step ahead: 5min, 10min, ..., 120min)
      - e_{t+h}^{tod}: future time-of-day embedding
      - e_{t+h}^{dow}: future day-of-week embedding

    y_{t+h,i} = f(z_i, e_h, e_{t+h}^{tod}, e_{t+h}^{dow})

    Replaces the original Conv2d(gpt_dim, output_len, 1) single-shot regression.
    """

    def __init__(self, gpt_dim, output_len, time_bins=288, time_dim=64,
                 horizon_dim=64, dropout=0.15):
        super().__init__()
        self.output_len = output_len
        self.time_bins = time_bins

        # Horizon step embedding: e_h
        self.horizon_emb = nn.Parameter(torch.empty(output_len, horizon_dim))
        nn.init.xavier_uniform_(self.horizon_emb)

        # Future time-of-day embedding: e_{t+h}^{tod}
        self.tod_emb = nn.Parameter(torch.empty(time_bins, time_dim))
        nn.init.xavier_uniform_(self.tod_emb)

        # Future day-of-week embedding: e_{t+h}^{dow}
        self.dow_emb = nn.Parameter(torch.empty(7, time_dim))
        nn.init.xavier_uniform_(self.dow_emb)

        # Fusion MLP: f(z_i, e_h, e_tod, e_dow) -> scalar flow
        fusion_in = gpt_dim + horizon_dim + time_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, z, last_tod, last_dow):
        """
        Args:
            z:        [B, N, gpt_dim]  GPT-2 output per node
            last_tod: [B, N]           tod at last input step, normalised [0, 1)
            last_dow: [B, N]           dow at last input step, normalised [0, 1)
        Returns:
            [B, output_len, N, 1]
        """
        B, N, D = z.shape
        H = self.output_len

        # Discretise current time position
        tod_step = (last_tod * self.time_bins).long().clamp(0, self.time_bins - 1)  # [B, N]
        dow_idx = (last_dow * 7).long().clamp(0, 6)                                   # [B, N]

        # Advance to each future horizon:  h = 1, 2, ..., H
        h_offset = torch.arange(1, H + 1, device=z.device).view(1, H, 1)              # [1, H, 1]
        absolute_step = tod_step.unsqueeze(1) + h_offset                              # [B, H, N]

        future_tod_step = absolute_step % self.time_bins                               # [B, H, N]
        days_advanced = absolute_step // self.time_bins                                # [B, H, N]
        future_dow_idx = (dow_idx.unsqueeze(1) + days_advanced) % 7                   # [B, H, N]

        # Lookup all embeddings (fully vectorised over horizons)
        e_tod = self.tod_emb[future_tod_step]     # [B, H, N, time_dim]
        e_dow = self.dow_emb[future_dow_idx]      # [B, H, N, time_dim]
        e_h = self.horizon_emb[:H].view(1, H, 1, -1).expand(B, H, N, -1)             # [B, H, N, horizon_dim]

        # Expand node representations over horizon dimension
        z_exp = z.unsqueeze(1).expand(B, H, N, D)                                     # [B, H, N, gpt_dim]

        # Concatenate all conditions and fuse
        feat = torch.cat([z_exp, e_h, e_tod, e_dow], dim=-1)                         # [B, H, N, fusion_in]
        out = self.fusion(feat)                                                        # [B, H, N, 1]

        return out
