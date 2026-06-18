import torch
import torch.nn as nn

from .mtd import MultiScaleDecomposition
from .tcdg import StaticGraphConv
from .gmsf import GatedMultiSourceFusion, LightweightTimeEmbedding
from .pfa import PartiallyFrozenAttention
from .horizon_decoder import HorizonAwareDecoder


class DeSTLLM(nn.Module):
    """
    DeST-LLM: Decomposed Spatial-Temporal Large Language Model for Traffic Prediction.

    Pipeline:
      1. Flow projection with local temporal context + time-feature injection
      2. MTD:  decompose into trend / periodic / residual
      3. Static GCN:  spatial smoothing per component (no noisy gate)
      4. Temporal pooling via learned attention
      5. GMSF:  time-conditioned gate-based fusion of the three components
      6. Node embedding:  explicit per-node identity token
      7. PFA:  GPT-2 models cross-node dependencies
      8. HorizonAwareDecoder: per-step conditioned prediction
    """

    def __init__(
        self,
        adj,                          # [N, N] static adjacency matrix
        input_dim=3,                  # channels: flow + time-of-day + day-of-week
        hidden_dim=64,                # internal feature dimension
        num_nodes=170,
        input_len=12,
        output_len=12,
        gpt_layers=3,                 # number of GPT-2 layers to use
        U=1,                          # number of top layers with unfrozen attention
        time_bins=288,                # resolution of time-of-day embedding (288 for 5-min)
        time_dim=64,                  # dimension for time-conditioning features
        gpt_dim=768,                  # GPT-2 hidden size
        trend_kernel=3,               # kernel size for trend filter in MTD
        period_kernel=3,              # kernel size for periodic filter in MTD
        tcdg_layers=2,                # number of stacked GCN layers
        dropout=0.15,                 # dropout rate
        use_horizon_decoder=True,     # True=HorizonAwareDecoder, False=Conv2d regression
        gpt2_path=None,               # local path to GPT-2 model directory
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len

        # Static adjacency
        self.register_buffer('adj', torch.tensor(adj, dtype=torch.float32))

        # 1. Input projection: flow channel with local temporal mixing
        #    kernel (1,3) gives each position context from t-1, t, t+1
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=(1, 3),
                                    padding=(0, 1), bias=False)

        # 1b. Time-feature injection:  tod/dow projected to same dim
        #     and added to flow features so time context flows through
        #     the entire decomposition + GCN pipeline, not just gating.
        self.has_time_features = input_dim >= 3
        aux_input_dim = max(input_dim - 1, 1)
        self.time_input_proj = nn.Conv2d(aux_input_dim, hidden_dim, kernel_size=1,
                                         bias=False)

        # 2. Multi-scale temporal decomposition
        self.mtd = MultiScaleDecomposition(hidden_dim, trend_kernel, period_kernel)

        # 3. Static graph convolution (no gate — proven better in ablation)
        self.spatial_gcn = StaticGraphConv(hidden_dim, num_layers=tcdg_layers,
                                           dropout=dropout)

        # 4. Temporal attention pooling
        self.temporal_query = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.xavier_uniform_(self.temporal_query)

        # 5. Time embedding for GMSF gate
        self.time_embed = LightweightTimeEmbedding(time_bins, time_dim)

        # 6. Gated multi-source fusion
        self.gmsf = GatedMultiSourceFusion(hidden_dim, gpt_dim, num_components=3,
                                           time_dim=time_dim, dropout=dropout)

        # 7. Explicit node embedding (like ST-LLM's node_emb)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, gpt_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # 8. Partially frozen GPT-2
        self.gpt = PartiallyFrozenAttention(gpt_layers, U, dropout=dropout,
                                            gpt2_path=gpt2_path)

        # 9. Prediction head
        self.use_horizon_decoder = use_horizon_decoder
        if use_horizon_decoder:
            self.decoder = HorizonAwareDecoder(
                gpt_dim=gpt_dim,
                output_len=output_len,
                time_bins=time_bins,
                time_dim=time_dim,
                horizon_dim=time_dim,
                dropout=dropout,
            )
        else:
            self.regression_layer = nn.Conv2d(gpt_dim, output_len, kernel_size=1)

    def param_num(self):
        return sum(p.nelement() for p in self.parameters())

    def _temporal_pool(self, x):
        """Learnable attention pooling over time steps: [B, T, N, D] -> [B, N, D]"""
        B, T, N, D = x.shape
        query = self.temporal_query.view(1, 1, 1, D)
        scores = (x * query).sum(dim=-1)                     # [B, T, N]
        attn = torch.softmax(scores / (D ** 0.5), dim=1)    # scaled dot-product
        return (x * attn.unsqueeze(-1)).sum(dim=1)          # [B, N, D]

    def forward(self, history_data):
        # history_data: [B, C, N, T]  (ST-LLM convention from train.py transpose)
        x = history_data.permute(0, 3, 2, 1)   # -> [B, T, N, C]
        B, T, N, C = x.shape

        # --- extract flow signal and auxiliary time features ---
        flow = x[..., 0:1]                                    # [B, T, N, 1]
        if self.has_time_features:
            time_feat = x[..., 1:]                             # [B, T, N, C-1]
        else:
            time_feat = torch.zeros(B, T, N, 1, device=x.device)

        # --- 1. flow projection with local temporal context ---
        flow = flow.permute(0, 3, 2, 1)                       # [B, 1, N, T]
        flow_proj = self.input_proj(flow)                      # [B, D, N, T]
        flow_proj = flow_proj.permute(0, 3, 2, 1)              # [B, T, N, D]

        # --- 1b. time-feature injection ---
        time_feat = time_feat.permute(0, 3, 2, 1)              # [B, C-1, N, T]
        time_proj = self.time_input_proj(time_feat)             # [B, D, N, T]
        time_proj = time_proj.permute(0, 3, 2, 1)              # [B, T, N, D]

        combined = flow_proj + time_proj                       # [B, T, N, D]

        # --- 2. MTD: decompose ---
        trend, periodic, residual = self.mtd(combined)

        # --- 3. static GCN: spatial smoothing per component ---
        trend_s = self.spatial_gcn(trend, self.adj)
        periodic_s = self.spatial_gcn(periodic, self.adj)
        residual_s = self.spatial_gcn(residual, self.adj)

        # --- 4. temporal pooling ---
        trend_p = self._temporal_pool(trend_s)                 # [B, N, D]
        periodic_p = self._temporal_pool(periodic_s)
        residual_p = self._temporal_pool(residual_s)

        # --- 5. time embedding for GMSF gate ---
        t_embed = self.time_embed(x).squeeze(1)                # [B, N, D_t]

        # --- 6. GMSF: gated fusion ---
        fused = self.gmsf([trend_p, periodic_p, residual_p], t_embed)  # [B, N, gpt_dim]

        # --- 7. add node embedding ---
        fused = fused + self.node_emb.unsqueeze(0)             # [B, N, gpt_dim]

        # --- 8. GPT-2: cross-node dependency modelling ---
        gpt_out = self.gpt(fused)                              # [B, N, gpt_dim]

        # --- 9. prediction head ---
        if self.use_horizon_decoder:
            last_tod = x[:, -1, :, 1] if self.has_time_features else \
                       torch.zeros(B, N, device=x.device)
            last_dow = x[:, -1, :, 2] if self.has_time_features else \
                       torch.zeros(B, N, device=x.device)
            out = self.decoder(gpt_out, last_tod, last_dow)        # [B, output_len, N, 1]
        else:
            gpt_out = gpt_out.permute(0, 2, 1).unsqueeze(-1)      # [B, gpt_dim, N, 1]
            out = self.regression_layer(gpt_out)                   # [B, output_len, N, 1]

        return out
