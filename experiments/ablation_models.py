"""
Ablation model variants for DeST-LLM.

Variants:
  full     — Complete DeST-LLM (baseline)
  no_mtd   — Remove MTD: single-path GCN, no decomposition, no GMSF
  no_tcdg  — Remove spatial GCN entirely (no graph convolution)
  no_gmsf  — Replace gated fusion with concat + linear projection
  no_llm   — Replace GPT-2 PFA with randomly-initialised TransformerEncoder
"""
import sys
import os

# Ensure the DeST-LLM/ package directory is on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)  # DeST-LLM/
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mtd import MultiScaleDecomposition
from model.tcdg import StaticGraphConv
from model.gmsf import GatedMultiSourceFusion, LightweightTimeEmbedding
from experiments.llm_backbones import (
    LLMBackbone, create_backbone, get_backbone_hidden_dim, ARCH_META,
)


# ---------------------------------------------------------------------------
# Cross-node MLP (replaces LLM in no_llm variant)
# ---------------------------------------------------------------------------
class CrossNodeMLP(nn.Module):
    """Simple cross-node MLP: 2-layer FFN applied per-node with residual."""

    def __init__(self, hidden_dim=768, expansion=4, dropout=0.1):
        super().__init__()
        inner = hidden_dim * expansion
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, N, D]
        return x + self.net(self.ln(x))


class CrossNodeTransformer(nn.Module):
    """Randomly-initialised TransformerEncoder replacing the LLM."""

    def __init__(self, hidden_dim=768, num_layers=3, num_heads=12, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, N, D] — N is the "sequence" dimension, D is embedding
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Ablation config flags
# ---------------------------------------------------------------------------
VARIANT_NAMES = ['full', 'no_mtd', 'no_tcdg', 'no_gmsf', 'no_llm']


# ---------------------------------------------------------------------------
# Configurable DeST-LLM for ablation experiments
# ---------------------------------------------------------------------------
class AblationDeSTLLM(nn.Module):
    """
    DeST-LLM with configurable components for ablation study.

    Parameters
    ----------
    adj : ndarray [N, N]
    input_dim, hidden_dim, num_nodes, input_len, output_len : standard
    time_bins, time_dim : time embedding config
    llm_dim : int
        Hidden dimension of the LLM backbone (auto-detected from backbone_type).
    trend_kernel, period_kernel : int
        MTD convolution kernel sizes.
    backbone_type : str
        One of 'gpt2', 'gpt2-random', 'qwen2', 'llama'.
    gpt_layers, U : int
        LLM layer count and unfrozen top layers.
    variant : str
        One of 'full', 'no_mtd', 'no_tcdg', 'no_gmsf', 'no_llm'.
    """

    def __init__(
        self,
        adj,
        input_dim=3,
        hidden_dim=64,
        num_nodes=307,
        input_len=12,
        output_len=12,
        time_bins=288,
        time_dim=64,
        trend_kernel=3,
        period_kernel=3,
        backbone_type='gpt2',
        gpt_layers=3,
        U=1,
        variant='full',
        tcdg_layers=2,
        dropout=0.15,
        gpt2_path=None,
    ):
        super().__init__()
        assert variant in VARIANT_NAMES, f'Unknown variant: {variant}'

        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.backbone_type = backbone_type
        self.variant = variant
        self.has_time_features = input_dim >= 3
        aux_input_dim = max(input_dim - 1, 1)

        # Determine LLM hidden dimension
        self.llm_dim = get_backbone_hidden_dim(backbone_type)

        # ---- Static adjacency ----
        self.register_buffer('adj', torch.tensor(adj, dtype=torch.float32))

        # ---- Input projection with local temporal context ----
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=(1, 3),
                                    padding=(0, 1), bias=False)

        # ---- Time-feature injection ----
        self.time_input_proj = nn.Conv2d(aux_input_dim, hidden_dim, kernel_size=1,
                                         bias=False)

        # ---- MTD (multi-scale decomposition) ----
        self.use_mtd = variant not in ('no_mtd',)
        if self.use_mtd:
            self.mtd = MultiScaleDecomposition(hidden_dim, trend_kernel, period_kernel)
            self.num_components = 3
        else:
            self.num_components = 1

        # ---- Spatial GCN (static, no gate) ----
        self.use_spatial = variant not in ('no_tcdg',)
        if self.use_spatial:
            self.spatial_gcn = StaticGraphConv(hidden_dim, num_layers=tcdg_layers,
                                               dropout=dropout)

        # ---- Temporal attention pooling ----
        self.temporal_query = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.xavier_uniform_(self.temporal_query)

        # ---- Time embedding for fusion ----
        self.time_embed = LightweightTimeEmbedding(time_bins, time_dim)

        # ---- GMSF / fusion ----
        self.use_gmsf = variant not in ('no_mtd', 'no_gmsf')
        if self.use_gmsf and self.num_components > 1:
            self.gmsf = GatedMultiSourceFusion(
                hidden_dim, self.llm_dim,
                num_components=self.num_components,
                time_dim=time_dim, dropout=dropout,
            )
        else:
            # Simple concat + projection (also used when num_components==1)
            self.concat_proj = nn.Linear(
                self.num_components * hidden_dim, self.llm_dim,
            )

        # ---- Node embedding ----
        self.node_emb = nn.Parameter(torch.empty(num_nodes, self.llm_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # ---- LLM / cross-node module ----
        self.use_llm = variant not in ('no_llm',)
        if self.use_llm:
            self.llm_backbone = create_backbone(backbone_type, gpt_layers, U,
                                                gpt2_path=gpt2_path,
                                                dropout=dropout)
        else:
            self.cross_node = CrossNodeTransformer(
                hidden_dim=self.llm_dim, num_layers=3, num_heads=12,
                dropout=dropout,
            )

        # ---- Regression ----
        self.regression_layer = nn.Conv2d(self.llm_dim, output_len, kernel_size=1)

    def param_num(self):
        return sum(p.nelement() for p in self.parameters())

    def _temporal_pool(self, x):
        """Learnable attention pooling: [B, T, N, D] -> [B, N, D]"""
        B, T, N, D = x.shape
        query = self.temporal_query.view(1, 1, 1, D)
        scores = (x * query).sum(dim=-1)                     # [B, T, N]
        attn = torch.softmax(scores / (D ** 0.5), dim=1)
        return (x * attn.unsqueeze(-1)).sum(dim=1)           # [B, N, D]

    def _spatial_forward(self, x):
        """Apply static GCN (if enabled), otherwise identity."""
        if self.use_spatial:
            return self.spatial_gcn(x, self.adj)
        return x

    def forward(self, history_data):
        # history_data: [B, C, N, T]
        x = history_data.permute(0, 3, 2, 1)                # [B, T, N, C]
        B, T, N, C = x.shape

        # ---- extract flow & time features ----
        flow = x[..., 0:1]                                   # [B, T, N, 1]
        if self.has_time_features:
            time_feat = x[..., 1:]
        else:
            time_feat = torch.zeros(B, T, N, 1, device=x.device)

        # ---- 1. input projection with local temporal context ----
        flow = flow.permute(0, 3, 2, 1)                      # [B, 1, N, T]
        flow_proj = self.input_proj(flow)                     # [B, D, N, T]
        flow_proj = flow_proj.permute(0, 3, 2, 1)             # [B, T, N, D]

        # ---- 1b. time-feature injection ----
        time_feat = time_feat.permute(0, 3, 2, 1)             # [B, C-1, N, T]
        time_proj = self.time_input_proj(time_feat)            # [B, D, N, T]
        time_proj = time_proj.permute(0, 3, 2, 1)             # [B, T, N, D]

        combined = flow_proj + time_proj                      # [B, T, N, D]

        if self.use_mtd:
            # ---- 2. MTD: decompose ----
            trend, periodic, residual = self.mtd(combined)

            # ---- 3. static GCN: spatial smoothing per component ----
            trend_s = self._spatial_forward(trend)
            periodic_s = self._spatial_forward(periodic)
            residual_s = self._spatial_forward(residual)

            # ---- 4. temporal pooling ----
            components = [
                self._temporal_pool(trend_s),
                self._temporal_pool(periodic_s),
                self._temporal_pool(residual_s),
            ]
        else:
            # ---- w/o MTD: single path ----
            spatial = self._spatial_forward(combined)
            components = [self._temporal_pool(spatial)]

        # ---- 5. time embedding for fusion ----
        t_embed = self.time_embed(x).squeeze(1)              # [B, N, time_dim]

        # ---- 6. fusion: GMSF or concat+project ----
        if self.use_gmsf and self.num_components > 1:
            fused = self.gmsf(components, t_embed)           # [B, N, llm_dim]
        else:
            concat = torch.cat(components, dim=-1)            # [B, N, num_comp*D]
            fused = self.concat_proj(concat)                  # [B, N, llm_dim]

        # ---- 7. node embedding ----
        fused = fused + self.node_emb.unsqueeze(0)            # [B, N, llm_dim]

        # ---- 8. cross-node modelling ----
        if self.use_llm:
            out = self.llm_backbone(fused)                    # [B, N, llm_dim]
        else:
            out = self.cross_node(fused)                      # [B, N, llm_dim]

        # ---- 9. regression ----
        out = out.permute(0, 2, 1).unsqueeze(-1)             # [B, llm_dim, N, 1]
        out = self.regression_layer(out)                      # [B, output_len, N, 1]

        return out
