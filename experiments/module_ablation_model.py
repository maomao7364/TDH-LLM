"""Current-model module ablations for the daily + weekly DeST-LLM."""
import os
import sys

import torch
import torch.nn as nn

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from model.gmsf import GatedMultiSourceFusion, LightweightTimeEmbedding
from model.horizon_decoder import HorizonAwareDecoder
from model.mtd import MultiScaleDecomposition
from model.pfa import PartiallyFrozenAttention
from model.tcdg import StaticGraphConv


MODULE_VARIANTS = (
    'full_modules',
    'no_mtd',
    'no_spatial',
    'no_gmsf',
    'no_llm',
    'no_horizon_decoder',
)


class CrossNodeTransformer(nn.Module):
    """Randomly initialised cross-node Transformer used for no_llm."""

    def __init__(self, hidden_dim=768, num_layers=3, num_heads=12, dropout=0.15):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class ModuleAblationDeSTLLM(nn.Module):
    """DeST-LLM with switchable modules for ablation experiments."""

    def __init__(
        self,
        adj,
        input_dim=5,
        hidden_dim=256,
        num_nodes=50,
        input_len=24,
        output_len=24,
        gpt_layers=4,
        U=1,
        time_bins=288,
        time_dim=32,
        gpt_dim=768,
        trend_kernel=5,
        period_kernel=7,
        tcdg_layers=2,
        dropout=0.20,
        module_variant='full_modules',
        gpt2_path=None,
    ):
        super().__init__()
        if module_variant not in MODULE_VARIANTS:
            raise ValueError(f'Unknown module_variant: {module_variant}')

        self.module_variant = module_variant
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.gpt_dim = gpt_dim

        self.use_mtd = module_variant != 'no_mtd'
        self.use_spatial = module_variant != 'no_spatial'
        self.use_gmsf = module_variant not in ('no_mtd', 'no_gmsf')
        self.use_llm = module_variant != 'no_llm'
        self.use_horizon_decoder = module_variant != 'no_horizon_decoder'

        self.has_time_features = input_dim >= 3
        self.has_daily = input_dim >= 4
        self.has_weekly = input_dim >= 5

        self.register_buffer('adj', torch.tensor(adj, dtype=torch.float32))

        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=(1, 3),
                                    padding=(0, 1), bias=False)
        self.time_input_proj = nn.Conv2d(2, hidden_dim, kernel_size=1, bias=False)

        if self.has_daily:
            self.daily_proj = nn.Conv2d(1, hidden_dim, kernel_size=(1, 3),
                                        padding=(0, 1), bias=False)
        if self.has_weekly:
            self.weekly_proj = nn.Conv2d(1, hidden_dim, kernel_size=(1, 3),
                                         padding=(0, 1), bias=False)
        if self.has_daily or self.has_weekly:
            ref_count = int(self.has_daily) + int(self.has_weekly)
            gate_in = hidden_dim * (1 + ref_count) + time_dim
            self.periodic_gate = nn.Sequential(
                nn.Linear(gate_in, hidden_dim),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.periodic_gate[0].bias, 1.0)

        if self.use_mtd:
            self.mtd = MultiScaleDecomposition(hidden_dim, trend_kernel, period_kernel)
            self.num_components = 3
        else:
            self.num_components = 1

        if self.use_spatial:
            self.spatial_gcn = StaticGraphConv(hidden_dim, num_layers=tcdg_layers,
                                               dropout=dropout)

        self.temporal_query = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.xavier_uniform_(self.temporal_query)

        self.time_embed = LightweightTimeEmbedding(time_bins, time_dim)

        if self.use_gmsf and self.num_components > 1:
            self.gmsf = GatedMultiSourceFusion(hidden_dim, gpt_dim, num_components=3,
                                               time_dim=time_dim, dropout=dropout)
        else:
            self.concat_proj = nn.Linear(self.num_components * hidden_dim, gpt_dim)

        self.node_emb = nn.Parameter(torch.empty(num_nodes, gpt_dim))
        nn.init.xavier_uniform_(self.node_emb)

        if self.use_llm:
            self.gpt = PartiallyFrozenAttention(gpt_layers, U, dropout=dropout,
                                                gpt2_path=gpt2_path)
        else:
            self.cross_node = CrossNodeTransformer(
                hidden_dim=gpt_dim, num_layers=max(1, gpt_layers), num_heads=12,
                dropout=dropout,
            )

        if self.use_horizon_decoder:
            self.decoder = HorizonAwareDecoder(
                gpt_dim=gpt_dim,
                output_len=output_len,
                time_bins=time_bins,
                time_dim=time_dim,
                horizon_dim=time_dim,
                dropout=dropout,
                reference_dim=2 if self.has_weekly else 1,
            )
        else:
            self.regression_layer = nn.Conv2d(gpt_dim, output_len, kernel_size=1)

    def param_num(self):
        return sum(p.nelement() for p in self.parameters())

    def _spatial_forward(self, x):
        if self.use_spatial:
            return self.spatial_gcn(x, self.adj)
        return x

    def _temporal_pool(self, x):
        B, T, N, D = x.shape
        query = self.temporal_query.view(1, 1, 1, D)
        scores = (x * query).sum(dim=-1)
        attn = torch.softmax(scores / (D ** 0.5), dim=1)
        return (x * attn.unsqueeze(-1)).sum(dim=1)

    def _reference_pool(self, x):
        ref_pooled = []
        if self.has_daily:
            daily_flow = x[..., 3:4].permute(0, 3, 2, 1)
            daily_proj = self.daily_proj(daily_flow).permute(0, 3, 2, 1)
            ref_pooled.append(self._temporal_pool(daily_proj))
        if self.has_weekly:
            weekly_flow = x[..., 4:5].permute(0, 3, 2, 1)
            weekly_proj = self.weekly_proj(weekly_flow).permute(0, 3, 2, 1)
            ref_pooled.append(self._temporal_pool(weekly_proj))
        return ref_pooled

    def _apply_reference_gate(self, base_component, ref_pooled, t_embed):
        if not ref_pooled:
            return base_component
        gate_input = torch.cat([base_component] + ref_pooled + [t_embed], dim=-1)
        gate = self.periodic_gate(gate_input)
        ref_context = torch.stack(ref_pooled, dim=0).mean(dim=0)
        return gate * base_component + (1 - gate) * ref_context

    def forward(self, history_data, future_daily=None, future_weekly=None):
        x = history_data.permute(0, 3, 2, 1)
        B, T, N, C = x.shape

        flow = x[..., 0:1].permute(0, 3, 2, 1)
        flow_proj = self.input_proj(flow).permute(0, 3, 2, 1)

        if self.has_time_features:
            time_feat = x[..., 1:3]
        else:
            time_feat = torch.zeros(B, T, N, 2, device=x.device)
        time_proj = self.time_input_proj(time_feat.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        combined = flow_proj + time_proj

        t_embed = self.time_embed(x).squeeze(1)
        ref_pooled = self._reference_pool(x)

        if self.use_mtd:
            trend, periodic, residual = self.mtd(combined)
            trend_s = self._spatial_forward(trend)
            periodic_s = self._spatial_forward(periodic)
            residual_s = self._spatial_forward(residual)
            trend_p = self._temporal_pool(trend_s)
            periodic_p = self._temporal_pool(periodic_s)
            residual_p = self._temporal_pool(residual_s)
            periodic_p = self._apply_reference_gate(periodic_p, ref_pooled, t_embed)
            components = [trend_p, periodic_p, residual_p]
        else:
            single = self._spatial_forward(combined)
            single_p = self._temporal_pool(single)
            single_p = self._apply_reference_gate(single_p, ref_pooled, t_embed)
            components = [single_p]

        if self.use_gmsf and self.num_components > 1:
            fused = self.gmsf(components, t_embed)
        else:
            fused = self.concat_proj(torch.cat(components, dim=-1))

        fused = fused + self.node_emb.unsqueeze(0)

        if self.use_llm:
            gpt_out = self.gpt(fused)
        else:
            gpt_out = self.cross_node(fused)

        if self.use_horizon_decoder:
            last_tod = x[:, -1, :, 1] if self.has_time_features else torch.zeros(B, N, device=x.device)
            last_dow = x[:, -1, :, 2] if self.has_time_features else torch.zeros(B, N, device=x.device)
            return self.decoder(
                gpt_out, last_tod, last_dow,
                future_daily=future_daily,
                future_weekly=future_weekly,
            )

        gpt_out = gpt_out.permute(0, 2, 1).unsqueeze(-1)
        return self.regression_layer(gpt_out)
