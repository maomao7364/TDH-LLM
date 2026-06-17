import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """Simple GCN layer adapted to [B, N, D] format."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: [B, N, D], adj: [N, N]
        support = torch.einsum('bnd,do->bno', x, self.weight)
        output = torch.einsum('nk,bkd->bnd', adj, support)
        return output + self.bias


class StaticGraphConv(nn.Module):
    """
    Multi-layer static GCN with residual connections and dropout.
    Replaces the time-conditioned gate (TCDG) with simple, effective
    spatial smoothing.  No gate = no noisy modulation.
    """

    def __init__(self, hidden_dim, num_layers=2, dropout=0.15):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: [B, T, N, D], adj: [N, N]
        B, T, N, D = x.shape
        for layer in self.layers:
            identity = x
            h = layer(x.reshape(B * T, N, D), adj)
            h = self.dropout(h)
            x = h.reshape(B, T, N, D) + identity
        return x


# ---------------------------------------------------------------------------
# Original TCDG classes kept for backward compatibility (ablation models).
# These are no longer used in the main DeST-LLM pipeline.
# ---------------------------------------------------------------------------

class TCDGLayer(nn.Module):
    """Single TCDG block: time-conditioned gate -> GCN -> residual."""

    def __init__(self, hidden_dim, time_dim, dropout=0.15):
        super().__init__()
        self.gcn = GraphConvolution(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, adj, t_context):
        B, T, N, D = x.shape
        t_feat = self.time_proj(t_context)
        t_feat = t_feat.unsqueeze(2).expand(-1, -1, N, -1)
        gate_input = torch.cat([x, t_feat], dim=-1)
        node_gate = self.gate(gate_input)
        x_modulated = x * node_gate
        x_modulated = x_modulated.reshape(B * T, N, D)
        h_spatial = self.gcn(x_modulated, adj)
        h_spatial = self.dropout(h_spatial.reshape(B, T, N, D))
        return h_spatial + x


class TimeConditionedDynamicGraph(nn.Module):
    """Stacked TCDG layers. Kept for backward compatibility only."""

    def __init__(self, hidden_dim, time_dim, num_layers=2, dropout=0.15):
        super().__init__()
        self.layers = nn.ModuleList([
            TCDGLayer(hidden_dim, time_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, adj, t_context):
        for layer in self.layers:
            x = layer(x, adj, t_context)
        return x
