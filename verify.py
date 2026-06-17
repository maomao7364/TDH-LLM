"""Quick shape verification for DeST-LLM. Run from DeST-LLM/ directory."""
import torch
import numpy as np
import sys
import os

# Ensure local DeST-LLM/ imports take priority
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from model.mtd import MultiScaleDecomposition
from model.tcdg import StaticGraphConv, GraphConvolution
from model.gmsf import GatedMultiSourceFusion, LightweightTimeEmbedding
from model.pfa import PartiallyFrozenAttention
from model.dest_llm import DeSTLLM
from util import load_dataset, load_adj_from_npy

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
B, T, N, C = 2, 12, 50, 3           # small for synthetic test
HIDDEN, TIME_DIM, GPT_DIM = 64, 64, 768
TIME_BINS = 48

torch.manual_seed(42)

# Build simple adjacency
np.random.seed(0)
adj_np = np.eye(N) + np.random.randn(N, N) * 0.05
adj_np = (adj_np + adj_np.T) / 2
adj_np = adj_np / adj_np.sum(1, keepdims=True).clip(min=1e-6)
ADJ = torch.from_numpy(adj_np).float()

print(f'Device: {DEVICE}')
print(f'Synthetic input: [{B}, {T}, {N}, {C}]')
print()

# ---- MTD ----
print('=== MTD ===')
mtd = MultiScaleDecomposition(HIDDEN, trend_kernel=3, period_kernel=3).to(DEVICE)
x = torch.randn(B, T, N, HIDDEN).to(DEVICE)
trend, periodic, residual = mtd(x)
print(f'  trend/periodic/residual: {trend.shape}')
assert trend.shape == x.shape and periodic.shape == x.shape and residual.shape == x.shape
print('  OK')

# ---- StaticGraphConv ----
print('=== StaticGraphConv ===')
gcn = StaticGraphConv(HIDDEN, num_layers=2, dropout=0.15).to(DEVICE)
out_s = gcn(trend, ADJ.to(DEVICE))
print(f'  output: {out_s.shape}')
assert out_s.shape == x.shape
print('  OK')

# ---- LightweightTimeEmbedding ----
print('=== LightweightTimeEmbedding ===')
tem = LightweightTimeEmbedding(TIME_BINS, TIME_DIM).to(DEVICE)
raw_x = torch.randn(B, T, N, C).to(DEVICE)
raw_x[..., 1] = torch.rand(B, T, N).clamp(0.001, 0.999)
raw_x[..., 2] = torch.rand(B, T, N).clamp(0.001, 0.999)
te = tem(raw_x)
assert te.shape == (B, 1, N, TIME_DIM), f'{te.shape} != ({B}, 1, {N}, {TIME_DIM})'
print(f'  output: {te.shape}  OK')

# ---- GMSF ----
print('=== GMSF ===')
gmsf = GatedMultiSourceFusion(HIDDEN, GPT_DIM, num_components=3, time_dim=TIME_DIM).to(DEVICE)
comp1 = torch.randn(B, N, HIDDEN).to(DEVICE)
comp2 = torch.randn(B, N, HIDDEN).to(DEVICE)
comp3 = torch.randn(B, N, HIDDEN).to(DEVICE)
fused = gmsf([comp1, comp2, comp3], te.squeeze(1))
print(f'  fused: {fused.shape}  OK')

# ---- PFA ----
print('=== PFA ===')
pfa = PartiallyFrozenAttention(gpt_layers=3, U=1).to(DEVICE)
gpt_out = pfa(fused)
print(f'  gpt_out: {gpt_out.shape}  OK')

# ---- Full DeSTLLM with synthetic data ----
print('\n=== Full DeSTLLM (synthetic) ===')
model = DeSTLLM(
    adj=ADJ.numpy(), input_dim=C, hidden_dim=HIDDEN, num_nodes=N,
    input_len=T, output_len=12, gpt_layers=3, U=1,
    time_bins=TIME_BINS, time_dim=TIME_DIM, gpt_dim=GPT_DIM,
).to(DEVICE)

print(f'  Total params: {model.param_num():,}')
inp = torch.randn(B, C, N, T).to(DEVICE)
inp_perm = inp.permute(0, 3, 2, 1)
inp_perm[..., 1] = torch.rand(B, T, N).clamp(0.001, 0.999)
inp_perm[..., 2] = torch.rand(B, T, N).clamp(0.001, 0.999)
inp = inp_perm.permute(0, 3, 2, 1)

with torch.no_grad():
    out = model(inp)
print(f'  output: {out.shape}')
assert out.shape == (B, 12, N, 1), f'{out.shape} != ({B}, 12, {N}, 1)'
print('  OK')

# ---- Full DeSTLLM with real PeMS04 data ----
print('\n=== Full DeSTLLM (PeMS04 real data) ===')
DATA_DIR = os.path.join(_script_dir, 'data', 'pems04')
if os.path.exists(DATA_DIR):
    dataloader = load_dataset(DATA_DIR, batch_size=4, valid_batch_size=4, test_batch_size=4)
    print(f'  Train x: {dataloader["x_train"].shape}, y: {dataloader["y_train"].shape}')

    adj = load_adj_from_npy(os.path.join(DATA_DIR, 'adj.npy'))
    num_nodes_real = dataloader['x_train'].shape[2]
    input_dim_real = dataloader['x_train'].shape[3]

    model_real = DeSTLLM(
        adj=adj, input_dim=input_dim_real, hidden_dim=HIDDEN,
        num_nodes=num_nodes_real, input_len=12, output_len=12,
        gpt_layers=3, U=1, time_bins=288, time_dim=TIME_DIM, gpt_dim=GPT_DIM,
    ).to(DEVICE)

    print(f'  Model params: {model_real.param_num():,}')

    # Forward one batch
    batch_x = torch.from_numpy(dataloader['x_train'][:4]).float().to(DEVICE)  # [B, T, N, C]
    batch_x = batch_x.transpose(1, 3)  # [B, C, N, T] — ST-LLM convention
    with torch.no_grad():
        out_real = model_real(batch_x)
    print(f'  One batch output: {out_real.shape}')
    assert out_real.shape == (4, 12, num_nodes_real, 1), f'{out_real.shape} != (4, 12, {num_nodes_real}, 1)'
    print('  OK')
else:
    print(f'  Skipped — {DATA_DIR} not found. Run preprocess_pems.py first.')

print('\n=== All checks passed ===')
