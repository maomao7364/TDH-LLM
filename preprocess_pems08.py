"""Preprocess PeMS08 raw data into the DeST-LLM / ST-LLM 24x24 format.

Input:
  data/PEMS08/pems08.npz   — 'data' array (17856, 170, 3): [flow, occupancy, speed]
  data/PEMS08/distance.csv — edge list: from, to, cost

Rules (2026-10-10, per user spec):
  - 288 steps/day; channels [flow, tod, dow, daily_flow(lag 288), weekly_flow(lag 2016)]
  - drop first 2016 steps (no weekly reference) -> 15840 steps
  - chronological split: train 9504 / val 3168 / test 3168;
    val/test borrow 24 steps of input history backward, targets stay in-split
  - sliding stride 1, 24 -> 24; y = [future flow, future daily ref, future weekly ref]
  - raw values saved (no normalization, no zero-imputation)

Adjacency (NOT preprocess_pems.py's approach — that one kernels the whole matrix,
giving weight exp(0)=1 to every unconnected pair):
  - only CSV-present edges are weighted: w = exp(-cost^2 / sigma^2),
    sigma = std of the (deduplicated, min-cost) edge costs
  - bidirectional; duplicates keep the shorter cost; diagonal = 1
  - symmetric normalization D^{-1/2} A D^{-1/2}

Output: data/pems08_24x24/{train,val,test}.npz + adj.npy
"""
import os

import numpy as np
import pandas as pd

INPUT_LEN = 24
OUTPUT_LEN = 24
STEPS_PER_DAY = 288
WEEKLY_OFFSET = STEPS_PER_DAY * 7
TRAIN_STEPS = 9504
VAL_STEPS = 3168
TEST_STEPS = 3168

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_NPZ = os.path.join(DATA_DIR, 'data', 'PEMS08', 'pems08.npz')
SRC_CSV = os.path.join(DATA_DIR, 'data', 'PEMS08', 'distance.csv')
OUT_DIR = os.path.join(DATA_DIR, 'data', 'pems08_24x24')


def build_adjacency_from_csv(csv_path, num_nodes):
    """仅 CSV 存在的连接赋高斯核权重；双向、重复取短、对角 1、对称归一化。"""
    edges = pd.read_csv(csv_path)

    # 无向最小成本矩阵（先初始化为 +inf，只填真实边）
    cost = np.full((num_nodes, num_nodes), np.inf)
    for _, row in edges.iterrows():
        src, dst, c = int(row['from']), int(row['to']), float(row['cost'])
        if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
            continue
        if src == dst:
            continue
        c = min(cost[src, dst], c)
        cost[src, dst] = c
        cost[dst, src] = c
    np.fill_diagonal(cost, 0.0)

    finite = cost[np.isfinite(cost) & (cost > 0)]
    sigma = finite.std() if finite.size > 0 else 1.0
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    mask = np.isfinite(cost) & (cost > 0)
    adj[mask] = np.exp(-cost[mask] ** 2 / (sigma ** 2 + 1e-8))
    np.fill_diagonal(adj, 1.0)                      # 自环权重 1（规格指定）

    # 对称归一化 D^{-1/2} A D^{-1/2}
    d = np.sqrt(np.sum(adj, axis=1))
    d = np.where(d > 0, 1.0 / d, 0.0)
    adj = d[:, None] * adj * d[None, :]
    return adj.astype(np.float32), mask.sum() // 2, sigma


def sliding_window(data, input_len, output_len):
    T = data.shape[0]
    xs, ys = [], []
    for t in range(0, T - input_len - output_len + 1):
        xs.append(data[t:t + input_len])
        y_flow = data[t + input_len:t + input_len + output_len, :, 0:1]
        y_daily = data[t + input_len:t + input_len + output_len, :, 3:4]
        y_weekly = data[t + input_len:t + input_len + output_len, :, 4:5]
        ys.append(np.concatenate([y_flow, y_daily, y_weekly], axis=-1))
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def main():
    print(f'Loading {SRC_NPZ}')
    raw = np.load(SRC_NPZ)['data']                       # (17856, 170, 3)
    T, N, _ = raw.shape
    print(f'Raw: T={T} ({T / STEPS_PER_DAY:.2f} days), N={N}')

    flow = raw[:, :, 0:1].astype(np.float32)

    t_idx = np.arange(T)
    tod = np.broadcast_to((t_idx % STEPS_PER_DAY)[:, None, None] / STEPS_PER_DAY,
                          (T, N, 1)).astype(np.float32)
    dow = np.broadcast_to(((t_idx // STEPS_PER_DAY) % 7)[:, None, None] / 7.0,
                          (T, N, 1)).astype(np.float32)

    daily = np.zeros_like(flow)
    daily[STEPS_PER_DAY:] = flow[:-STEPS_PER_DAY]
    weekly = np.zeros_like(flow)
    weekly[WEEKLY_OFFSET:] = flow[:-WEEKLY_OFFSET]

    data = np.concatenate([flow, tod, dow, daily, weekly], axis=-1).astype(np.float32)
    data = data[WEEKLY_OFFSET:]                          # 15840 步
    print(f'After dropping first week: T={data.shape[0]}')

    # 固定分割点：train 9504 / val 3168 / test 3168（borrow 24 步输入历史）
    train_end = TRAIN_STEPS
    val_end = TRAIN_STEPS + VAL_STEPS
    test_end = TRAIN_STEPS + VAL_STEPS + TEST_STEPS
    assert test_end == data.shape[0], (test_end, data.shape)
    data_train = data[:train_end]
    data_val = data[train_end - INPUT_LEN:val_end]
    data_test = data[val_end - INPUT_LEN:test_end]
    print(f'Split timelines: train {data_train.shape[0]}, '
          f'val {data_val.shape[0]}, test {data_test.shape[0]}')

    out = {}
    for name, seg in [('train', data_train), ('val', data_val), ('test', data_test)]:
        x, y = sliding_window(seg, INPUT_LEN, OUTPUT_LEN)
        out[name] = (x, y)
        print(f'{name}: x={x.shape}, y={y.shape}')

    # ---- 校验 ----
    expected = {'train': 9457, 'val': 3145, 'test': 3145}
    for name, (x, y) in out.items():
        assert x.shape == (expected[name], 24, 170, 5), x.shape
        assert y.shape == (expected[name], 24, 170, 3), y.shape
        assert not np.isnan(x).any() and not np.isinf(x).any(), name
        assert not np.isnan(y).any() and not np.isinf(y).any(), name
    # 目标严格落在各自划分内：train 末窗 y 止于 9504；val 首窗 y 起于 9504
    assert np.isclose(out['train'][1][-1, -1, 0, 0], data[train_end - 1, 0, 0])
    assert np.isclose(out['val'][1][0, 0, 0, 0], data[train_end, 0, 0])
    assert np.isclose(out['val'][1][-1, -1, 0, 0], data[val_end - 1, 0, 0])
    assert np.isclose(out['test'][1][0, 0, 0, 0], data[val_end, 0, 0])
    print('Boundary checks passed (targets strictly in-split).')

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, (x, y) in out.items():
        np.savez_compressed(os.path.join(OUT_DIR, f'{name}.npz'), x=x, y=y)

    adj, n_edges, sigma = build_adjacency_from_csv(SRC_CSV, N)
    assert adj.shape == (170, 170)
    assert not np.isnan(adj).any() and not np.isinf(adj).any()
    np.save(os.path.join(OUT_DIR, 'adj.npy'), adj)

    z = out['train'][0][..., 0]
    print(f'adj: shape {adj.shape}, undirected edges {n_edges}, sigma(cost std) '
          f'{sigma:.1f}; weights: max {adj.max():.4f}, '
          f'nonzero off-diag {(adj > 0).sum() - 170}, diag range '
          f'[{np.diag(adj).min():.4f}, {np.diag(adj).max():.4f}]')
    print(f'flow zeros in train x: {(z == 0).sum()} ({(z == 0).mean() * 100:.2f}%) '
          f'— kept as-is, no imputation')
    print(f'Saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
