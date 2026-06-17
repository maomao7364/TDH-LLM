"""
Preprocess PeMS04 raw data into the format expected by DeST-LLM / ST-LLM.

Input:
  PeMS04.npz  — array of shape (T, N, C_raw) where C_raw = [flow, occupancy, speed]
  PeMS04.csv  — edge list: from, to, cost

Output:
  pems04/      — directory with train.npz, val.npz, test.npz
  pems04_adj.npz — adjacency matrix [N, N]
"""
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_LEN = 12
OUTPUT_LEN = 12
# Standard PeMS04 split ratios
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
# 5-min intervals -> 288 steps / day
STEPS_PER_DAY = 288

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_NPZ = os.path.join(DATA_DIR, 'PeMS04.npz')
RAW_CSV = os.path.join(DATA_DIR, 'PeMS04.csv')
OUT_DIR = os.path.join(DATA_DIR, 'data', 'pems04')


def build_adjacency_from_csv(csv_path, num_nodes):
    """Build adjacency matrix from edge-list CSV (from,to,cost)."""
    import pandas as pd
    edges = pd.read_csv(csv_path)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for _, row in edges.iterrows():
        src, dst = int(row['from']), int(row['to'])
        cost = float(row['cost'])
        if src < num_nodes and dst < num_nodes:
            # Gaussian kernel: exp(-dist^2 / sigma^2)
            adj[src, dst] = cost
            adj[dst, src] = cost  # undirected

    # Thresholded Gaussian kernel
    sigma = adj[adj > 0].std() if adj[adj > 0].sum() > 0 else 1.0
    adj = np.exp(-adj ** 2 / (sigma ** 2 + 1e-8))
    adj[adj < 0.1] = 0.0

    # Symmetric normalisation
    d = np.sqrt(np.sum(adj, axis=1))
    d = np.where(d > 0, 1.0 / d, 0.0)
    adj = d[:, None] * adj * d[None, :]

    return adj


def sliding_window(data, input_len, output_len):
    """
    data: (T, N, C)
    returns: x (S, input_len, N, C), y (S, output_len, N, 1) — flow channel only, kept 4D
    """
    T = data.shape[0]
    x_list, y_list = [], []
    for t in range(0, T - input_len - output_len + 1):
        x_list.append(data[t:t + input_len])               # [input_len, N, C]
        y_list.append(data[t + input_len:t + input_len + output_len, :, 0:1])  # [output_len, N, 1]
    return np.stack(x_list, axis=0), np.stack(y_list, axis=0)


def main():
    print(f'Loading raw data from {RAW_NPZ}')
    raw = np.load(RAW_NPZ, allow_pickle=True)['data']       # (T, N, 3)
    T, N, C_raw = raw.shape
    print(f'Raw shape: T={T}, N={N}, channels={C_raw}')

    flow = raw[:, :, 0:1]                                    # (T, N, 1)

    # ---- generate time features ----
    tod = np.zeros((T, N, 1), dtype=np.float32)
    dow = np.zeros((T, N, 1), dtype=np.float32)
    for t in range(T):
        minute_of_day = (t % STEPS_PER_DAY) / STEPS_PER_DAY       # [0, 1)
        day_of_week = ((t // STEPS_PER_DAY) % 7) / 7.0            # [0, 1)
        tod[t, :, 0] = minute_of_day
        dow[t, :, 0] = day_of_week

    # Stack: [flow, time_of_day, day_of_week]
    data = np.concatenate([flow, tod, dow], axis=-1).astype(np.float32)  # (T, N, 3)
    print(f'Feature shape: {data.shape}  (channels: flow, tod, dow)')

    # ---- chronological splits ----
    train_end = int(T * TRAIN_RATIO)
    val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

    data_train = data[:train_end]
    data_val = data[train_end - INPUT_LEN:val_end]   # slight overlap for continuity
    data_test = data[val_end - INPUT_LEN:]

    print(f'Split: train T={data_train.shape[0]}, val T={data_val.shape[0]}, test T={data_test.shape[0]}')

    # ---- sliding windows ----
    x_train, y_train = sliding_window(data_train, INPUT_LEN, OUTPUT_LEN)
    x_val, y_val = sliding_window(data_val, INPUT_LEN, OUTPUT_LEN)
    x_test, y_test = sliding_window(data_test, INPUT_LEN, OUTPUT_LEN)

    print(f'Train: x={x_train.shape}, y={y_train.shape}')
    print(f'Val:   x={x_val.shape}, y={y_val.shape}')
    print(f'Test:  x={x_test.shape}, y={y_test.shape}')

    # ---- save ----
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, (xx, yy) in [('train', (x_train, y_train)),
                            ('val', (x_val, y_val)),
                            ('test', (x_test, y_test))]:
        np.savez_compressed(os.path.join(OUT_DIR, f'{name}.npz'), x=xx, y=yy)
    print(f'Saved .npz files to {OUT_DIR}')

    # ---- adjacency ----
    adj = build_adjacency_from_csv(RAW_CSV, N)
    adj_path = os.path.join(OUT_DIR, 'adj.npy')
    np.save(adj_path, adj)
    print(f'Saved adjacency ({adj.shape}) to {adj_path}')


if __name__ == '__main__':
    main()
