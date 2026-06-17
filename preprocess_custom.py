"""
Preprocess user-provided 50-node PeMS07 data.

Input:
  data/pems 07-1/flow_05_06.csv           — traffic flow (T, N)
  data/pems 07-1/station_lon_lat_50(1).csv — sensor ID, lat, lon

Output (data/pems07_custom/):
  train.npz, val.npz, test.npz  — (S, len, N, C)
  adj.npy                        — (N, N) distance-based adjacency
  nodes.npy                      — sensor IDs

Usage:
    python preprocess_custom.py
    python preprocess_custom.py --input-len 12 --output-len 24
"""

import numpy as np
import os
import sys
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(_script_dir, 'data', 'pems 07-1')
OUT_DIR = os.path.join(_script_dir, 'data', 'pems07_custom')

STEPS_PER_DAY = 288
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2


def build_adj_from_latlon(lat, lon, sigma=None, threshold=0.1):
    """Build adjacency matrix from GPS coordinates using Gaussian kernel."""
    N = len(lat)
    # Haversine-like: use Euclidean on lat/lon as approximation
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    # Approximate distance (valid for small regions)
    dist = np.sqrt(dlat ** 2 + dlon ** 2)

    if sigma is None:
        sigma = dist.std() if dist.std() > 0 else 1.0

    adj = np.exp(-dist ** 2 / (2 * sigma ** 2))
    adj[adj < threshold] = 0.0

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    d = np.sqrt(np.sum(adj, axis=1))
    d = np.where(d > 0, 1.0 / d, 0.0)
    adj = d[:, None] * adj * d[None, :]

    return adj.astype(np.float32)


def sliding_window(data, input_len, output_len):
    xs, ys = [], []
    window = input_len + output_len
    for t in range(0, data.shape[0] - window + 1):
        xs.append(data[t:t + input_len])
        ys.append(data[t + input_len:t + window, :, 0:1])
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Preprocess custom PeMS07 50-node data')
    parser.add_argument('--input-len', type=int, default=12)
    parser.add_argument('--output-len', type=int, default=24)
    args = parser.parse_args()

    input_len = args.input_len
    output_len = args.output_len

    # --- Load flow data ---
    import pandas as pd
    flow_df = pd.read_csv(os.path.join(SRC_DIR, 'flow_05_06.csv'))
    sensor_ids = flow_df.columns.astype(str).tolist()
    flow = flow_df.values.astype(np.float32)  # (T, N)
    T, N = flow.shape
    ndays = T // STEPS_PER_DAY
    print(f'Flow: T={T} ({ndays} days), N={N} nodes')
    print(f'  Range: [{flow.min():.0f}, {flow.max():.0f}]  Mean: {flow.mean():.1f}')
    print(f'  Sensor IDs: {sensor_ids[:5]}...{sensor_ids[-3:]}')

    # --- Load station metadata ---
    stations = pd.read_csv(os.path.join(SRC_DIR, 'station_lon_lat_50(1).csv'))
    # Drop trailing empty rows
    stations = stations.dropna(subset=['ID'])
    stations['ID'] = stations['ID'].astype(int).astype(str)
    print(f'Stations: {len(stations)} rows')

    # Build adjacency from lat/lon
    adj = build_adj_from_latlon(stations['Latitude'].values, stations['Longitude'].values)
    sparsity = (adj > 0).sum() / (N * N) * 100
    print(f'Adj: {adj.shape}  non-zero: {sparsity:.2f}%')

    # --- Time features ---
    flow_3d = flow.reshape(T, N, 1)  # (T, N, 1)

    tod = np.zeros((T, N, 1), dtype=np.float32)
    dow = np.zeros((T, N, 1), dtype=np.float32)
    for t in range(T):
        tod[t, :, 0] = (t % STEPS_PER_DAY) / STEPS_PER_DAY
        dow[t, :, 0] = ((t // STEPS_PER_DAY) % 7) / 7.0

    data = np.concatenate([flow_3d, tod, dow], axis=-1).astype(np.float32)
    print(f'Features: {data.shape}  (flow, tod, dow)')

    # --- Chronological splits ---
    train_end = int(T * TRAIN_RATIO)
    val_end = int(T * (TRAIN_RATIO + VAL_RATIO))
    data_train = data[:train_end]
    data_val = data[train_end - input_len:val_end]
    data_test = data[val_end - input_len:]

    print(f'Split: train={data_train.shape[0]}, val={data_val.shape[0]}, test={data_test.shape[0]}')

    # --- Sliding windows ---
    x_train, y_train = sliding_window(data_train, input_len, output_len)
    x_val, y_val = sliding_window(data_val, input_len, output_len)
    x_test, y_test = sliding_window(data_test, input_len, output_len)

    print(f'Train: x={x_train.shape}, y={y_train.shape}')
    print(f'Val:   x={x_val.shape},   y={y_val.shape}')
    print(f'Test:  x={x_test.shape},  y={y_test.shape}')

    # --- Save ---
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, (xx, yy) in [('train', (x_train, y_train)),
                            ('val', (x_val, y_val)),
                            ('test', (x_test, y_test))]:
        np.savez_compressed(os.path.join(OUT_DIR, f'{name}.npz'), x=xx, y=yy)

    # Save sensor IDs and adjacency
    np.save(os.path.join(OUT_DIR, 'nodes.npy'),
            np.array([int(s) for s in sensor_ids], dtype=np.int32))
    np.save(os.path.join(OUT_DIR, 'adj.npy'), adj)

    print(f'Saved to {OUT_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
