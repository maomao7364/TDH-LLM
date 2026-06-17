import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import pickle


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapping numpy arrays (supports multi-worker prefetching)
# ---------------------------------------------------------------------------
class TrafficDataset(Dataset):
    """Wraps numpy arrays for efficient multi-worker data loading."""

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.xs[idx]), torch.from_numpy(self.ys[idx])


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------
class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# ---------------------------------------------------------------------------
# Data loading (returns PyTorch DataLoaders)
# ---------------------------------------------------------------------------
def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None,
                 num_workers=2):
    """Load train/val/test .npz files and return PyTorch DataLoaders."""
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    scaler = StandardScaler(
        mean=data['x_train'][..., 0].mean(),
        std=data['x_train'][..., 0].std(),
    )
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(
            data['x_' + category][..., 0])

    # Note: PyTorch DataLoader handles shuffling each epoch via shuffle=True,
    # so we don't manually permute the arrays here.

    data['train_loader'] = TorchDataLoader(
        TrafficDataset(data['x_train'], data['y_train']),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    data['val_loader'] = TorchDataLoader(
        TrafficDataset(data['x_val'], data['y_val']),
        batch_size=valid_batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    data['test_loader'] = TorchDataLoader(
        TrafficDataset(data['x_test'], data['y_test']),
        batch_size=test_batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    data['scaler'] = scaler

    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def MAE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))


def metric(pred, real):
    mae = MAE_torch(pred, real, 0).item()
    mape = MAPE_torch(pred, real, 0).item()
    wmape = WMAPE_torch(pred, real, 0).item()
    rmse = RMSE_torch(pred, real, 0).item()
    return mae, mape, rmse, wmape


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')


def load_adj_from_pkl(pkl_filename):
    _, _, adj_mx = load_pickle(pkl_filename)
    return _densify(adj_mx)


def load_adj_from_npy(npy_filename):
    return np.load(npy_filename).astype(np.float32)


def _densify(m):
    import scipy.sparse as sp
    if sp.issparse(m):
        return m.toarray().astype(np.float32)
    return np.array(m, dtype=np.float32)


def build_correlation_adj(data, threshold=0.5):
    """Build adjacency from correlation of traffic flow across nodes."""
    flow = data[..., 0]                        # [samples, T, N]
    flow = flow.reshape(-1, flow.shape[-1])    # [samples*T, N]
    corr = np.corrcoef(flow.T)                 # [N, N]
    adj = np.where(np.abs(corr) > threshold, corr, 0)
    d = np.sum(adj, axis=1)
    d = np.where(d > 0, 1.0 / np.sqrt(d), 0)
    adj = d[:, None] * adj * d[None, :]
    return adj
