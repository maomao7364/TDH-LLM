"""DeST-LLM 测试脚本：加载训练好的模型，逐 horizon 评估预测效果。

用法：
    python test.py --checkpoint ./logs/xxx-pems07_custom/best_model.pth

输出每个预测步长（5min/step）的 MAE/RMSE/MAPE/WMAPE，并标注关键区间：
  30min (step 6), 60min (step 12), 90min (step 18), 120min (step 24)
"""

import torch
import numpy as np
import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from util import load_dataset, metric
from util import load_adj_from_pkl, load_adj_from_npy, build_correlation_adj
from model.dest_llm import DeSTLLM


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeST-LLM test')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='训练好的模型权重路径 (best_model.pth)')
parser.add_argument('--config', type=str, default=None,
                    help='model_args.json 路径，自动覆盖模型超参（与 checkpoint 同目录下自动查找）')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='pems07_custom',
                    help='数据集名称 (data/ 下的目录)')
parser.add_argument('--adj', type=str, default=None,
                    help='邻接矩阵路径，默认自动查找')
parser.add_argument('--input_dim', type=int, default=3)
parser.add_argument('--num_nodes', type=int, default=50,
                    help='路网节点数')
parser.add_argument('--input_len', type=int, default=24, help='输入时间步长 (24=2h)')
parser.add_argument('--output_len', type=int, default=24,
                    help='预测步长 (24=120min)')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gpt2_path', type=str, default=None,
                    help='GPT-2 模型本地路径（目录），为空则从 HuggingFace 加载')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--time_dim', type=int, default=32)
parser.add_argument('--gpt_dim', type=int, default=768)
parser.add_argument('--gpt_layers', type=int, default=4)
parser.add_argument('--U', type=int, default=1)
parser.add_argument('--tcdg_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.20)
parser.add_argument('--trend_kernel', type=int, default=5)
parser.add_argument('--period_kernel', type=int, default=7)
parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()

# --- 从 model_args.json 读取模型超参（自动或手动指定） ---
import json as _json

_config_path = args.config
if _config_path is None:
    _ckpt_dir = (args.checkpoint if os.path.isdir(args.checkpoint)
                 else os.path.dirname(args.checkpoint))
    _auto_config = os.path.join(_ckpt_dir, 'model_args.json')
    if os.path.exists(_auto_config):
        _config_path = _auto_config
        print(f'Auto-detected config: {_config_path}')

if _config_path and os.path.exists(_config_path):
    with open(_config_path) as _f:
        _cfg = _json.load(_f)
    for _k, _v in _cfg.items():
        setattr(args, _k, _v)
    print(f'Loaded model config: {_config_path}')


def main():
    device = torch.device(args.device)

    # --- 解析数据集 ---
    if args.data == 'pems07_custom':
        args.num_nodes = 50
        data_dir = os.path.join(_script_dir, 'data', 'pems07_custom')
    else:
        data_dir = (os.path.join(_script_dir, 'data', args.data)
                    if not os.path.isabs(args.data) else args.data)

    print(f'Dataset : {args.data}')
    print(f'Nodes   : {args.num_nodes}')
    print(f'Horizons: {args.output_len} steps '
          f'({args.output_len * 5} min)')

    # --- GPT-2 模型路径（改成你的实际路径） ---
    if args.gpt2_path is None:
        args.gpt2_path = os.path.join(_script_dir, 'snapshots',
                                      '607a30d783dfa663caf39e06633721c8d4cfcd7e')

    # --- 加载数据 ---
    dataloader = load_dataset(data_dir, args.batch_size, args.batch_size, args.batch_size,
                              num_workers=args.num_workers)
    scaler = dataloader['scaler']

    # --- 邻接矩阵 ---
    if args.adj and os.path.exists(args.adj):
        adj = (load_adj_from_npy(args.adj) if args.adj.endswith('.npy')
               else load_adj_from_pkl(args.adj))
    else:
        auto_adj = os.path.join(data_dir, 'adj.npy')
        if os.path.exists(auto_adj):
            adj = load_adj_from_npy(auto_adj)
        else:
            auto_pkl = os.path.join(data_dir, 'adj.pkl')
            if os.path.exists(auto_pkl):
                adj = load_adj_from_pkl(auto_pkl)
            else:
                adj = build_correlation_adj(dataloader['x_train'])
    print(f'Adj shape: {adj.shape}')

    # --- 重建模型并加载权重 ---
    time_bins = getattr(args, 'time_bins', 288)  # from config or default PeMS 5-min
    model = DeSTLLM(
        adj=adj,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        gpt_layers=args.gpt_layers,
        U=args.U,
        time_bins=time_bins,
        time_dim=args.time_dim,
        gpt_dim=args.gpt_dim,
        trend_kernel=args.trend_kernel,
        period_kernel=args.period_kernel,
        tcdg_layers=args.tcdg_layers,
        dropout=args.dropout,
        gpt2_path=args.gpt2_path,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f'Checkpoint loaded: {args.checkpoint}')
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    # --- 逐 batch 推理 ---
    real_y = torch.Tensor(dataloader['y_test']).to(device)
    real_y = real_y.transpose(1, 3)[:, 0, :, :]  # [samples, N, out_len]

    preds = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for x, _ in dataloader['test_loader']:
            test_x = x.float().to(device, non_blocking=True)
            test_x = test_x.transpose(1, 3)                     # [B, C, N, T]
            out = model(test_x).transpose(1, 3)                 # [B, 1, N, out_len]
            preds.append(out.squeeze(1))

    y_hat = torch.cat(preds, dim=0)                            # [samples, N, out_len]
    y_hat = y_hat[:real_y.size(0), ...]

    # --- 逐 horizon 评估 ---
    print()
    print(f'{"Horizon":<12} {"MAE":>10} {"RMSE":>10} {"MAPE":>10} {"WMAPE":>10}')
    print('-' * 55)

    all_mae, all_rmse, all_mape, all_wmape = [], [], [], []
    key_horizons = {3: '15min', 6: '30min', 9: '45min', 12: '60min',
                    18: '90min', 24: '120min'}

    for j in range(args.output_len):
        pred_j = scaler.inverse_transform(y_hat[:, :, j])
        real_j = real_y[:, :, j]
        mae_j, mape_j, rmse_j, wmape_j = metric(pred_j, real_j)
        all_mae.append(mae_j)
        all_rmse.append(rmse_j)
        all_mape.append(mape_j)
        all_wmape.append(wmape_j)

        step = j + 1
        minutes = step * 5
        marker = '  <-- ' + key_horizons[step] if step in key_horizons else ''
        print(f'{step:>2} ({minutes:>3}min) {mae_j:>10.4f} {rmse_j:>10.4f} '
              f'{mape_j:>10.4f} {wmape_j:>10.4f}{marker}')

    print('-' * 55)
    print(f'{"Average":<12} {np.mean(all_mae):>10.4f} {np.mean(all_rmse):>10.4f} '
          f'{np.mean(all_mape):>10.4f} {np.mean(all_wmape):>10.4f}')

    # --- 关键区间汇总 ---
    print()
    print('=== 关键预测区间汇总 ===')
    print(f'{"区间":<12} {"Horizon":>8} {"MAE":>10} {"RMSE":>10} {"MAPE":>10}')
    print('-' * 52)
    for step, label in sorted(key_horizons.items()):
        if step <= args.output_len:
            idx = step - 1
            print(f'{label:<12} {step:>8} {all_mae[idx]:>10.4f} '
                  f'{all_rmse[idx]:>10.4f} {all_mape[idx]:>10.4f}')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
