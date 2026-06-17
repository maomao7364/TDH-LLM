import os as _os
if not _os.environ.get('OMP_NUM_THREADS'):
    _os.environ['OMP_NUM_THREADS'] = '1'

import torch
import numpy as np
import pandas as pd
import argparse
import time
import os
import sys
import random

# Ensure the DeST-LLM/ package directory is on sys.path so imports work
# regardless of whether this script is run from the repo root or DeST-LLM/
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from util import load_dataset, metric, MAE_torch, MAPE_torch, RMSE_torch, WMAPE_torch
from util import load_adj_from_pkl, load_adj_from_npy, build_correlation_adj
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from ranger21 import Ranger

from model.dest_llm import DeSTLLM


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeST-LLM training')
# 数据
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='pems07_custom', help='data/ 下的数据集目录名')
parser.add_argument('--adj', type=str, default=None, help='邻接矩阵路径 (.npy 或 .pkl)')
parser.add_argument('--input_dim', type=int, default=3, help='输入特征维度')
parser.add_argument('--num_nodes', type=int, default=50, help='路网节点数')
parser.add_argument('--input_len', type=int, default=24, help='输入时间步长 (24=2h)')
parser.add_argument('--output_len', type=int, default=24, help='预测时间步长 (24=120min)')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gpt2_path', type=str, default=None,
                    help='GPT-2 模型本地路径（目录），为空则从 HuggingFace 加载')

# 模型结构
parser.add_argument('--hidden_dim', type=int, default=256, help='MTD / TCDG 隐层维度')
parser.add_argument('--time_dim', type=int, default=128, help='时间嵌入维度')
parser.add_argument('--gpt_dim', type=int, default=768, help='GPT-2 隐层维度')
parser.add_argument('--gpt_layers', type=int, default=4, help='使用的 GPT-2 层数')
parser.add_argument('--U', type=int, default=2, help='PFA 中顶部解冻的 attention 层数')
parser.add_argument('--tcdg_layers', type=int, default=2, help='TCDG 堆叠层数')
parser.add_argument('--dropout', type=float, default=0.15, help='Dropout 比例')
parser.add_argument('--trend_kernel', type=int, default=5, help='MTD 趋势分解卷积核大小')
parser.add_argument('--period_kernel', type=int, default=7, help='MTD 周期分解卷积核大小')

# 训练
parser.add_argument('--lrate', type=float, default=5e-4, help='学习率')
parser.add_argument('--wdecay', type=float, default=1e-4, help='权重衰减')
parser.add_argument('--epochs', type=int, default=300, help='最大训练轮数')

parser.add_argument('--es_patience', type=int, default=100,
                    help='早停耐心值：超过该轮数无改善则停止')
parser.add_argument('--test_every', type=int, default=20,
                    help='每 N 轮在测试集上评估一次 (0=仅在 best 时测试)')
parser.add_argument('--num_workers', type=int, default=2,
                    help='DataLoader 预取进程数（0=主进程加载）')
parser.add_argument('--no_amp', action='store_true',
                    help='禁用混合精度训练')
parser.add_argument('--save', type=str, default=None)

args = None  # resolved in __main__


# ---------------------------------------------------------------------------
# 时间槽数量：PeMS 系列数据集均为 5 分钟采样，一天 288 个时间步
# ---------------------------------------------------------------------------
def get_time_bins(num_nodes, data_name):
    return 288


# ---------------------------------------------------------------------------
# Trainer：封装优化器 Ranger、损失 MAE、AMP 混合精度、梯度裁剪
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, scaler, lrate, wdecay, device, epochs=300,
                 warmup_epochs=10, clip=5, use_amp=True):
        self.model = model.to(device)
        self.loss_fn = MAE_torch
        self.data_scaler = scaler
        self.clip = clip
        self.use_amp = use_amp
        self.amp_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        # Grouped LR: GPT-2 pretrained weights get smaller LR
        gpt_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('gpt.'):
                gpt_params.append(p)
            else:
                other_params.append(p)

        gpt_lr = min(lrate * 0.1, 5e-5)
        self.optimizer = Ranger([
            {'params': other_params, 'lr': lrate, 'weight_decay': wdecay},
            {'params': gpt_params, 'lr': gpt_lr, 'weight_decay': wdecay},
        ])
        print(f'LR: other={lrate}, GPT-2={gpt_lr}  |  '
              f'Params: other={sum(p.numel() for p in other_params):,}, '
              f'GPT-2={sum(p.numel() for p in gpt_params):,}')

        # LR schedule: warmup → cosine annealing
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs)
        self.scheduler = SequentialLR(self.optimizer,
                                     schedulers=[warmup, cosine],
                                     milestones=[warmup_epochs])

        print(f'Param count: {model.param_num():,}')
        print(f'AMP: {use_amp}')
        print(f'LR schedule: warmup {warmup_epochs} epochs → cosine to epoch {epochs}')
        print(model)

    def train_step(self, input_, real_val):
        """单步训练：AMP 前向 + 反向 + 梯度裁剪 + 优化器更新"""
        self.model.train()
        self.optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            output = self.model(input_)
            output = output.transpose(1, 3)                          # -> [B, 1, N, out_len]
            real = torch.unsqueeze(real_val, dim=1)                  # -> [B, 1, N, out_len]
            predict = self.data_scaler.inverse_transform(output)     # 反归一化后再算指标
            loss = self.loss_fn(predict, real, 0.0)

        if self.use_amp:
            self.amp_scaler.scale(loss).backward()
            if self.clip is not None:
                self.amp_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

        with torch.no_grad():
            mape = MAPE_torch(predict, real, 0.0).item()
            rmse = RMSE_torch(predict, real, 0.0).item()
            wmape = WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    @torch.no_grad()
    def eval_step(self, input_, real_val):
        """单步评估（不计算梯度）"""
        self.model.eval()
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            output = self.model(input_)
            output = output.transpose(1, 3)
            real = torch.unsqueeze(real_val, dim=1)
            predict = self.data_scaler.inverse_transform(output)
            loss = self.loss_fn(predict, real, 0.0)
        mape = MAPE_torch(predict, real, 0.0).item()
        rmse = RMSE_torch(predict, real, 0.0).item()
        wmape = WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape


# ---------------------------------------------------------------------------
# 固定随机种子，保证可复现
# ---------------------------------------------------------------------------
def seed_it(seed):
    random.seed(seed)
    os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 测试评估（全量累积，与 test.py 一致）
# ---------------------------------------------------------------------------
def run_test_eval(model, dataloader, scaler, device, output_len, use_amp, title='Test'):
    model.eval()
    real_y = torch.Tensor(dataloader['y_test']).to(device)
    real_y = real_y.transpose(1, 3)[:, 0, :, :]

    preds = []
    for x, _ in dataloader['test_loader']:
        test_x = x.float().to(device, non_blocking=True)
        test_x = test_x.transpose(1, 3)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            out = model(test_x).transpose(1, 3).squeeze(1)
        preds.append(out)

    y_hat = torch.cat(preds, dim=0)
    y_hat = y_hat[:real_y.size(0), ...]

    key_horizons = {3: '15min', 6: '30min', 12: '60min', 18: '90min', 24: '120min'}
    print(f'\n  === {title} ===')
    print(f'  {"Horizon":<12} {"MAE":>10} {"RMSE":>10} {"MAPE":>10} {"WMAPE":>10}')
    print(f'  {"-" * 52}')

    horizon_mae, horizon_mape, horizon_rmse, horizon_wmape = [], [], [], []
    for j in range(output_len):
        pred_j = scaler.inverse_transform(y_hat[:, :, j])
        real_j = real_y[:, :, j]
        mae_j, mape_j, rmse_j, wmape_j = metric(pred_j, real_j)
        horizon_mae.append(mae_j)
        horizon_mape.append(mape_j)
        horizon_rmse.append(rmse_j)
        horizon_wmape.append(wmape_j)

        step = j + 1
        marker = f'  <-- {key_horizons[step]}' if step in key_horizons else ''
        print(f'  {step:>2} ({step * 5:>3}min) {mae_j:>10.4f} {rmse_j:>10.4f} '
              f'{mape_j:>10.4f} {wmape_j:>10.4f}{marker}')

    avg_mae = np.mean(horizon_mae)
    avg_rmse = np.mean(horizon_rmse)
    avg_mape = np.mean(horizon_mape)
    avg_wmape = np.mean(horizon_wmape)
    print(f'  {"Average":<12} {avg_mae:>10.4f} {avg_rmse:>10.4f} '
          f'{avg_mape:>10.4f} {avg_wmape:>10.4f}\n')

    return avg_mae, avg_rmse, avg_mape, avg_wmape, horizon_mae, horizon_rmse, horizon_mape, horizon_wmape


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------
def main():
    seed_it(6666)

    # --- 解析数据集路径 ---
    data_name = args.data
    if args.data == 'pems07_custom':
        args.num_nodes = 50
        data_dir = os.path.join(_script_dir, 'data', 'pems07_custom')
    else:
        data_dir = (os.path.join(_script_dir, 'data', args.data)
                    if not os.path.isabs(args.data) else args.data)

    device = torch.device(args.device)

    # --- GPT-2 模型路径（改成你的实际路径） ---
    if args.gpt2_path is None:
        args.gpt2_path = os.path.join(_script_dir, 'snapshots',
                                      '607a30d783dfa663caf39e06633721c8d4cfcd7e')

    # --- 加载数据（train/val/test + scaler） ---
    dataloader = load_dataset(data_dir, args.batch_size, args.batch_size, args.batch_size,
                              num_workers=args.num_workers)
    scaler = dataloader['scaler']

    # --- 构建 / 加载邻接矩阵 ---
    # 优先级：--adj 指定 > data/ 下自动发现 > 用训练数据计算相关性矩阵
    time_bins = get_time_bins(args.num_nodes, data_name)

    if args.adj is not None and os.path.exists(args.adj):
        if args.adj.endswith('.npy'):
            adj = load_adj_from_npy(args.adj)
        else:
            adj = load_adj_from_pkl(args.adj)
        print(f'Loaded adjacency from {args.adj}, shape={adj.shape}')
    else:
        auto_adj_npy = os.path.join(data_dir, 'adj.npy')
        auto_adj_pkl = os.path.join(data_dir, 'adj.pkl')
        if os.path.exists(auto_adj_npy):
            adj = load_adj_from_npy(auto_adj_npy)
            print(f'Loaded adjacency from {auto_adj_npy}, shape={adj.shape}')
        elif os.path.exists(auto_adj_pkl):
            adj = load_adj_from_pkl(auto_adj_pkl)
            print(f'Loaded adjacency from {auto_adj_pkl}, shape={adj.shape}')
        else:
            print('No --adj provided, building correlation adjacency from training data ...')
            adj = build_correlation_adj(dataloader['x_train'])
            print(f'Built correlation adjacency, shape={adj.shape}')

    # --- 初始化模型 ---
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
    )

    # --- 创建 Trainer ---
    engine = Trainer(model, scaler, args.lrate, args.wdecay, device,
                     epochs=args.epochs, use_amp=not args.no_amp)

    torch.cuda.empty_cache()
    mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f'GPU memory: allocated={mem_alloc:.2f} GiB, reserved={mem_reserved:.2f} GiB')

    # --- 日志保存路径 ---
    if args.save is None:
        args.save = os.path.join(_script_dir, 'logs',
                                 time.strftime('%Y-%m-%d-%H-%M-%S') + '-' + data_name) + '/'
    save_path = args.save
    os.makedirs(save_path, exist_ok=True)

    # --- 训练状态变量 ---
    loss_best = float('inf')
    test_best = float('inf')
    epochs_no_best = 0
    best_epoch = 0
    result_log = []
    test_log = []
    train_times = []
    val_times = []

    print(args)
    print('Start training ...', flush=True)

    for epoch in range(1, args.epochs + 1):
        # ---- 训练阶段 ----
        train_losses, train_mapes, train_rmses, train_wmapes = [], [], [], []
        t1 = time.time()
        for x, y in dataloader['train_loader']:
            train_x = x.float().to(device, non_blocking=True)   # [B, T, N, C]
            train_x = train_x.transpose(1, 3)                   # [B, C, N, T]
            train_y = y.float().to(device, non_blocking=True)   # [B, T, N, C]
            train_y = train_y.transpose(1, 3)                   # [B, C, N, T]
            # train_y[:, 0, :, :] 取第一个特征（流量）作为回归目标
            loss, mape, rmse, wmape = engine.train_step(train_x, train_y[:, 0, :, :])
            train_losses.append(loss)
            train_mapes.append(mape)
            train_rmses.append(rmse)
            train_wmapes.append(wmape)
        t2 = time.time()
        train_times.append(t2 - t1)

        # ---- 验证阶段 ----
        valid_losses, valid_mapes, valid_rmses, valid_wmapes = [], [], [], []
        s1 = time.time()
        for x, y in dataloader['val_loader']:
            val_x = x.float().to(device, non_blocking=True)
            val_x = val_x.transpose(1, 3)
            val_y = y.float().to(device, non_blocking=True)
            val_y = val_y.transpose(1, 3)
            loss, mape, rmse, wmape = engine.eval_step(val_x, val_y[:, 0, :, :])
            valid_losses.append(loss)
            valid_mapes.append(mape)
            valid_rmses.append(rmse)
            valid_wmapes.append(wmape)
        s2 = time.time()
        val_times.append(s2 - s1)

        m_train_loss = np.mean(train_losses)
        m_valid_loss = np.mean(valid_losses)

        print('-' * 55)
        print(f'Epoch {epoch:03d} | Train Loss: {m_train_loss:.4f}  '
              f'RMSE: {np.mean(train_rmses):.4f}  MAPE: {np.mean(train_mapes):.4f}  '
              f'WMAPE: {np.mean(train_wmapes):.4f}')
        print(f'          | Valid Loss: {m_valid_loss:.4f}  '
              f'RMSE: {np.mean(valid_rmses):.4f}  MAPE: {np.mean(valid_mapes):.4f}  '
              f'WMAPE: {np.mean(valid_wmapes):.4f}')

        result_log.append(dict(
            epoch=epoch,
            train_loss=m_train_loss,
            train_rmse=np.mean(train_rmses),
            train_mape=np.mean(train_mapes),
            train_wmape=np.mean(train_wmapes),
            valid_loss=m_valid_loss,
            valid_rmse=np.mean(valid_rmses),
            valid_mape=np.mean(valid_mapes),
            valid_wmape=np.mean(valid_wmapes),
        ))

        # ---- 模型保存 ----
        is_best = m_valid_loss < loss_best
        if is_best:
            loss_best = m_valid_loss
            torch.save(engine.model.state_dict(), save_path + 'best_model.tmp')
            os.replace(save_path + 'best_model.tmp', save_path + 'best_model.pth')
            best_epoch = epoch
            epochs_no_best = 0
            print(f'  -> Best model updated (epoch {epoch}, loss {m_valid_loss:.4f})')
        else:
            epochs_no_best += 1

        # ---- 测试集评估：best 时 或 每 N 轮 ----
        do_test = (epoch > 1 and
                   (is_best or (args.test_every > 0 and epoch % args.test_every == 0)))

        if do_test:
            torch.cuda.empty_cache()
            avg_mae, avg_rmse, avg_mape, avg_wmape, _, _, _, _ = run_test_eval(
                engine.model, dataloader, scaler, device,
                args.output_len, not args.no_amp,
                title=f'Epoch {epoch:03d}')

            test_log.append(dict(
                epoch=epoch, test_mae=avg_mae, test_rmse=avg_rmse,
                test_mape=avg_mape, test_wmape=avg_wmape,
            ))

            if avg_mae < test_best:
                test_best = avg_mae

            engine.model.train()

        # ---- LR schedule step ----
        engine.scheduler.step()

        # ---- 早停 ----
        if epochs_no_best >= args.es_patience and epoch >= 50:
            print(f'Early stop at epoch {epoch}')
            break

    # ---- final test with best model ----
    print(f'\nTraining finished. Best epoch: {best_epoch}, Best valid loss: {loss_best:.4f}')
    print(f'Avg training time/epoch: {np.mean(train_times):.4f}s')
    print(f'Avg inference time:       {np.mean(val_times):.4f}s')

    best_path = save_path + 'best_model.pth'
    if os.path.exists(best_path):
        engine.model.load_state_dict(torch.load(best_path, map_location=device))
        print(f'Loaded best model from epoch {best_epoch} for final test.')
        final_mae, final_rmse, final_mape, final_wmape, h_mae, h_rmse, h_mape, h_wmape = run_test_eval(
            engine.model, dataloader, scaler, device,
            args.output_len, not args.no_amp,
            title=f'Final Test (best epoch {best_epoch})')

        final_row = {'best_epoch': best_epoch,
                     'test_mae': final_mae, 'test_rmse': final_rmse,
                     'test_mape': final_mape, 'test_wmape': final_wmape}
        for j in range(len(h_mae)):
            final_row[f'h{j+1}_mae'] = round(h_mae[j], 6)
            final_row[f'h{j+1}_rmse'] = round(h_rmse[j], 6)
            final_row[f'h{j+1}_mape'] = round(h_mape[j], 6)
            final_row[f'h{j+1}_wmape'] = round(h_wmape[j], 6)
        pd.DataFrame([final_row]).round(8).to_csv(save_path + 'final_test.csv', index=False)
    else:
        print('Warning: best_model.pth not found, skipping final test.')

    # save logs
    pd.DataFrame(result_log).round(8).to_csv(save_path + 'train_log.csv', index=False)
    if test_log:
        pd.DataFrame(test_log).round(8).to_csv(save_path + 'test_log.csv', index=False)

    print(f'Logs saved to {save_path}')


if __name__ == '__main__':
    args = parser.parse_args()

    # --- startup banner ---
    print(f'Dataset : {args.data}  |  Nodes: {args.num_nodes}  |  '
          f'Horizon: {args.output_len} steps ({args.output_len * 5} min)  |  '
          f'Batch: {args.batch_size}')
    if args.save:
        print(f'Save to : {args.save}')

    torch.cuda.empty_cache()
    t_start = time.time()
    main()
    print(f'Total time: {time.time() - t_start:.2f}s')
