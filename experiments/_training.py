"""
Shared training infrastructure for DeST-LLM experiments.

Not meant to be run directly — imported by individual experiment scripts.
"""
import torch
import numpy as np
import pandas as pd
import argparse
import time
import os
import sys
import random

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from util import (
    load_dataset, StandardScaler, metric,
    MAE_torch, MAPE_torch, RMSE_torch, WMAPE_torch,
    load_adj_from_pkl, load_adj_from_npy, build_correlation_adj,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from ranger21 import Ranger
from experiments.ablation_models import AblationDeSTLLM


# ---------------------------------------------------------------------------
# CLI shared by all experiment scripts
# ---------------------------------------------------------------------------
def make_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--data', type=str, default='pems07_custom')
    p.add_argument('--adj', type=str, default=None)
    p.add_argument('--input_dim', type=int, default=3)
    p.add_argument('--num_nodes', type=int, default=50)
    p.add_argument('--input_len', type=int, default=36)
    p.add_argument('--output_len', type=int, default=24)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--gpt2_path', type=str, default=None,
                   help='GPT-2 模型本地路径')
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--time_dim', type=int, default=128)
    p.add_argument('--gpt_layers', type=int, default=4)
    p.add_argument('--U', type=int, default=2)
    p.add_argument('--tcdg_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.15)
    p.add_argument('--trend_kernel', type=int, default=5)
    p.add_argument('--period_kernel', type=int, default=7)
    p.add_argument('--lrate', type=float, default=5e-4)
    p.add_argument('--wdecay', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--es_patience', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=2,
                   help='DataLoader num_workers (0=main process only)')
    p.add_argument('--no_amp', action='store_true',
                   help='disable AMP mixed precision')
    p.add_argument('--save', type=str,
                    default=os.path.join(_parent_dir, 'logs',
                                         time.strftime('%Y-%m-%d-%H-%M-%S')) + '/')
    return p


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_time_bins(num_nodes, data_name):
    if num_nodes in (170, 307, 358, 883) or 'pems' in data_name.lower():
        return 288
    if num_nodes in (250, 266) or 'bike' in data_name.lower() or 'taxi' in data_name.lower():
        return 48
    if num_nodes > 200:
        return 96
    return 288


def seed_it(seed=6666):
    random.seed(seed)
    os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def resolve_dataset(args, parent_dir):
    if args.data == 'pems07_custom':
        args.num_nodes = 50
        return os.path.join(parent_dir, 'data', 'pems07_custom')
    return os.path.join(parent_dir, 'data', args.data) if not os.path.isabs(args.data) else args.data


def load_adjacency(args, data_dir, dataloader):
    if args.adj and os.path.exists(args.adj):
        adj = load_adj_from_npy(args.adj) if args.adj.endswith('.npy') else load_adj_from_pkl(args.adj)
    else:
        auto_adj_npy = os.path.join(data_dir, 'adj.npy')
        auto_adj_pkl = os.path.join(data_dir, 'adj.pkl')
        if os.path.exists(auto_adj_npy):
            adj = load_adj_from_npy(auto_adj_npy)
        elif os.path.exists(auto_adj_pkl):
            adj = load_adj_from_pkl(auto_adj_pkl)
        else:
            adj = build_correlation_adj(dataloader['x_train'])
    print(f'Adjacency shape: {adj.shape}')
    return adj


# ---------------------------------------------------------------------------
# Trainer (with AMP mixed precision)
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, scaler, lrate, wdecay, device, epochs=300,
                 warmup_epochs=10, clip=5, use_amp=True):
        self.model = model.to(device)
        self.optimizer = Ranger(model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss_fn = MAE_torch
        self.data_scaler = scaler
        self.clip = clip
        self.use_amp = use_amp
        self.amp_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs)
        self.scheduler = SequentialLR(self.optimizer,
                                     schedulers=[warmup, cosine],
                                     milestones=[warmup_epochs])

        print(f'Trainable params: {model.param_num():,}')
        print(f'AMP: {use_amp}')
        print(f'LR schedule: warmup {warmup_epochs} epochs -> cosine to epoch {epochs}')

    def train_step(self, input_, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            output = self.model(input_)
            output = output.transpose(1, 3)
            real = torch.unsqueeze(real_val, dim=1)
            predict = self.data_scaler.inverse_transform(output)
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
# Main training loop
# ---------------------------------------------------------------------------
def run_experiment(args, variant, backbone_type, exp_name=None):
    """Run one experiment with given variant and backbone. Returns summary dict."""
    if exp_name is None:
        exp_name = f'{variant}_{backbone_type}'
    print(f'\n{"=" * 70}')
    print(f'Experiment: {exp_name}')
    print(f'{"=" * 70}\n')

    data_dir = resolve_dataset(args, _parent_dir)
    device = torch.device(args.device)

    # Auto-detect local GPT-2 model
    if args.gpt2_path is None:
        snapshots_dir = os.path.join(_parent_dir, 'snapshots')
        if os.path.isdir(snapshots_dir):
            dirs = [d for d in os.listdir(snapshots_dir)
                    if os.path.isdir(os.path.join(snapshots_dir, d))]
            if dirs:
                args.gpt2_path = os.path.join(snapshots_dir, dirs[0])
                print(f'Auto-detected GPT-2 at {args.gpt2_path}')

    time_bins = get_time_bins(args.num_nodes, args.data)
    save_path = os.path.join(args.save, args.data, exp_name) + '/'
    os.makedirs(save_path, exist_ok=True)

    # Data
    dataloader = load_dataset(data_dir, args.batch_size, args.batch_size, args.batch_size,
                              num_workers=args.num_workers)
    scaler = dataloader['scaler']
    adj = load_adjacency(args, data_dir, dataloader)

    # Model
    model = AblationDeSTLLM(
        adj=adj,
        input_dim=args.input_dim, hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes, input_len=args.input_len,
        output_len=args.output_len, time_bins=time_bins,
        time_dim=args.time_dim, trend_kernel=args.trend_kernel,
        period_kernel=args.period_kernel,
        backbone_type=backbone_type, gpt_layers=args.gpt_layers, U=args.U,
        variant=variant,
        tcdg_layers=args.tcdg_layers, dropout=args.dropout,
        gpt2_path=args.gpt2_path,
    )

    engine = Trainer(model, scaler, args.lrate, args.wdecay, device,
                     epochs=args.epochs, use_amp=not args.no_amp)

    loss_best = float('inf')
    test_best_mae = float('inf')
    epochs_no_best = 0
    best_epoch = 0
    result_log = []
    test_log = []
    train_times = []

    print('Start training ...', flush=True)

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        train_losses, train_mapes, train_rmses, train_wmapes = [], [], [], []
        t1 = time.time()
        for x, y in dataloader['train_loader']:
            train_x = x.float().to(device, non_blocking=True)
            train_x = train_x.transpose(1, 3)
            train_y = y.float().to(device, non_blocking=True)
            train_y = train_y.transpose(1, 3)
            loss, mape, rmse, wmape = engine.train_step(train_x, train_y[:, 0, :, :])
            train_losses.append(loss)
            train_mapes.append(mape)
            train_rmses.append(rmse)
            train_wmapes.append(wmape)
        t2 = time.time()
        train_times.append(t2 - t1)

        # ---- validate ----
        valid_losses, valid_mapes, valid_rmses, valid_wmapes = [], [], [], []
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

        m_train_loss = np.mean(train_losses)
        m_valid_loss = np.mean(valid_losses)

        print(f'Epoch {epoch:03d} | Train Loss: {m_train_loss:.4f}  '
              f'Valid Loss: {m_valid_loss:.4f}  '
              f'RMSE: {np.mean(valid_rmses):.4f}  MAPE: {np.mean(valid_mapes):.4f}')

        result_log.append(dict(
            epoch=epoch,
            train_loss=m_train_loss, valid_loss=m_valid_loss,
            valid_rmse=np.mean(valid_rmses), valid_mape=np.mean(valid_mapes),
            valid_wmape=np.mean(valid_wmapes),
        ))

        # ---- checkpoint + test ----
        if m_valid_loss < loss_best:
            loss_best = m_valid_loss
            torch.save(engine.model.state_dict(), save_path + 'best_model.pth')
            best_epoch = epoch
            epochs_no_best = 0

            if epoch > 100:
                engine.model.eval()
                torch.cuda.empty_cache()

                H = args.output_len
                sum_ae = np.zeros(H)
                sum_se = np.zeros(H)
                sum_ape = np.zeros(H)
                sum_real = np.zeros(H)
                count = np.zeros(H, dtype=np.int64)

                for x, y in dataloader['test_loader']:
                    test_x = x.float().to(device, non_blocking=True)
                    test_x = test_x.transpose(1, 3)
                    real = y.float().to(device, non_blocking=True)
                    real = real.transpose(1, 3)[:, 0, :, :]

                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=not args.no_amp):
                        out = engine.model(test_x).transpose(1, 3).squeeze(1)

                    for j in range(H):
                        pred_j = scaler.inverse_transform(out[:, :, j])
                        real_j = real[:, :, j]
                        mask = real_j != 0
                        diff = (pred_j - real_j).abs() * mask
                        sum_ae[j] += diff.sum().item()
                        sum_se[j] += (diff ** 2).sum().item()
                        sum_ape[j] += (diff / (real_j.abs() + 1e-8) * mask).sum().item()
                        sum_real[j] += (real_j * mask).sum().item()
                        count[j] += mask.sum().item()

                key_h = {3: '15min', 6: '30min', 12: '60min', 18: '90min', 24: '120min'}
                print(f'\n  {"Horizon":<12} {"MAE":>10} {"RMSE":>10} '
                      f'{"MAPE":>10} {"WMAPE":>10}')
                print(f'  {"-" * 52}')

                row = {'epoch': epoch}
                horizon_mae, horizon_mape, horizon_rmse, horizon_wmape = [], [], [], []
                for j in range(H):
                    n = max(count[j], 1)
                    mae_j = sum_ae[j] / n
                    rmse_j = np.sqrt(sum_se[j] / n)
                    mape_j = sum_ape[j] / n
                    wmape_j = sum_ae[j] / (sum_real[j] + 1e-8)
                    horizon_mae.append(mae_j)
                    horizon_mape.append(mape_j)
                    horizon_rmse.append(rmse_j)
                    horizon_wmape.append(wmape_j)

                    step = j + 1
                    marker = f'  <-- {key_h[step]}' if step in key_h else ''
                    print(f'  {step:>2} ({step*5:>3}min) {mae_j:>10.4f} '
                          f'{rmse_j:>10.4f} {mape_j:>10.4f} {wmape_j:>10.4f}{marker}')
                    row[f'h{step}_mae'] = round(mae_j, 6)
                    row[f'h{step}_rmse'] = round(rmse_j, 6)
                    row[f'h{step}_mape'] = round(mape_j, 6)
                    row[f'h{step}_wmape'] = round(wmape_j, 6)

                avg_mae = np.mean(horizon_mae)
                avg_rmse = np.mean(horizon_rmse)
                avg_mape = np.mean(horizon_mape)
                avg_wmape = np.mean(horizon_wmape)
                print(f'  {"Average":<12} {avg_mae:>10.4f} {avg_rmse:>10.4f} '
                      f'{avg_mape:>10.4f} {avg_wmape:>10.4f}\n')

                row.update(test_mae=avg_mae, test_rmse=avg_rmse,
                          test_mape=avg_mape, test_wmape=avg_wmape)
                test_log.append(row)
                if avg_mae < test_best_mae:
                    test_best_mae = avg_mae
        else:
            epochs_no_best += 1

        engine.scheduler.step()

        if epochs_no_best >= args.es_patience and epoch >= 50:
            print(f'Early stop at epoch {epoch}')
            break

    print(f'\n{exp_name} finished. Best epoch: {best_epoch}, '
          f'Best test MAE: {test_best_mae:.4f}')

    # ---- final test with best model ----
    best_path = save_path + 'best_model.pth'
    if os.path.exists(best_path):
        engine.model.load_state_dict(torch.load(best_path, map_location=device))
        engine.model.eval()
        torch.cuda.empty_cache()
        print(f'Loaded best model from epoch {best_epoch} for final test.')

        H = args.output_len
        sum_ae = np.zeros(H)
        sum_se = np.zeros(H)
        sum_ape = np.zeros(H)
        sum_real = np.zeros(H)
        count = np.zeros(H, dtype=np.int64)

        for x, y in dataloader['test_loader']:
            test_x = x.float().to(device, non_blocking=True)
            test_x = test_x.transpose(1, 3)
            real = y.float().to(device, non_blocking=True)
            real = real.transpose(1, 3)[:, 0, :, :]

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=not args.no_amp):
                out = engine.model(test_x).transpose(1, 3).squeeze(1)

            for j in range(H):
                pred_j = scaler.inverse_transform(out[:, :, j])
                real_j = real[:, :, j]
                mask = real_j != 0
                diff = (pred_j - real_j).abs() * mask
                sum_ae[j] += diff.sum().item()
                sum_se[j] += (diff ** 2).sum().item()
                sum_ape[j] += (diff / (real_j.abs() + 1e-8) * mask).sum().item()
                sum_real[j] += (real_j * mask).sum().item()
                count[j] += mask.sum().item()

        key_h = {3: '15min', 6: '30min', 12: '60min', 18: '90min', 24: '120min'}
        print(f'\n  === Final Test (best epoch {best_epoch}) ===')
        print(f'  {"Horizon":<12} {"MAE":>10} {"RMSE":>10} '
              f'{"MAPE":>10} {"WMAPE":>10}')
        print(f'  {"-" * 52}')

        final_row = {'experiment': exp_name, 'best_epoch': best_epoch}
        final_mae, final_mape, final_rmse, final_wmape = [], [], [], []
        for j in range(H):
            n = max(count[j], 1)
            mae_j = sum_ae[j] / n
            rmse_j = np.sqrt(sum_se[j] / n)
            mape_j = sum_ape[j] / n
            wmape_j = sum_ae[j] / (sum_real[j] + 1e-8)
            final_mae.append(mae_j)
            final_mape.append(mape_j)
            final_rmse.append(rmse_j)
            final_wmape.append(wmape_j)

            step = j + 1
            marker = f'  <-- {key_h[step]}' if step in key_h else ''
            print(f'  {step:>2} ({step*5:>3}min) {mae_j:>10.4f} '
                  f'{rmse_j:>10.4f} {mape_j:>10.4f} {wmape_j:>10.4f}{marker}')
            final_row[f'h{step}_mae'] = round(mae_j, 6)
            final_row[f'h{step}_rmse'] = round(rmse_j, 6)
            final_row[f'h{step}_mape'] = round(mape_j, 6)
            final_row[f'h{step}_wmape'] = round(wmape_j, 6)

        avg_mae_f = np.mean(final_mae)
        avg_rmse_f = np.mean(final_rmse)
        avg_mape_f = np.mean(final_mape)
        avg_wmape_f = np.mean(final_wmape)
        print(f'  {"Average":<12} {avg_mae_f:>10.4f} {avg_rmse_f:>10.4f} '
              f'{avg_mape_f:>10.4f} {avg_wmape_f:>10.4f}\n')

        final_row.update(test_mae=avg_mae_f, test_rmse=avg_rmse_f,
                        test_mape=avg_mape_f, test_wmape=avg_wmape_f)
        pd.DataFrame([final_row]).round(8).to_csv(save_path + 'final_test.csv', index=False)
    else:
        print('Warning: best_model.pth not found, skipping final test.')

    pd.DataFrame(result_log).round(8).to_csv(save_path + 'train_log.csv', index=False)
    if test_log:
        pd.DataFrame(test_log).round(8).to_csv(save_path + 'test_log.csv', index=False)

    return {
        'experiment': exp_name, 'variant': variant, 'llm': backbone_type,
        'best_epoch': best_epoch,
        'best_valid_loss': round(loss_best, 6),
        'best_test_mae': round(test_best_mae, 6),
        'best_test_rmse': round(avg_rmse_f, 6),
        'best_test_mape': round(avg_mape_f, 6),
        'best_test_wmape': round(avg_wmape_f, 6),
        'avg_train_time': round(np.mean(train_times), 2),
        'params': model.param_num(),
    }
