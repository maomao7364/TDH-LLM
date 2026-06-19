"""
Reference ablation training for the current daily + weekly DeST-LLM model.

The main train.py remains the full production path. This module runs controlled
experiments that mask daily and/or weekly references while keeping all other
training settings and evaluation code aligned.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from dataset_checks import validate_dataset_shapes
from model.dest_llm import DeSTLLM
from ranger21 import Ranger
from util import (
    MAE_torch,
    MAPE_torch,
    RMSE_torch,
    WMAPE_torch,
    build_correlation_adj,
    load_adj_from_npy,
    load_adj_from_pkl,
    load_dataset,
    metric,
    topk_adj,
)


REFERENCE_MODES = ('full_refs', 'no_daily', 'no_weekly', 'no_refs')


def make_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--data', type=str, default='pems07_custom')
    p.add_argument('--adj', type=str, default=None)
    p.add_argument('--topk', type=int, default=0)
    p.add_argument('--input_dim', type=int, default=5)
    p.add_argument('--num_nodes', type=int, default=50)
    p.add_argument('--input_len', type=int, default=24)
    p.add_argument('--output_len', type=int, default=24)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--gpt2_path', type=str, default=None)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--time_dim', type=int, default=32)
    p.add_argument('--gpt_dim', type=int, default=768)
    p.add_argument('--gpt_layers', type=int, default=4)
    p.add_argument('--U', type=int, default=1)
    p.add_argument('--tcdg_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.20)
    p.add_argument('--trend_kernel', type=int, default=5)
    p.add_argument('--period_kernel', type=int, default=7)
    p.add_argument('--lrate', type=float, default=4e-4)
    p.add_argument('--wdecay', type=float, default=2e-4)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--es_patience', type=int, default=50)
    p.add_argument('--test_every', type=int, default=20)
    default_workers = 0 if os.name == 'nt' else 2
    p.add_argument('--num_workers', type=int, default=default_workers)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--save', type=str, default=None)
    return p


def seed_it(seed=6666):
    random.seed(seed)
    os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def resolve_dataset(args):
    if args.data == 'pems07_custom':
        args.num_nodes = 50
        return os.path.join(_parent_dir, 'data', 'pems07_custom')
    return os.path.join(_parent_dir, 'data', args.data) if not os.path.isabs(args.data) else args.data


def load_adjacency(args, data_dir, dataloader):
    if args.adj is not None and os.path.exists(args.adj):
        adj = load_adj_from_npy(args.adj) if args.adj.endswith('.npy') else load_adj_from_pkl(args.adj)
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
            adj = build_correlation_adj(dataloader['x_train'])
            print(f'Built correlation adjacency, shape={adj.shape}')

    if args.topk > 0:
        adj = topk_adj(adj, args.topk)
        nz = (adj > 0).sum()
        print(f'Top-{args.topk} adjacency: {nz} non-zero edges ({nz / (adj.shape[0] ** 2) * 100:.1f}%)')
    return adj


def mask_references(x, y_transposed, reference_mode):
    """Mask historical channels and future refs for one ablation mode."""
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f'Unknown reference_mode: {reference_mode}')

    x = x.clone()
    future_daily = y_transposed[:, 1, :, :] if y_transposed.size(1) > 1 else None
    future_weekly = y_transposed[:, 2, :, :] if y_transposed.size(1) > 2 else None

    if reference_mode in ('no_daily', 'no_refs') and x.size(1) > 3:
        x[:, 3, :, :] = 0
        future_daily = None
    if reference_mode in ('no_weekly', 'no_refs') and x.size(1) > 4:
        x[:, 4, :, :] = 0
        future_weekly = None

    return x, future_daily, future_weekly


class Trainer:
    def __init__(self, model, scaler, lrate, wdecay, device, epochs=300,
                 warmup_epochs=10, clip=5, use_amp=True):
        self.model = model.to(device)
        self.loss_fn = MAE_torch
        self.data_scaler = scaler
        self.clip = clip
        self.use_amp = use_amp
        self.amp_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        gpt_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('gpt.'):
                gpt_params.append(p)
            else:
                other_params.append(p)

        gpt_lr = min(lrate * 0.1, 1e-5)
        self.optimizer = Ranger([
            {'params': other_params, 'lr': lrate, 'weight_decay': wdecay},
            {'params': gpt_params, 'lr': gpt_lr, 'weight_decay': wdecay},
        ])

        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine],
                                      milestones=[warmup_epochs])
        print(f'LR: other={lrate}, GPT-2={gpt_lr}')
        print(f'Param count: {model.param_num():,}')
        print(f'AMP: {use_amp}')

    def train_step(self, input_, real_val, future_daily=None, future_weekly=None):
        self.model.train()
        self.optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            output = self.model(input_, future_daily=future_daily,
                                future_weekly=future_weekly)
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
    def eval_step(self, input_, real_val, future_daily=None, future_weekly=None):
        self.model.eval()
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            output = self.model(input_, future_daily=future_daily,
                                future_weekly=future_weekly)
            output = output.transpose(1, 3)
            real = torch.unsqueeze(real_val, dim=1)
            predict = self.data_scaler.inverse_transform(output)
            loss = self.loss_fn(predict, real, 0.0)
        mape = MAPE_torch(predict, real, 0.0).item()
        rmse = RMSE_torch(predict, real, 0.0).item()
        wmape = WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape


@torch.no_grad()
def run_test_eval(model, dataloader, scaler, device, output_len, use_amp,
                  reference_mode, title='Test'):
    model.eval()
    real_y = torch.Tensor(dataloader['y_test']).to(device)
    real_y = real_y.transpose(1, 3)[:, 0, :, :]

    preds = []
    for x, y in dataloader['test_loader']:
        test_x = x.float().to(device, non_blocking=True).transpose(1, 3)
        test_y = y.float().to(device, non_blocking=True).transpose(1, 3)
        test_x, test_fd, test_fw = mask_references(test_x, test_y, reference_mode)
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(test_x, future_daily=test_fd,
                        future_weekly=test_fw).transpose(1, 3).squeeze(1)
        preds.append(out)

    y_hat = torch.cat(preds, dim=0)[:real_y.size(0), ...]
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


def run_reference_experiment(args, reference_mode):
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f'Unknown reference_mode: {reference_mode}')
    seed_it(6666)
    data_dir = resolve_dataset(args)
    device = torch.device(args.device)
    data_name = args.data

    if args.gpt2_path is None:
        args.gpt2_path = os.path.join(
            _parent_dir, 'snapshots', '607a30d783dfa663caf39e06633721c8d4cfcd7e'
        )

    dataloader = load_dataset(data_dir, args.batch_size, args.batch_size, args.batch_size,
                              num_workers=args.num_workers)
    shape_info = validate_dataset_shapes(
        dataloader, input_len=args.input_len, output_len=args.output_len,
        input_dim=args.input_dim, num_nodes=args.num_nodes, target_dim=3,
    )
    print(f'Reference mode: {reference_mode}')
    print(f'Data shape: {shape_info}')
    scaler = dataloader['scaler']
    adj = load_adjacency(args, data_dir, dataloader)

    model = DeSTLLM(
        adj=adj,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        gpt_layers=args.gpt_layers,
        U=args.U,
        time_bins=288,
        time_dim=args.time_dim,
        gpt_dim=args.gpt_dim,
        trend_kernel=args.trend_kernel,
        period_kernel=args.period_kernel,
        tcdg_layers=args.tcdg_layers,
        dropout=args.dropout,
        use_horizon_decoder=True,
        gpt2_path=args.gpt2_path,
    )

    engine = Trainer(model, scaler, args.lrate, args.wdecay, device,
                     epochs=args.epochs, use_amp=not args.no_amp)

    if args.save is None:
        args.save = os.path.join(
            _parent_dir, 'logs', time.strftime('%Y-%m-%d-%H-%M-%S') + '-ablation'
        ) + '/'
    save_path = os.path.join(args.save, data_name, reference_mode) + '/'
    os.makedirs(save_path, exist_ok=True)

    model_args = vars(args).copy()
    model_args.update(reference_mode=reference_mode, use_horizon_decoder=True)
    with open(os.path.join(save_path, 'model_args.json'), 'w', encoding='utf-8') as f:
        json.dump(model_args, f, indent=2)

    loss_best = float('inf')
    test_best = float('inf')
    epochs_no_best = 0
    best_epoch = 0
    result_log = []
    test_log = []

    print('Start reference ablation training ...', flush=True)
    for epoch in range(1, args.epochs + 1):
        train_losses, train_mapes, train_rmses, train_wmapes = [], [], [], []
        for x, y in dataloader['train_loader']:
            train_x = x.float().to(device, non_blocking=True).transpose(1, 3)
            train_y = y.float().to(device, non_blocking=True).transpose(1, 3)
            train_x, train_fd, train_fw = mask_references(train_x, train_y, reference_mode)
            loss, mape, rmse, wmape = engine.train_step(
                train_x, train_y[:, 0, :, :], future_daily=train_fd,
                future_weekly=train_fw,
            )
            train_losses.append(loss)
            train_mapes.append(mape)
            train_rmses.append(rmse)
            train_wmapes.append(wmape)

        valid_losses, valid_mapes, valid_rmses, valid_wmapes = [], [], [], []
        for x, y in dataloader['val_loader']:
            val_x = x.float().to(device, non_blocking=True).transpose(1, 3)
            val_y = y.float().to(device, non_blocking=True).transpose(1, 3)
            val_x, val_fd, val_fw = mask_references(val_x, val_y, reference_mode)
            loss, mape, rmse, wmape = engine.eval_step(
                val_x, val_y[:, 0, :, :], future_daily=val_fd,
                future_weekly=val_fw,
            )
            valid_losses.append(loss)
            valid_mapes.append(mape)
            valid_rmses.append(rmse)
            valid_wmapes.append(wmape)

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

        is_best = m_valid_loss < loss_best
        if is_best:
            loss_best = m_valid_loss
            torch.save(engine.model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            best_epoch = epoch
            epochs_no_best = 0
            print(f'  -> Best model updated (epoch {epoch}, loss {m_valid_loss:.4f})')
        else:
            epochs_no_best += 1

        do_test = epoch > 1 and (is_best or (args.test_every > 0 and epoch % args.test_every == 0))
        if do_test:
            avg_mae, avg_rmse, avg_mape, avg_wmape, _, _, _, _ = run_test_eval(
                engine.model, dataloader, scaler, device, args.output_len,
                not args.no_amp, reference_mode, title=f'Epoch {epoch:03d}'
            )
            test_log.append(dict(
                epoch=epoch, test_mae=avg_mae, test_rmse=avg_rmse,
                test_mape=avg_mape, test_wmape=avg_wmape,
            ))
            test_best = min(test_best, avg_mae)
            engine.model.train()

        engine.scheduler.step()
        if epochs_no_best >= args.es_patience and epoch >= 50:
            print(f'Early stop at epoch {epoch}')
            break

    print(f'\nTraining finished. Best epoch: {best_epoch}, Best valid loss: {loss_best:.4f}')
    best_path = os.path.join(save_path, 'best_model.pth')
    if os.path.exists(best_path):
        engine.model.load_state_dict(torch.load(best_path, map_location=device))
        final_mae, final_rmse, final_mape, final_wmape, h_mae, h_rmse, h_mape, h_wmape = run_test_eval(
            engine.model, dataloader, scaler, device, args.output_len, not args.no_amp,
            reference_mode, title=f'Final Test (best epoch {best_epoch})'
        )
        final_row = {
            'reference_mode': reference_mode,
            'best_epoch': best_epoch,
            'test_mae': final_mae,
            'test_rmse': final_rmse,
            'test_mape': final_mape,
            'test_wmape': final_wmape,
        }
        for j in range(len(h_mae)):
            final_row[f'h{j+1}_mae'] = round(h_mae[j], 6)
            final_row[f'h{j+1}_rmse'] = round(h_rmse[j], 6)
            final_row[f'h{j+1}_mape'] = round(h_mape[j], 6)
            final_row[f'h{j+1}_wmape'] = round(h_wmape[j], 6)
        pd.DataFrame([final_row]).round(8).to_csv(
            os.path.join(save_path, 'final_test.csv'), index=False
        )

    pd.DataFrame(result_log).round(8).to_csv(os.path.join(save_path, 'train_log.csv'), index=False)
    if test_log:
        pd.DataFrame(test_log).round(8).to_csv(os.path.join(save_path, 'test_log.csv'), index=False)
    print(f'Logs saved to {save_path}')
    print(f'Best test MAE observed during training: {test_best:.4f}')
