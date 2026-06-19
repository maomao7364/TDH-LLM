"""Training entry point shared by current-model module ablation scripts."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from dataset_checks import validate_dataset_shapes
from experiments.module_ablation_model import MODULE_VARIANTS, ModuleAblationDeSTLLM
from experiments.reference_ablation_training import (
    Trainer,
    load_adjacency,
    load_dataset,
    make_parser,
    mask_references,
    resolve_dataset,
    run_test_eval,
    seed_it,
)


def run_module_experiment(args, module_variant):
    if module_variant not in MODULE_VARIANTS:
        raise ValueError(f'Unknown module_variant: {module_variant}')

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
        dataloader,
        input_len=args.input_len,
        output_len=args.output_len,
        input_dim=args.input_dim,
        num_nodes=args.num_nodes,
        target_dim=3,
    )
    print(f'Module variant: {module_variant}')
    print(f'Data shape: {shape_info}')

    scaler = dataloader['scaler']
    adj = load_adjacency(args, data_dir, dataloader)

    model = ModuleAblationDeSTLLM(
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
        module_variant=module_variant,
        gpt2_path=args.gpt2_path,
    )

    engine = Trainer(model, scaler, args.lrate, args.wdecay, device,
                     epochs=args.epochs, use_amp=not args.no_amp)

    if args.save is None:
        args.save = os.path.join(
            _parent_dir, 'logs', time.strftime('%Y-%m-%d-%H-%M-%S') + '-module-ablation'
        ) + '/'
    save_path = os.path.join(args.save, data_name, module_variant) + '/'
    os.makedirs(save_path, exist_ok=True)

    model_args = vars(args).copy()
    model_args.update(module_variant=module_variant)
    with open(os.path.join(save_path, 'model_args.json'), 'w', encoding='utf-8') as f:
        json.dump(model_args, f, indent=2)

    loss_best = float('inf')
    test_best = float('inf')
    epochs_no_best = 0
    best_epoch = 0
    result_log = []
    test_log = []

    print('Start module ablation training ...', flush=True)
    for epoch in range(1, args.epochs + 1):
        train_losses, train_mapes, train_rmses, train_wmapes = [], [], [], []
        for x, y in dataloader['train_loader']:
            train_x = x.float().to(device, non_blocking=True).transpose(1, 3)
            train_y = y.float().to(device, non_blocking=True).transpose(1, 3)
            train_x, train_fd, train_fw = mask_references(train_x, train_y, 'full_refs')
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
            val_x, val_fd, val_fw = mask_references(val_x, val_y, 'full_refs')
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
                not args.no_amp, 'full_refs', title=f'Epoch {epoch:03d}'
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
            'full_refs', title=f'Final Test (best epoch {best_epoch})'
        )
        final_row = {
            'module_variant': module_variant,
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


__all__ = ['make_parser', 'run_module_experiment']
