#!/usr/bin/env python3
"""
Optuna hyperparameter tuning script for time series models.

Usage:
    python optuna_tune.py --model_type strats --dataset physionet_2012 --n_trials 50
    python optuna_tune.py --model_type dbvt --dataset physionet_2012 --n_trials 100 --gpu 0
    python optuna_tune.py --model_type warpformer --dataset mimic_iii --n_trials 100 --gpu 0
"""

import os
import sys
import argparse
import gc
import traceback
import warnings
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import torch
import numpy as np
import json
import torch.nn as nn
import random
# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from dataset import Dataset
from models import (
    Strats, GRUD_TS, KEDGN, HiPatch, 
    Raindrop, Warpformer, MTM, DBVT
)
from torch.optim import AdamW
from utils import Evaluator
def count_parameters(logger, model: nn.Module):
    """Print no. of parameters in model, no. of traininable parameters,
     no. of parameters in each dtype."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write('\nModel details:')
    logger.write('# parameters: '+str(total))
    logger.write('# trainable parameters: '+str(trainable)+', '\
                 +str(100*trainable/total)+'%')

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: 
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    logger.write('#params by dtype:')
    for k, v in dtypes.items():
        logger.write(str(k)+': '+str(v)+', '+str(100*v/total)+'%')

def set_all_seeds(seed: int) -> None:
    """Function to set seeds for all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count()>0:
        torch.cuda.manual_seed_all(seed)

def _is_oom_error(err: BaseException) -> bool:
    """Check if error is OOM."""
    if isinstance(err, torch.OutOfMemoryError):
        return True
    if isinstance(err, RuntimeError) and 'out of memory' in str(err).lower():
        return True
    return False


def _cleanup_cuda() -> None:
    """Clean up CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def suggest_parameters(trial, model_type, dataset):
    """
    Suggest hyperparameters for the given model type.
    
    Args:
        trial: Optuna trial object
        model_type: Model type string
        dataset: Dataset name
        
    Returns:
        Dictionary of suggested parameters
    """
    params = {}
    
    # Common parameters for most models
    if model_type in ['strats', 'istrats']:
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 128, 256])
        params['num_layers'] = trial.suggest_int('num_layers', 1, 4)
        params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
        params['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
        params['attention_dropout'] = trial.suggest_float('attention_dropout', 0.1, 0.3)
        params['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])
        
    elif model_type == 'warpformer':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 128, 256])
        params['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        params['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4, 8])  # 添加1头(官方默认)
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.3)  # 官方默认0.0
        params['lr'] = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])
        
        # WarpFormer specific parameters (官方: warpfunc=l2, warpact=relu, nonneg_trans=sigmoid)
        params['warp_num'] = [0, trial.suggest_int('warp_num_max', 6, 24)]
        params['warpfunc'] = trial.suggest_categorical('warpfunc', ['l1', 'l2', 'l3'])
        params['warpact'] = trial.suggest_categorical('warpact', ['relu', 'sigmoid'])
        params['nonneg_trans'] = trial.suggest_categorical('nonneg_trans', ['abs', 'sigmoid', 'softplus'])  # 添加sigmoid
        params['warp_full_attn'] = trial.suggest_categorical('warp_full_attn', [True, False])
        
    elif model_type == 'kedgn':
        # 根据官方脚本参数范围调整
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [8, 12, 16])
        params['query_vector_dim'] = trial.suggest_int('query_vector_dim', 5, 9)
        params['node_emb_dim'] = trial.suggest_categorical('node_emb_dim', [7, 9, 12, 16])
        params['node_enc_layer'] = trial.suggest_int('node_enc_layer', 1, 3)
        params['rarity_alpha'] = trial.suggest_float('rarity_alpha', 0.5, 2.5)  # Official range: 0.8~2.0
        params['lr'] = trial.suggest_categorical('lr', [0.001, 0.005])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [256, 512])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0])
        
    elif model_type == 'hipatch':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [16, 32, 64])
        params['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4])
        params['num_layers'] = trial.suggest_int('num_layers', 1, 2)  # 减少层数避免显存问题
        patch_max = 12 if dataset == 'physionet_2012' else (8 if dataset == 'physionet_2019' else 6)
        params['patch_size'] = trial.suggest_int('patch_size', 6, patch_max)  # 最小patch_size增大
        params['alpha'] = trial.suggest_float('alpha', 0.7, 1.0)
        params['res'] = trial.suggest_categorical('res', [0, 1])
        params['lr'] = trial.suggest_float('lr', 1e-4, 5e-4, log=True)
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64])  # 增大batch_size加速！
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])
        
    elif model_type == 'raindrop':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [64, 96, 128, 256])
        params['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        params['num_heads'] = trial.suggest_categorical('num_heads', [4, 8, 16])
        params['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)
        params['lr'] = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])
        
    elif model_type == 'grud':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 128])
        params['dropout'] = trial.suggest_float('dropout', 0.1, 0.3)
        params['lr'] = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])
    
    elif model_type == 'mtm':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [64, 96, 128])
        params['r_hid'] = trial.suggest_categorical('r_hid', [2, 4])
        params['dropout'] = trial.suggest_float('dropout', 0.05, 0.2)
        params['down_mode'] = trial.suggest_categorical('down_mode', ['concat', 'max', 'avg'])
        # Ratios - suggest number of layers and ratio per layer
        num_layers = trial.suggest_int('num_mtm_layers', 2, 4)
        ratio_val = trial.suggest_int('downsample_ratio', 2, 4)
        params['ratios'] = [ratio_val] * num_layers
        params['lr'] = trial.suggest_float('lr', 1e-4, 5e-4, log=True)
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])
        
    elif model_type == 'dbvt':
        # GRU branch
        params['gru_hid_dim'] = trial.suggest_categorical('gru_hid_dim', [64, 128, 256])
        params['gru_dropout'] = trial.suggest_categorical('gru_dropout', [0.1, 0.2, 0.3, 0.4])
        # Transformer branch
        params['transformer_hid_dim'] = trial.suggest_categorical('transformer_hid_dim', [32, 64, 128])
        params['transformer_num_layers'] = trial.suggest_int('transformer_num_layers', 1, 4)
        params['transformer_num_heads'] = trial.suggest_categorical('transformer_num_heads', [2, 4, 8])
        params['transformer_dropout'] = trial.suggest_categorical('transformer_dropout', [0.1, 0.15, 0.2, 0.3])
        # Fusion layer
        params['fusion_hid_dim'] = trial.suggest_categorical('fusion_hid_dim', [128, 256])
        params['fusion_dropout'] = trial.suggest_categorical('fusion_dropout', [0.1, 0.2, 0.3])
        # Training
        params['lr'] = trial.suggest_float('lr', 5e-5, 5e-4, log=True)
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0])
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return params


def train_and_evaluate(args, trial: optuna.Trial = None) -> float:
    """
    Train model and return validation metric AUPRC.
    
    Args:
        args: Namespace with all configuration
        trial: Optuna trial for pruning (optional)
    
    Returns:
        Combined metric auprc on validation set
    """
    set_all_seeds(args.seed)
    
    # Warpformer uses cumsum which doesn't support deterministic mode
    if args.model_type == 'warpformer':
        torch.use_deterministic_algorithms(False)
    
    # Load data
    dataset = Dataset(args)
    
    # Load model
    model_class = {
        'strats': Strats, 'istrats': Strats,
        'kedgn': KEDGN, 'grud': GRUD_TS,
        'hipatch': HiPatch, 'raindrop': Raindrop,
        'warpformer': Warpformer, 'mtm': MTM, 'dbvt': DBVT
    }
    model = model_class[args.model_type](args)
    model.to(args.device)
    
    # Training setup
    num_train = len(dataset.splits['train'])
    num_batches_per_epoch = int(np.ceil(num_train / args.train_batch_size))
    args.logger.write(f'No. of training batches per epoch = {num_batches_per_epoch}')
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    evaluator = Evaluator(args)
    
    wait = args.patience
    best_val_metric = 0
    
    model.train()
    for epoch in range(1, args.max_epochs + 1):
        args.logger.write(f'>>> Epoch {epoch}/{args.max_epochs}')
        
        # Train for one epoch
        total_loss = 0
        for batch_idx in range(num_batches_per_epoch):
            batch = dataset.get_batch()
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            loss = model(**batch)
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches_per_epoch
        args.logger.write(f'Epoch {epoch} completed | Avg Loss: {avg_loss:.4f}')
        
        # Validation
        val_res = evaluator.evaluate(model, dataset, 'val', epoch=epoch)
        model.train()
        
        curr_val_metric = val_res['auprc']  # Use AUPRC as metric (consistent with main.py)
        
        if curr_val_metric > best_val_metric:
            best_val_metric = curr_val_metric
            wait = args.patience
        else:
            wait -= 1
            if wait == 0:
                args.logger.write('Early stopping triggered')
                break
        
        # Optuna pruning
        if trial is not None:
            trial.report(curr_val_metric, epoch)
            if epoch > int(args.max_epochs * 0.3) and trial.should_prune():
                raise optuna.TrialPruned()
    
    return best_val_metric


class SilentLogger:
    """A silent logger that only prints to console (no file output)."""
    def write(self, msg):
        pass  # Silent - no output
    def close(self):
        pass


def objective(trial, base_args):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        base_args: Base arguments namespace
        
    Returns:
        Validation AUPRC score
    """
    # Create a copy of args
    args = argparse.Namespace(**vars(base_args))
    
    # Suggest hyperparameters
    suggested_params = suggest_parameters(trial, args.model_type, args.dataset)
    
    # Update args with suggested parameters
    for key, value in suggested_params.items():
        setattr(args, key, value)
    
    # No output directory needed - use silent logger
    args.output_dir = None
    args.logger = SilentLogger()
    
    try:
        # Train and evaluate
        score = train_and_evaluate(args, trial)
        return score
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        if _is_oom_error(e):
            _cleanup_cuda()
            raise optuna.TrialPruned()
        raise
    finally:
        _cleanup_cuda()


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning')
    
    # Required arguments
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['strats', 'istrats', 'kedgn', 'hipatch', 'grud', 
                               'raindrop', 'warpformer', 'mtm', 'dbvt'],
                       help='Model type to tune')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['physionet_2012', 'physionet_2019', 'mimic_iii'],
                       help='Dataset to use')
    
    # Optuna arguments
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials to run')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Study name (default: {model_type}_{dataset})')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for study storage (e.g., sqlite:///optuna.db)')
    
    # Training arguments
    parser.add_argument('--gpu', type=int, default=2,
                       help='GPU device ID')
    parser.add_argument('--max_epochs', type=int, default=20,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=2026,
                       help='Random seed')
    parser.add_argument('--fold', type=int, default=1,
                       help='Fold number for K-fold CV (default: 1)')
    parser.add_argument('--split_file', type=str, default=None,
                       help='Split file name (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Set study name
    if args.study_name is None:
        args.study_name = f'{args.model_type}_{args.dataset}'
    
    # Set device
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    # Set root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.root_dir = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Set split file automatically if not specified
    if args.split_file is None:
        args.split_file = f'{args.dataset}_5fold.json'
    
    # Fixed parameters
    args.eval_batch_size = 64
    args.validate_every = None
    args.max_obs = 880
    args.max_timesteps = 880
    
    print("=" * 80)
    print(f"Optuna Hyperparameter Tuning")
    print("=" * 80)
    print(f"Model: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()
    
    # Set default storage if not provided
    if args.storage is None:
        optuna_db_dir = os.path.join(args.root_dir, 'optuna_studies')
        os.makedirs(optuna_db_dir, exist_ok=True)
        args.storage = f'sqlite:///{optuna_db_dir}/{args.study_name}.db'
    
    print(f"Storage: {args.storage}")
    print("=" * 80)
    print()
    
    # Create study
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction='maximize',
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print()
    print("=" * 80)
    print("Optimization Results")
    print("=" * 80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (AUROC + AUPRC): {study.best_value:.4f}")
    print()
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Save results
    results_dir = os.path.join(args.root_dir, 'optuna_results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'{args.study_name}_best_params.txt')
    with open(results_file, 'w') as f:
        f.write(f"Study: {args.study_name}\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_value:.4f}\n")
        f.write("\nBest hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nResults saved to: {results_file}")
    
if __name__ == '__main__':
    main()
