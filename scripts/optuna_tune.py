#!/usr/bin/env python3
"""Optuna hyperparameter tuning driven by configs/*.yaml."""

import os
import sys
import argparse
import concurrent.futures
import gc
import traceback
import warnings
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import numpy as np
import json
import torch.nn as nn
import random
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)

from dataset import Dataset
from models import ModelWrapper
from torch.optim import AdamW
from utils import Evaluator
from utils.config_loader import load_model_config

SUPPORTED_MODEL_TYPES = [
    'dbvt',
    'cde', 'cde++',
    'gru', 'gru+', 'gru++',
    'grud', 'grud+', 'grud++',
    'hipatch', 'hipatch+', 'hipatch++',
    'kedgn', 'kedgn+', 'kedgn++',
    'mtm', 'mtm+', 'mtm++',
    'raindrop', 'raindrop+', 'raindrop++',
    'sand',
    'strats', 'strats+', 'strats++',
    'tcn',
    'transformer', 'transformer+', 'transformer++',
    'warpformer', 'warpformer+', 'warpformer++',
]


def resolve_model_type(model_type: str) -> tuple[str, bool, str]:
    """Return runtime backbone type, branch2 flag, and exact config variant name."""
    if model_type.lower() == 'dbvt':
        return 'transformer', True, 'transformer+'
    if model_type.endswith('++'):
        return model_type[:-2], True, model_type
    if model_type.endswith('+'):
        return model_type[:-1], True, model_type
    return model_type, False, model_type


DUAL_BRANCH_LAMBDA_AUX_CHOICES = [0.05, 0.1, 0.2, 0.3, 0.5]


def add_dual_branch_params(
    trial,
    params,
    *,
    gru_hid_choices,
    gru_dropout_choices,
    fusion_hid_choices,
    fusion_dropout_choices,
):
    """Shared tuning knobs for top-level dual-branch gated-sum models."""
    params['gru_hid_dim'] = trial.suggest_categorical('gru_hid_dim', gru_hid_choices)
    params['gru_dropout'] = trial.suggest_categorical('gru_dropout', gru_dropout_choices)
    params['fusion_hid_dim'] = trial.suggest_categorical('fusion_hid_dim', fusion_hid_choices)
    params['fusion_dropout'] = trial.suggest_categorical('fusion_dropout', fusion_dropout_choices)
    params['lambda_aux'] = trial.suggest_categorical('lambda_aux', DUAL_BRANCH_LAMBDA_AUX_CHOICES)


def parse_folds(folds_arg: str, default_fold: int) -> list[int]:
    if folds_arg is None:
        return [default_fold]

    folds_arg = folds_arg.strip().lower()
    if folds_arg == 'fast':
        return [1, 2]
    if folds_arg == 'all':
        return [1, 2, 3, 4, 5]

    folds = []
    for token in folds_arg.split(','):
        token = token.strip()
        if not token:
            continue
        fold = int(token)
        if fold < 1 or fold > 5:
            raise ValueError(f'Fold must be in [1, 5], got: {fold}')
        folds.append(fold)

    if not folds:
        raise ValueError('No valid folds were provided.')

    return sorted(set(folds))


def _validate_common_params(model_type, params):
    if 'hid_dim' in params and 'num_heads' in params and params['hid_dim'] % params['num_heads'] != 0:
        return False
    if (
        'transformer_hid_dim' in params
        and 'transformer_num_heads' in params
        and params['transformer_hid_dim'] % params['transformer_num_heads'] != 0
    ):
        return False
    if 'patch_size' in params and params['patch_size'] <= 0:
        return False
    return True


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


def parse_gpu_ids(gpus_arg: str | None, fallback_gpu: int) -> list[int]:
    if gpus_arg is None or not gpus_arg.strip():
        return [fallback_gpu]

    gpu_ids = []
    for token in gpus_arg.split(','):
        token = token.strip()
        if not token:
            continue
        gpu_ids.append(int(token))

    if not gpu_ids:
        raise ValueError('No valid GPU ids were provided.')
    return gpu_ids


def parse_cpu_sets(cpu_sets_arg: str | None) -> list[str] | None:
    if cpu_sets_arg is None or not cpu_sets_arg.strip():
        return None

    cpu_sets = []
    for token in cpu_sets_arg.split(';'):
        token = token.strip()
        if token:
            cpu_sets.append(token)

    if not cpu_sets:
        raise ValueError('No valid CPU sets were provided.')
    return cpu_sets


def is_nan_metric_error(err: BaseException) -> bool:
    return isinstance(err, ValueError) and 'Input contains NaN' in str(err)


def _apply_cpu_affinity(cpu_set: str | None) -> None:
    if not cpu_set or not hasattr(os, 'sched_setaffinity'):
        return

    allowed_cpus: set[int] = set()
    for chunk in cpu_set.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start_str, end_str = chunk.split('-', 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f'Invalid CPU range: {chunk}')
            allowed_cpus.update(range(start, end + 1))
        else:
            allowed_cpus.add(int(chunk))

    if not allowed_cpus:
        raise ValueError(f'No valid CPUs parsed from cpu_set={cpu_set!r}')
    os.sched_setaffinity(0, allowed_cpus)


def _run_parallel_fold_task(task: dict[str, object]) -> tuple[int, float]:
    cpu_set = task.get('cpu_set')
    if isinstance(cpu_set, str):
        _apply_cpu_affinity(cpu_set)

    args_data = dict(task['args_data'])
    args_data['fold'] = int(task['fold'])
    gpu_id = int(task['gpu_id'])
    args_data['device'] = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    args_data['output_dir'] = None
    args_data['logger'] = SilentLogger()
    args = argparse.Namespace(**args_data)
    try:
        score = train_and_evaluate(args, None)
        return int(task['fold']), float(score)
    finally:
        _cleanup_cuda()


def run_parallel_cv_objective(trial, args):
    folds = getattr(args, 'cv_folds', [args.fold])
    max_workers = min(len(folds), len(args.gpu_ids))
    if max_workers <= 0:
        raise ValueError('Parallel CV requires at least one GPU id.')

    args_data = {k: v for k, v in vars(args).items() if k not in {'logger', 'output_dir'}}
    task_queue = []
    for idx, fold in enumerate(folds):
        task_queue.append({
            'fold': fold,
            'gpu_id': args.gpu_ids[idx % len(args.gpu_ids)],
            'cpu_set': None if args.cpu_sets is None else args.cpu_sets[idx % len(args.cpu_sets)],
            'args_data': args_data,
        })

    scores_by_fold = {}
    ctx = torch.multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_to_task = {executor.submit(_run_parallel_fold_task, task): task for task in task_queue}
        for completed_steps, future in enumerate(concurrent.futures.as_completed(future_to_task), start=1):
            fold, score = future.result()
            scores_by_fold[fold] = score
            ordered_scores = [scores_by_fold[curr_fold] for curr_fold in sorted(scores_by_fold)]
            trial.report(float(np.mean(ordered_scores)), completed_steps)

    ordered_final_scores = [scores_by_fold[fold] for fold in sorted(scores_by_fold)]
    return float(np.mean(ordered_final_scores))


def suggest_parameters(trial, config_model_type, dataset):
    params = {}

    if config_model_type == 'gru':
        params['gru_hid_dim'] = trial.suggest_categorical('gru_hid_dim', [64, 128, 256])
        params['gru_dropout'] = trial.suggest_categorical('gru_dropout', [0.1, 0.2, 0.3, 0.4])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    elif config_model_type in ('gru+', 'gru++'):
        params['gru_hid_dim'] = trial.suggest_categorical('gru_hid_dim', [64, 128, 256])
        params['gru_dropout'] = trial.suggest_categorical('gru_dropout', [0.1, 0.2, 0.3, 0.4])
        params['transformer_hid_dim'] = trial.suggest_categorical('transformer_hid_dim', [32, 64, 128])
        params['transformer_num_layers'] = trial.suggest_categorical('transformer_num_layers', [2, 3, 4, 6])
        params['transformer_num_heads'] = trial.suggest_categorical('transformer_num_heads', [2, 4, 8])
        params['transformer_dropout'] = trial.suggest_categorical('transformer_dropout', [0.05, 0.1, 0.15, 0.2, 0.3])
        params['fusion_hid_dim'] = trial.suggest_categorical('fusion_hid_dim', [128, 200, 256, 384])
        params['fusion_dropout'] = trial.suggest_categorical('fusion_dropout', [0.1, 0.15, 0.2, 0.3, 0.4])
        params['lambda_aux'] = trial.suggest_categorical('lambda_aux', DUAL_BRANCH_LAMBDA_AUX_CHOICES)
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    elif config_model_type == 'grud':
        if dataset == 'physionet_2012':
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [128, 256])
            params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2])
            params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0])
        elif dataset == 'physionet_2019':
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [128, 256])
            params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2])
            params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0])
        else:
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [64, 128])
            params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3])
            params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [8, 16, 32])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0])

    elif config_model_type in ('grud+', 'grud++'):
        if dataset == 'physionet_2012':
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [128, 256])
            params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2])
            params['transformer_hid_dim'] = trial.suggest_categorical('transformer_hid_dim', [32, 64])
            params['transformer_num_layers'] = trial.suggest_categorical('transformer_num_layers', [2, 4])
            params['transformer_num_heads'] = trial.suggest_categorical('transformer_num_heads', [2, 4, 8])
            params['transformer_dropout'] = trial.suggest_categorical('transformer_dropout', [0.1, 0.15, 0.2])
            params['fusion_hid_dim'] = trial.suggest_categorical('fusion_hid_dim', [128, 256])
            params['fusion_dropout'] = trial.suggest_categorical('fusion_dropout', [0.1, 0.2, 0.3])
            params['lambda_aux'] = trial.suggest_categorical('lambda_aux', [0.1, 0.2, 0.3])
            params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [8, 16, 32])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0])
        elif dataset == 'physionet_2019':
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [128, 256])
            params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2])
            params['transformer_hid_dim'] = trial.suggest_categorical('transformer_hid_dim', [32, 64])
            params['transformer_num_layers'] = trial.suggest_categorical('transformer_num_layers', [2, 4])
            params['transformer_num_heads'] = trial.suggest_categorical('transformer_num_heads', [2, 4])
            params['transformer_dropout'] = trial.suggest_categorical('transformer_dropout', [0.1, 0.15, 0.2])
            params['fusion_hid_dim'] = trial.suggest_categorical('fusion_hid_dim', [128, 256])
            params['fusion_dropout'] = trial.suggest_categorical('fusion_dropout', [0.1, 0.2, 0.3])
            params['lambda_aux'] = trial.suggest_categorical('lambda_aux', [0.1, 0.2, 0.3])
            params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [8, 16, 32])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0])
        else:
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [64, 128])
            params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3])
            params['transformer_hid_dim'] = trial.suggest_categorical('transformer_hid_dim', [32, 64])
            params['transformer_num_layers'] = trial.suggest_categorical('transformer_num_layers', [4, 6])
            params['transformer_num_heads'] = trial.suggest_categorical('transformer_num_heads', [2, 4])
            params['transformer_dropout'] = trial.suggest_categorical('transformer_dropout', [0.05, 0.1, 0.15, 0.2])
            params['fusion_hid_dim'] = trial.suggest_categorical('fusion_hid_dim', [128, 256])
            params['fusion_dropout'] = trial.suggest_categorical('fusion_dropout', [0.1, 0.2, 0.3])
            params['lambda_aux'] = trial.suggest_categorical('lambda_aux', [0.1, 0.2, 0.3, 0.5])
            params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [4, 8, 16])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [1.0, 3.0, 5.0])

    elif config_model_type == 'strats':
        if dataset == 'physionet_2012':
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64])
            params['num_layers'] = trial.suggest_categorical('num_layers', [2, 3, 4])
            params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
            params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.15, 0.2, 0.3])
            params['attention_dropout'] = trial.suggest_categorical('attention_dropout', [0.05, 0.1, 0.15])
            params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [8, 16, 32])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0])
        elif dataset == 'physionet_2019':
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64])
            params['num_layers'] = trial.suggest_categorical('num_layers', [2, 3, 4])
            params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4])
            params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.15, 0.2, 0.3])
            params['attention_dropout'] = trial.suggest_categorical('attention_dropout', [0.05, 0.1, 0.15])
            params['lr'] = trial.suggest_categorical('lr', [2e-4, 3e-4, 5e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [8, 16, 32])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])
        else:
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64])
            params['num_layers'] = trial.suggest_categorical('num_layers', [4, 6])
            params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4])
            params['dropout'] = trial.suggest_categorical('dropout', [0.05, 0.1, 0.15, 0.2])
            params['attention_dropout'] = trial.suggest_categorical('attention_dropout', [0.05, 0.1, 0.15])
            params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [4, 8, 16])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [1.0, 3.0, 5.0])

    elif config_model_type in ('strats+', 'strats++'):
        if dataset == 'physionet_2012':
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64])
            params['num_layers'] = trial.suggest_categorical('num_layers', [2, 3, 4])
            params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
            params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.15, 0.2, 0.3])
            params['attention_dropout'] = trial.suggest_categorical('attention_dropout', [0.05, 0.1, 0.15])
            params['gru_hid_dim'] = trial.suggest_categorical('gru_hid_dim', [64, 128])
            params['gru_dropout'] = trial.suggest_categorical('gru_dropout', [0.1, 0.2, 0.3])
            params['fusion_hid_dim'] = trial.suggest_categorical('fusion_hid_dim', [128, 256])
            params['fusion_dropout'] = trial.suggest_categorical('fusion_dropout', [0.1, 0.2, 0.3])
            params['lambda_aux'] = trial.suggest_categorical('lambda_aux', [0.1, 0.2, 0.3])
            params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [8, 16, 32])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0])
        elif dataset == 'physionet_2019':
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64])
            params['num_layers'] = trial.suggest_categorical('num_layers', [2, 3, 4])
            params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4])
            params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.15, 0.2, 0.3])
            params['attention_dropout'] = trial.suggest_categorical('attention_dropout', [0.05, 0.1, 0.15])
            params['gru_hid_dim'] = trial.suggest_categorical('gru_hid_dim', [64, 128, 256])
            params['gru_dropout'] = trial.suggest_categorical('gru_dropout', [0.1, 0.2, 0.3])
            params['fusion_hid_dim'] = trial.suggest_categorical('fusion_hid_dim', [128, 256])
            params['fusion_dropout'] = trial.suggest_categorical('fusion_dropout', [0.1, 0.2, 0.3])
            params['lambda_aux'] = trial.suggest_categorical('lambda_aux', [0.1, 0.2, 0.3])
            params['lr'] = trial.suggest_categorical('lr', [2e-4, 3e-4, 5e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [8, 16, 32])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])
        else:
            params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64])
            params['num_layers'] = trial.suggest_categorical('num_layers', [4, 6])
            params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4])
            params['dropout'] = trial.suggest_categorical('dropout', [0.05, 0.1, 0.15, 0.2])
            params['attention_dropout'] = trial.suggest_categorical('attention_dropout', [0.05, 0.1, 0.15])
            params['gru_hid_dim'] = trial.suggest_categorical('gru_hid_dim', [64, 128])
            params['gru_dropout'] = trial.suggest_categorical('gru_dropout', [0.2, 0.3])
            params['fusion_hid_dim'] = trial.suggest_categorical('fusion_hid_dim', [128, 256])
            params['fusion_dropout'] = trial.suggest_categorical('fusion_dropout', [0.1, 0.2, 0.3])
            params['lambda_aux'] = trial.suggest_categorical('lambda_aux', [0.1, 0.2, 0.3, 0.5])
            params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4])
            params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [4, 8, 16])
            params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [1.0, 3.0, 5.0])

    elif config_model_type == 'transformer':
        params['transformer_hid_dim'] = trial.suggest_categorical('transformer_hid_dim', [32, 64, 128])
        params['transformer_num_layers'] = trial.suggest_categorical('transformer_num_layers', [2, 3, 4, 6])
        params['transformer_num_heads'] = trial.suggest_categorical('transformer_num_heads', [2, 4, 8])
        params['transformer_dropout'] = trial.suggest_categorical('transformer_dropout', [0.05, 0.1, 0.15, 0.2, 0.3])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    elif config_model_type in ('transformer+', 'transformer++'):
        params['gru_hid_dim'] = trial.suggest_categorical('gru_hid_dim', [64, 128])
        params['gru_dropout'] = trial.suggest_categorical('gru_dropout', [0.1, 0.2, 0.3])
        params['transformer_hid_dim'] = trial.suggest_categorical('transformer_hid_dim', [32, 64, 128])
        params['transformer_num_layers'] = trial.suggest_categorical('transformer_num_layers', [2, 4, 6])
        params['transformer_num_heads'] = trial.suggest_categorical('transformer_num_heads', [2, 4, 8])
        params['transformer_dropout'] = trial.suggest_categorical('transformer_dropout', [0.05, 0.1, 0.15, 0.2, 0.3])
        params['fusion_hid_dim'] = trial.suggest_categorical('fusion_hid_dim', [128, 256])
        params['fusion_dropout'] = trial.suggest_categorical('fusion_dropout', [0.1, 0.2, 0.3])
        params['lambda_aux'] = trial.suggest_categorical('lambda_aux', DUAL_BRANCH_LAMBDA_AUX_CHOICES)
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    elif config_model_type == 'mtm':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [64, 96, 128])
        params['r_hid'] = trial.suggest_categorical('r_hid', [2, 4, 8])
        params['dropout'] = trial.suggest_categorical('dropout', [0.05, 0.1, 0.15, 0.2, 0.3])
        params['down_mode'] = trial.suggest_categorical('down_mode', ['concat', 'max', 'avg'])
        params['num_mtm_layers'] = trial.suggest_categorical('num_mtm_layers', [2, 3, 4])
        params['downsample_ratio'] = trial.suggest_categorical('downsample_ratio', [2, 3, 4])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 4e-4, 7e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    elif config_model_type in ('mtm+', 'mtm++'):
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [64, 96, 128])
        params['r_hid'] = trial.suggest_categorical('r_hid', [2, 4, 8])
        params['dropout'] = trial.suggest_categorical('dropout', [0.05, 0.1, 0.15, 0.2, 0.3])
        params['down_mode'] = trial.suggest_categorical('down_mode', ['concat', 'max', 'avg'])
        params['num_mtm_layers'] = trial.suggest_categorical('num_mtm_layers', [2, 3, 4])
        params['downsample_ratio'] = trial.suggest_categorical('downsample_ratio', [2, 3, 4])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 4e-4, 7e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

        add_dual_branch_params(
            trial,
            params,
            gru_hid_choices=[32, 64, 128],
            gru_dropout_choices=[0.1, 0.2, 0.3],
            fusion_hid_choices=[128, 256],
            fusion_dropout_choices=[0.1, 0.2, 0.3],
        )

    elif config_model_type == 'warpformer':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 128])
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])
        params['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        params['dropout'] = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])
        params['warp_num_max'] = trial.suggest_categorical('warp_num_max', [4, 7, 10, 13, 16, 18, 20, 24])
        params['warpfunc'] = trial.suggest_categorical('warpfunc', ['l1', 'l2', 'l3'])
        params['warpact'] = trial.suggest_categorical('warpact', ['relu', 'sigmoid'])
        params['nonneg_trans'] = trial.suggest_categorical('nonneg_trans', ['abs', 'sigmoid', 'softplus'])
        params['warp_full_attn'] = trial.suggest_categorical('warp_full_attn', [True, False])

    elif config_model_type in ('warpformer+', 'warpformer++'):
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 128])
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])
        params['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        params['dropout'] = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])
        params['warp_num_max'] = trial.suggest_categorical('warp_num_max', [4, 7, 10, 16, 24])
        params['warpfunc'] = trial.suggest_categorical('warpfunc', ['l1', 'l2', 'l3'])
        params['warpact'] = trial.suggest_categorical('warpact', ['relu', 'sigmoid'])
        params['nonneg_trans'] = trial.suggest_categorical('nonneg_trans', ['abs', 'sigmoid', 'softplus'])
        params['warp_full_attn'] = trial.suggest_categorical('warp_full_attn', [True, False])

        add_dual_branch_params(
            trial,
            params,
            gru_hid_choices=[32, 64, 128],
            gru_dropout_choices=[0.1, 0.2, 0.3],
            fusion_hid_choices=[128, 256],
            fusion_dropout_choices=[0.1, 0.2, 0.3],
        )

    elif config_model_type == 'hipatch':
        if dataset == 'mimic_iii':
            patch_choices = [4, 5, 6, 7, 8]
        elif dataset == 'physionet_2019':
            patch_choices = [6, 7, 8, 9, 10]
        else:
            patch_choices = [5, 6, 7, 8, 9]

        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 96])
        params['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4])
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])
        params['patch_size'] = trial.suggest_categorical('patch_size', patch_choices)
        params['alpha'] = trial.suggest_categorical('alpha', [0.6, 0.7, 0.8, 0.9, 1.0])
        params['res'] = trial.suggest_categorical('res', [0, 1])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 5e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    elif config_model_type in ('hipatch+', 'hipatch++'):
        if dataset == 'mimic_iii':
            patch_choices = [4, 5, 6, 7, 8]
        elif dataset == 'physionet_2019':
            patch_choices = [6, 7, 8, 9, 10]
        else:
            patch_choices = [5, 6, 7, 8, 9]

        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 96])
        params['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4])
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])
        params['patch_size'] = trial.suggest_categorical('patch_size', patch_choices)
        params['alpha'] = trial.suggest_categorical('alpha', [0.6, 0.7, 0.8, 0.9, 1.0])
        params['res'] = trial.suggest_categorical('res', [0, 1])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 5e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

        add_dual_branch_params(
            trial,
            params,
            gru_hid_choices=[32, 64, 128],
            gru_dropout_choices=[0.1, 0.2, 0.3],
            fusion_hid_choices=[128, 256],
            fusion_dropout_choices=[0.1, 0.2, 0.3],
        )

    elif config_model_type == 'cde':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 96, 128])
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])
        params['vf_hid_dim'] = trial.suggest_categorical('vf_hid_dim', [64, 128, 192, 256])
        params['dropout'] = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 5e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    elif config_model_type == 'cde++':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 96, 128])
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])
        params['vf_hid_dim'] = trial.suggest_categorical('vf_hid_dim', [64, 128, 192, 256])
        params['dropout'] = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 5e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

        add_dual_branch_params(
            trial,
            params,
            gru_hid_choices=[32, 64, 128],
            gru_dropout_choices=[0.1, 0.2, 0.3, 0.4],
            fusion_hid_choices=[128, 200, 256, 384],
            fusion_dropout_choices=[0.1, 0.2, 0.3, 0.4],
        )

    elif config_model_type == 'kedgn':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [8, 12, 16, 24])
        params['query_vector_dim'] = trial.suggest_categorical('query_vector_dim', [5, 7, 9, 12])
        params['node_emb_dim'] = trial.suggest_categorical('node_emb_dim', [7, 9, 12, 16])
        params['node_enc_layer'] = trial.suggest_categorical('node_enc_layer', [1, 2, 3])
        params['rarity_alpha'] = trial.suggest_categorical('rarity_alpha', [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        params['lr'] = trial.suggest_categorical('lr', [5e-4, 1e-3, 2e-3, 5e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [128, 256, 512])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    elif config_model_type in ('kedgn+', 'kedgn++'):
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [8, 12, 16, 24])
        params['query_vector_dim'] = trial.suggest_categorical('query_vector_dim', [5, 7, 9, 12])
        params['node_emb_dim'] = trial.suggest_categorical('node_emb_dim', [7, 9, 12, 16])
        params['node_enc_layer'] = trial.suggest_categorical('node_enc_layer', [1, 2, 3])
        params['rarity_alpha'] = trial.suggest_categorical('rarity_alpha', [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        params['lr'] = trial.suggest_categorical('lr', [5e-4, 1e-3, 2e-3, 5e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [128, 256, 512])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

        add_dual_branch_params(
            trial,
            params,
            gru_hid_choices=[16, 32, 64],
            gru_dropout_choices=[0.1, 0.2, 0.3],
            fusion_hid_choices=[128, 256],
            fusion_dropout_choices=[0.1, 0.2, 0.3],
        )

    elif config_model_type == 'raindrop':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [64, 128, 256])
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])
        params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
        params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.18, 0.2, 0.3, 0.4])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 3e-4, 5e-4, 8e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    elif config_model_type in ('raindrop+', 'raindrop++'):
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [64, 128, 256])
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])
        params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
        params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.18, 0.2, 0.3, 0.4])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 3e-4, 5e-4, 8e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

        add_dual_branch_params(
            trial,
            params,
            gru_hid_choices=[64, 128, 256],
            gru_dropout_choices=[0.1, 0.2, 0.3],
            fusion_hid_choices=[128, 256],
            fusion_dropout_choices=[0.1, 0.2, 0.3],
        )

    elif config_model_type == 'tcn':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 128, 256])
        params['num_layers'] = trial.suggest_categorical('num_layers', [2, 3, 4, 5])
        params['kernel_size'] = trial.suggest_categorical('kernel_size', [2, 3, 5, 7])
        params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 5e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64, 128])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    elif config_model_type == 'sand':
        params['hid_dim'] = trial.suggest_categorical('hid_dim', [32, 64, 128])
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])
        params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
        params['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3])
        params['r'] = trial.suggest_categorical('r', [5, 10, 15])
        params['M'] = trial.suggest_categorical('M', [8, 16, 32])
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 5e-4, 1e-3])
        params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [16, 32, 64])
        params['grad_clip_norm'] = trial.suggest_categorical('grad_clip_norm', [0.3, 1.0, 3.0, 5.0])

    else:
        raise ValueError(f'Unknown model type: {config_model_type}')

    if not _validate_common_params(config_model_type, params):
        raise optuna.TrialPruned()

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
    set_all_seeds(args.seed + args.fold)
    
    # Load data
    dataset = Dataset(args)
    
    # Load model
    model = ModelWrapper(args)
    model.to(args.device)
    
    # Training setup
    num_train = len(dataset.splits['train'])
    num_batches_per_epoch = int(np.ceil(num_train / args.train_batch_size))
    args.logger.write(f'No. of training batches per epoch = {num_batches_per_epoch}')
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
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
    suggested_params = suggest_parameters(trial, args.config_model_type, args.dataset)
    
    # Update args with suggested parameters
    for key, value in suggested_params.items():
        setattr(args, key, value)

    # No output directory needed - use silent logger
    args.output_dir = None
    args.logger = SilentLogger()
    
    try:
        folds = getattr(args, 'cv_folds', [args.fold])

        if len(folds) == 1:
            args.fold = folds[0]
            return train_and_evaluate(args, trial)

        if len(args.gpu_ids) > 1:
            return run_parallel_cv_objective(trial, args)

        scores = []
        for step, fold in enumerate(folds, start=1):
            fold_args = argparse.Namespace(**vars(args))
            fold_args.fold = fold
            score = train_and_evaluate(fold_args, None)
            scores.append(score)
            mean_score = float(np.mean(scores))
            trial.report(mean_score, step)
            if step >= 2 and trial.should_prune():
                raise optuna.TrialPruned()
            _cleanup_cuda()

        return float(np.mean(scores))
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        if _is_oom_error(e) or is_nan_metric_error(e):
            _cleanup_cuda()
            raise optuna.TrialPruned()
        raise
    finally:
        _cleanup_cuda()


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning')
    
    # Required arguments
    parser.add_argument('--model_type', type=str, required=True,
                       choices=SUPPORTED_MODEL_TYPES,
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
    parser.add_argument('--gpu', type=int, default=0,
                       help='Fallback single GPU device ID')
    parser.add_argument('--gpus', type=str, default=None,
                       help='Comma-separated GPU ids used for parallel fold CV inside one trial')
    parser.add_argument('--cpu_sets', type=str, default=None,
                       help='Optional semicolon-separated CPU affinity sets matched to --gpus')
    parser.add_argument('--max_epochs', type=int, default=20,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=2026,
                       help='Random seed')
    parser.add_argument('--fold', type=int, default=1,
                       help='Fold number for K-fold CV (default: 1)')
    parser.add_argument('--folds', type=str, default=None,
                       help='Comma-separated folds to tune on, or "fast"/"all"')
    parser.add_argument('--split_file', type=str, default=None,
                       help='Split file name (auto-detected if not specified)')
    
    args = parser.parse_args()

    args.cv_folds = parse_folds(args.folds, args.fold)
    args.fold = args.cv_folds[0]
    args.gpu_ids = parse_gpu_ids(args.gpus, args.gpu)
    args.cpu_sets = parse_cpu_sets(args.cpu_sets)
    if args.cpu_sets is not None and len(args.cpu_sets) != len(args.gpu_ids):
        raise ValueError('When --cpu_sets is provided, it must have the same number of entries as --gpus.')

    # Set study name
    if args.study_name is None:
        fold_tag = '-'.join(str(fold) for fold in args.cv_folds)
        args.study_name = f'{args.model_type}_{args.dataset}_folds-{fold_tag}'
    
    # Set device
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    # Set root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.root_dir = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Resolve model type / branch2 mode
    args.output_model_type = args.model_type
    args.model_type, args.branch2_enable, args.config_model_type = resolve_model_type(args.model_type)

    # Set split file automatically if not specified
    if args.split_file is None:
        args.split_file = f'{args.dataset}_5fold.json'
    
    # Fixed parameters
    args.eval_batch_size = 64
    args.validate_every = None
    args.max_obs = 880
    args.max_timesteps = 880

    base_config = load_model_config(args.config_model_type, args.dataset)
    if not base_config:
        raise ValueError(f'No config found for {args.config_model_type} on {args.dataset}')
    for key, value in base_config.items():
        setattr(args, key, value)
    
    print("=" * 80)
    print(f"Optuna Hyperparameter Tuning")
    print("=" * 80)
    print(f"Model: {args.output_model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Folds: {args.cv_folds}")
    print(f"Primary device: {args.device}")
    print(f"Parallel fold GPUs: {args.gpu_ids}")
    if args.cpu_sets is not None:
        print(f"Parallel fold CPU sets: {args.cpu_sets}")
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
    
    complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    # Print results
    print()
    print("=" * 80)
    print("Optimization Results")
    print("=" * 80)

    results_dir = os.path.join(args.root_dir, 'optuna_results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'{args.study_name}_best_params.txt')

    if not complete_trials:
        print('No completed trials were produced.')
        with open(results_file, 'w') as f:
            f.write(f"Study: {args.study_name}\n")
            f.write(f"Model: {args.output_model_type}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Folds: {args.cv_folds}\n")
            f.write('No completed trials were produced.\n')
        print("=" * 80)
        print(f"\nResults saved to: {results_file}")
        return

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (AUPRC): {study.best_value:.4f}")
    print()
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    with open(results_file, 'w') as f:
        f.write(f"Study: {args.study_name}\n")
        f.write(f"Model: {args.output_model_type}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Folds: {args.cv_folds}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_value:.4f}\n")
        f.write("\nBest hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")

    print(f"\nResults saved to: {results_file}")
    
if __name__ == '__main__':
    main()
