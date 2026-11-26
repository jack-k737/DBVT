import argparse
import glob
import json
import os
import re
import shlex
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from utils.logger import Logger
from utils.config_loader import apply_config
from dataset import Dataset
from models import (
    ModelWrapper
)


import torch
import torch.nn as nn
import random
from torch.backends import cudnn
import numpy as np
from torch.optim import AdamW
from utils import Evaluator

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)

def set_all_seeds(seed: int) -> None:
    """Function to set seeds for all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count()>0:
        torch.cuda.manual_seed_all(seed)

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

def set_output_dir(args: argparse.Namespace) -> None:
    """Function to automatically set output dir 
    if it is not passed in args."""
    if args.output_dir is None:
        output_model_type = getattr(args, 'output_model_type', args.model_type)
        args.output_dir = args.root_dir + '/outputs/' + args.dataset + '/' + output_model_type
        args.output_dir += f'|fold:{args.fold}'
    os.makedirs(args.output_dir, exist_ok=True)

def set_split_file(args: argparse.Namespace) -> None:
    """Function to automatically set split file if not specified."""
    if args.split_file is None:
        splits_dir = os.path.join(args.root_dir, 'data', 'splits')
        
        # Try simple format first: {dataset}_5fold.json
        candidate = f'{args.dataset}_5fold.json'
        candidate_path = os.path.join(splits_dir, candidate)
        if os.path.exists(candidate_path):
            args.split_file = candidate
            return
        
        # Try to find any matching file
        pattern = os.path.join(splits_dir, f'{args.dataset}_*.json')
        matches = glob.glob(pattern)
        if matches:
            args.split_file = os.path.basename(matches[0])
            return
        
        # Default (will raise error in dataset.py if not found)
        args.split_file = f'{args.dataset}_5fold.json'

def _parse_override_value(raw: str):
    lowered = raw.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if re.fullmatch(r'[-+]?\d+', raw):
        return int(raw)
    if re.fullmatch(r'[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', raw):
        return float(raw)
    return raw


def apply_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Apply repeated --override key=value arguments after config loading."""
    for override in getattr(args, 'override', []):
        if '=' not in override:
            raise ValueError(f'Invalid override "{override}". Expected KEY=VALUE.')
        key, raw_value = override.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f'Invalid override "{override}". Key cannot be empty.')
        setattr(args, key, _parse_override_value(raw_value.strip()))
    return args



def resolve_model_variant(raw_model_type: str) -> tuple[str, str, bool]:
    """
    Resolve runtime model type, config lookup name, and branch2 flag.

    Rules:
    - dbvt  -> paper name alias for historical transformer+
    - xxx   -> single branch, use configs/xxx.yaml
    - xxx+  -> dual branch, prefer configs/xxx+.yaml
    - xxx++ -> dual branch, use configs/xxx++.yaml
    """
    model_type = raw_model_type.strip()
    if model_type.lower() == 'dbvt':
        return 'transformer', 'transformer+', True
    if model_type.endswith('++'):
        base_model_type = model_type[:-2]
        return base_model_type, model_type, True
    if model_type.endswith('+'):
        base_model_type = model_type[:-1]
        return base_model_type, model_type, True
    return model_type, model_type, False


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Model-specific parameters are loaded from config files in configs/{model_type}.yaml
    Command-line arguments can override config file values for tuning.
    """
    parser = argparse.ArgumentParser(
        description='Time Series Classification with Multiple Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========== Core Arguments (Required) ==========
    parser.add_argument('--dataset', type=str, default='physionet_2012',
                        choices=['physionet_2012', 'physionet_2019', 'mimic_iii'])
    parser.add_argument('--model_type', type=str, default='dbvt')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Split file name in data/splits/ (e.g., physionet_2012_5fold_seed2026.json)')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold number for k-fold CV (1-indexed)')

    # ========== Path Arguments ==========
    parser.add_argument(
        '--root_dir',
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
    )
    parser.add_argument('--output_dir', type=str, default=None)

    # ========== Training Arguments (Commonly Tuned) ==========
    # parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--validate_every', type=int, default=None)
    # parser.add_argument('--grad_clip_norm', type=float, default=3.0)

    # ========== Other Model Parameters ==========
    # Note: Common parameters (hid_dim, num_layers, num_heads, dropout) are loaded from config files
    # Only add parameters here if you need to override them frequently via command line
    parser.add_argument('--max_obs', type=int, default=880)
    parser.add_argument('--max_timesteps', type=int, default=880)

    # ========== Other Arguments ==========
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--override', action='append', default=[],
                        help='Override config/model parameter as key=value. Can be passed multiple times.')

    # ========== Paper Experiments ==========
    parser.add_argument('--ablation_mode', type=str, default=None,
                        choices=['full', 'no_gru', 'no_transformer', 'no_fusion'],
                        help='Table IV ablation: freeze one branch or use concat fusion.')
    parser.add_argument('--variable_drop_rate', type=float, default=0.0,
                        help='Table III leave-variables-out ratio, e.g. 0.1, 0.3, 0.5.')
    parser.add_argument('--variable_drop_seed', type=int, default=None,
                        help='Seed for variable removal (defaults to --seed).')


    args = parser.parse_args()
    run_tag = ''
    if args.ablation_mode:
        run_tag += f'|ablation:{args.ablation_mode}'
    if args.variable_drop_rate > 0:
        run_tag += f'|drop:{int(round(args.variable_drop_rate * 100))}'
    args.output_model_type = args.model_type.strip() + run_tag
    args.plus_ablation_mode = args.ablation_mode or 'full'
    args.model_type, args.config_model_type, args.branch2_enable = resolve_model_variant(
        args.model_type
    )

    # Load model config and then apply explicit overrides.
    args = apply_config(args)
    args = apply_overrides(args)
    # Set split file and output directory
    set_split_file(args)
    set_output_dir(args)

    return args




if __name__ == "__main__":
    # Preliminary setup.
    args = parse_args()
    
    
    
    args.logger = Logger(args.output_dir, 'log.txt')
    args.logger.write(str(args))
    # args.device = torch.device('cuda')
    set_all_seeds(args.seed + args.fold)
    model_path_best = os.path.join(args.output_dir, 'checkpoint_best.bin')

    # load data
    dataset = Dataset(args)

    # load model
    model = ModelWrapper(args)
    model.to(args.device)
    count_parameters(args.logger, model)
    # training loop
    num_train = len(dataset.splits['train'])
    num_batches_per_epoch = int(np.ceil(num_train / args.train_batch_size))
    args.logger.write(f'\nNo. of training batches per epoch = {num_batches_per_epoch}')
    
    wait = args.patience
    best_val_metric = 0
    best_val_res, best_test_res = None, None
    optimizer = AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
    evaluator = Evaluator(args)
    
    args.logger.write('\n' + '='*60)
    args.logger.write(' '*15 + 'Starting training...')
    args.logger.write('='*60)
    
    model.train()
    for epoch in range(1, args.max_epochs + 1):
        args.logger.write(f'>>> Epoch {epoch}/{args.max_epochs}')
        epoch_loss = 0.0
        num_batches = 0
        
        # Train for one epoch
        for batch_idx in range(num_batches_per_epoch):
            
            # load batch
            batch = dataset.get_batch()
            batch = {k:v.to(args.device) for k,v in batch.items()}

            # forward pass
            loss = model(**batch)
            
            # backward pass
            if not torch.isnan(loss):
                epoch_loss += loss.item()
                num_batches += 1
                
                loss.backward()
                total = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
        
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        args.logger.write(f'Epoch {epoch} completed | Avg Loss: {avg_loss:.4f}')
        
        # Validation at the end of each epoch
        evaluator.evaluate(model, dataset, 'eval_train', epoch=epoch)
        val_res = evaluator.evaluate(model, dataset, 'val', epoch=epoch)
        model.train(True)

        # Save ckpt if there is an improvement on validation set (using AUPRC as metric)
        curr_val_metric = val_res['auprc']
        if curr_val_metric > best_val_metric:
            # Improvement found
            improvement = curr_val_metric - best_val_metric
            args.logger.write('AUPRC improved: {:.4f} ---> {:.4f} (+{:.4f})'.format(
                best_val_metric, curr_val_metric, improvement))
            
            best_val_metric = curr_val_metric
            best_val_res = val_res
            args.logger.write('Saving checkpoint to ' + model_path_best)
            torch.save(model.state_dict(), model_path_best)
            wait = args.patience
        else:
            wait -= 1
            args.logger.write(f'No improvement. Wait: {wait}/{args.patience}')
            if wait == 0:
                args.logger.write('Patience reached. Early stopping.')
                break
        
        args.logger.write('='*60)
    
    # Load best model and evaluate on test set
    args.logger.write('\n' + '='*60)
    args.logger.write('TRAINING COMPLETED')
    args.logger.write('='*60)

    args.logger.write('Best Validation Results:')
    args.logger.write('  AUROC: {:.4f} | AUPRC: {:.4f} | MinRP: {:.4f}'.format(
        best_val_res['auroc'], best_val_res['auprc'], best_val_res['minrp']))
    
    # Load best model and test
    args.logger.write('Loading best model from ' + model_path_best)
    model.load_state_dict(torch.load(model_path_best))
    args.logger.write('Evaluating on test set...')
    test_res = evaluator.evaluate(model, dataset, 'test')
    args.logger.write('Final Test Results:')
    args.logger.write('  AUROC: {:.4f} | AUPRC: {:.4f} | MinRP: {:.4f} | Accuracy: {:.4f}'.format(
        test_res['auroc'], test_res['auprc'], test_res['minrp'], test_res['accuracy']))
    args.logger.write('='*60)
