"""
Generate train/val/test splits for medical time series datasets.

Usage:
    python generate_splits.py --dataset physionet_2012
    python generate_splits.py --dataset mimic_iii
    python generate_splits.py --dataset physionet_2019
"""

import argparse
import os
import pickle
import json
import numpy as np


def load_processed_data(dataset_name: str, processed_dir: str = 'processed'):
    """Load processed data."""
    filepath = os.path.join(processed_dir, f'{dataset_name}.pkl')
    loaded = pickle.load(open(filepath, 'rb'))
    
    # Support both old format [data, oc, train, val, test] and new format [data, oc, None, None, None]
    if len(loaded) == 5:
        data, oc, _, _, _ = loaded
    elif len(loaded) == 2:
        data, oc = loaded
    else:
        raise ValueError(f"Unexpected pickle format with {len(loaded)} elements")
    
    # Get subject info if available (for patient-level splitting)
    subject_mapping = None
    if 'SUBJECT_ID' in oc.columns:
        subject_mapping = oc.set_index('ts_id')['SUBJECT_ID'].to_dict()
    
    return data, oc, subject_mapping


def filter_data_by_dataset(data, oc, dataset_name: str):
    """Apply dataset-specific filtering."""
    if dataset_name == 'mimic_iii':
        data_filtered = data.loc[(data.minute >= 0) & (data.minute <= 24 * 60)]
        valid_ts_ids = set(data_filtered.ts_id.unique())
    elif dataset_name == 'physionet_2012':
        black_list = {140501, 150649, 140936, 143656, 141264, 145611, 142998, 
                     147514, 142731, 150309, 155655, 156254}
        valid_ts_ids = set(data.ts_id.unique()) - black_list
    elif dataset_name == 'physionet_2019':
        patient_stats = data.groupby('ts_id')['minute'].agg(['max', 'nunique'])
        patient_stats['max_hours'] = patient_stats['max'] / 60
        valid_patients = patient_stats[
            (patient_stats['max_hours'] <= 60) & 
            (patient_stats['nunique'] > 3)
        ].index
        valid_ts_ids = set(valid_patients)
    else:
        valid_ts_ids = set(data.ts_id.unique())
    
    # Only include ts_ids that have labels
    labeled_ts_ids = set(oc.ts_id.unique())
    valid_ts_ids = valid_ts_ids & labeled_ts_ids
    
    return np.array(list(valid_ts_ids))


def generate_5fold_splits(ts_ids: np.ndarray, val_ratio: float, subject_mapping: dict = None, seed: int = 2026):
    """
    Generate 5-fold cross validation splits.
    
    Each fold uses a different 20% as test set. From remaining 80%, 
    val_ratio is used for validation, rest for training.
    
    Args:
        ts_ids: Array of time series IDs
        val_ratio: Ratio of trainval data for validation (default: 0.1 means 10% of trainval)
        subject_mapping: Optional mapping from ts_id to subject_id for patient-level splitting
        seed: Random seed
    
    Returns:
        Dictionary with 'folds' list
    """
    np.random.seed(seed)
    n_folds = 5
    
    if subject_mapping is not None:
        # Patient-level splitting
        unique_subjects = list(set(subject_mapping.values()))
        np.random.shuffle(unique_subjects)
        fold_size = len(unique_subjects) // n_folds
        
        folds = []
        for fold_idx in range(n_folds):
            if fold_idx < n_folds - 1:
                test_subjects = set(unique_subjects[fold_idx * fold_size: (fold_idx + 1) * fold_size])
            else:
                test_subjects = set(unique_subjects[fold_idx * fold_size:])
            
            trainval_subjects = [s for s in unique_subjects if s not in test_subjects]
            n_val = int(len(trainval_subjects) * val_ratio)
            val_subjects = set(trainval_subjects[:n_val])
            train_subjects = set(trainval_subjects[n_val:])
            
            fold_train_ids = [int(tid) for tid in ts_ids if subject_mapping.get(tid) in train_subjects]
            fold_val_ids = [int(tid) for tid in ts_ids if subject_mapping.get(tid) in val_subjects]
            fold_test_ids = [int(tid) for tid in ts_ids if subject_mapping.get(tid) in test_subjects]
            
            folds.append({
                'train_ids': fold_train_ids,
                'val_ids': fold_val_ids,
                'test_ids': fold_test_ids
            })
    else:
        # Simple random splitting
        ts_ids = np.array(ts_ids)
        np.random.shuffle(ts_ids)
        fold_size = len(ts_ids) // n_folds
        
        folds = []
        for fold_idx in range(n_folds):
            if fold_idx < n_folds - 1:
                test_ids_fold = ts_ids[fold_idx * fold_size: (fold_idx + 1) * fold_size]
            else:
                test_ids_fold = ts_ids[fold_idx * fold_size:]
            
            trainval_ids = np.setdiff1d(ts_ids, test_ids_fold)
            n_val = int(len(trainval_ids) * val_ratio)
            np.random.shuffle(trainval_ids)
            val_ids_fold = trainval_ids[:n_val]
            train_ids_fold = trainval_ids[n_val:]
            
            folds.append({
                'train_ids': [int(tid) for tid in train_ids_fold],
                'val_ids': [int(tid) for tid in val_ids_fold],
                'test_ids': [int(tid) for tid in test_ids_fold]
            })
    
    return {'folds': folds, 'n_folds': n_folds}


def main():
    parser = argparse.ArgumentParser(description='Generate 5-fold CV splits')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['physionet_2012', 'physionet_2019', 'mimic_iii'],
                        help='Dataset name')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation ratio from trainval data (default: 0.1)')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Random seed (default: 2026)')
    parser.add_argument('--output_dir', type=str, default='../splits',
                        help='Output directory (default: ../splits)')
    parser.add_argument('--processed_dir', type=str, default='../processed',
                        help='Processed data directory (default: ../processed)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and filter data
    print(f"Loading data for {args.dataset}...")
    data, oc, subject_mapping = load_processed_data(args.dataset, args.processed_dir)
    
    print(f"Filtering data...")
    valid_ts_ids = filter_data_by_dataset(data, oc, args.dataset)
    print(f"Valid samples after filtering: {len(valid_ts_ids)}")
    
    # Generate 5-fold splits
    print(f"Generating 5-fold cross validation splits...")
    splits = generate_5fold_splits(valid_ts_ids, args.val_ratio, subject_mapping, args.seed)
    
    # Fixed output filename
    output_filename = f"{args.dataset}_5fold.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Save splits
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of folds: 5")
    print(f"Validation ratio: {args.val_ratio:.2%} of trainval data")
    print()
    
    for i, fold in enumerate(splits['folds']):
        total = len(fold['train_ids']) + len(fold['val_ids']) + len(fold['test_ids'])
        train_pct = len(fold['train_ids']) / total * 100
        val_pct = len(fold['val_ids']) / total * 100
        test_pct = len(fold['test_ids']) / total * 100
        print(f"Fold {i+1}: Train {len(fold['train_ids'])} ({train_pct:.1f}%), "
              f"Val {len(fold['val_ids'])} ({val_pct:.1f}%), "
              f"Test {len(fold['test_ids'])} ({test_pct:.1f}%)")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
