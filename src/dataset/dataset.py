"""
Main Dataset class for time series data loading and preprocessing.
"""
import hashlib
import pickle
import numpy as np
import os
import json
import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.append(str(Path(__file__).resolve().parent))
from utils import CycleIndex, Logger
from adapters import get_adapter

# Static variables for each dataset
STATIC_VARIS = {
    "mimic_iii": ['Age', 'Gender', 'Weight'],
    "physionet_2012": ['Age', 'Gender', 'Height', 'Weight', 'ICUType_1', 'ICUType_2', 'ICUType_3', 'ICUType_4'],
    "physionet_2019": ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
}


class Dataset:
    
    def __init__(self, args):
        # Read data
        args.logger.write(f'\nPreparing dataset {args.dataset}')
        data, oc = self._load_data(args)
        
        # Load pre-generated splits
        train_ids, val_ids, test_ids = self._load_splits(args)
        args.logger.write(f'Loaded splits from {args.split_file}')
        
        # Filter and rebuild splits(in case some variables are missing in training set yet present in val/test set)
        data = self._filter_data(args, data)
        data, train_ids, val_ids, test_ids = self._rebuild_splits(args, data, train_ids, val_ids, test_ids)
        args.logger.write(f'# train, val, test TS: {[len(train_ids), len(val_ids), len(test_ids)]}')
        
        # Build index mapping
        sup_ts_ids = np.concatenate((train_ids, val_ids, test_ids))
        sup_ts_ids_set = set(sup_ts_ids)
        ts_id_to_ind = {ts_id: i for i, ts_id in enumerate(sup_ts_ids)}
        ts_ind_to_id = {i: ts_id for ts_id, i in ts_id_to_ind.items()}
        
        # Filter data and add ts_ind column
        data = data.loc[data.ts_id.isin(sup_ts_ids_set)]
        data['ts_ind'] = data['ts_id'].map(ts_id_to_ind).astype(int)
        
        # Store ID mappings
        self.ts_id_to_ind = ts_id_to_ind
        self.ts_ind_to_id = ts_ind_to_id
        
        # Get labels
        oc = oc.loc[oc.ts_id.isin(sup_ts_ids)]
        oc['ts_ind'] = oc['ts_id'].map(ts_id_to_ind)
        oc = oc.sort_values(by='ts_ind')
        y = np.array(oc['label'])
        
        # Store basic info
        self.N = len(sup_ts_ids)
        self.y = y
        self.args = args
        self.static_varis = STATIC_VARIS[args.dataset]
        
        # Create splits
        self.splits = {
            'train': [ts_id_to_ind[i] for i in train_ids],
            'val': [ts_id_to_ind[i] for i in val_ids],
            'test': [ts_id_to_ind[i] for i in test_ids]
        }
        self.splits['eval_train'] = self.splits['train'][:2000]
        
        # Create training cycler
        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)
        
        # Log class statistics
        self._log_class_statistics(args, y, train_ids, val_ids, test_ids)
        
        # Extract static data with missingness indicator
        data, demo = self._extract_static_features(data)

        # Leave-variables-out robustness protocol (paper Table III).
        data = self._apply_variable_type_drop(args, data)

        # Prepare variable mappings (sorted for consistency)
        variables = sorted(data.variable.unique())
        var_to_ind = {v: i for i, v in enumerate(variables)}
        self.variables = variables
        self.var_to_ind = var_to_ind
        self.ind_to_var = {i: v for v, i in var_to_ind.items()}

        args.var_to_ind = var_to_ind
        args.ind_to_var = self.ind_to_var
        
        # Get model-specific adapter with y and demo
        self.adapter = get_adapter(args.model_type, args, args.logger, y, demo)
        # Pass train_ind (indices) instead of train_ids for normalization
        train_ind = self.splits['train']
        self.adapter.prepare_data(data, self.N, train_ind, var_to_ind, variables)
        
    def _load_data(self, args):
        """Load processed data (without split information)."""
        filepath = os.path.join(args.root_dir, 'data', 'processed', args.dataset + '.pkl')      
        data, oc = pickle.load(open(filepath, 'rb'))
        
        
        return data, oc
    
    def _load_splits(self, args):
        """
        Load pre-generated train/val/test splits from JSON file.
        
        The split file can be either:
        - K-fold format: Each fold contains 'train_ids', 'val_ids', 'test_ids'
        - Single split format: Contains 'train_ids', 'val_ids', 'test_ids'
        
        Args:
            args: Argument namespace with 'split_file' and 'fold' parameters
            
        Returns:
            Tuple of (train_ids, val_ids, test_ids) as numpy arrays
        """
        split_path = os.path.join(args.root_dir, 'data', 'splits', args.split_file)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Split file not found: {split_path}\n"
                f"Please run 'python data/process_scripts/generate_splits.py --dataset {args.dataset}' first."
            )
        
        with open(split_path, 'r') as f:
            splits = json.load(f)
        
        if 'folds' in splits:
            # K-fold format: each fold has its own train/val/test
            n_folds = splits['n_folds']
            fold_idx = args.fold - 1  # Convert 1-indexed to 0-indexed
            
            if fold_idx < 0 or fold_idx >= n_folds:
                raise ValueError(f"Invalid fold {args.fold}. Must be between 1 and {n_folds}.")
            
            fold_data = splits['folds'][fold_idx]
            train_ids = np.array(fold_data['train_ids'])
            val_ids = np.array(fold_data['val_ids'])
            test_ids = np.array(fold_data['test_ids'])
            
            args.logger.write(f'Using fold {args.fold}/{n_folds}')
        else:
            # Single split format
            train_ids = np.array(splits['train_ids'])
            val_ids = np.array(splits['val_ids'])
            test_ids = np.array(splits['test_ids'])
        
        return train_ids, val_ids, test_ids
    
    def _filter_data(self, args, data):
        """
        Filter data based on dataset-specific rules.
        
        Args:
            args: Argument namespace
            data: pandas DataFrame with raw time series data
            
        Returns:
            Filtered DataFrame
        """
        if args.dataset == 'mimic_iii':
            # Filter to first 24 hours
            data = data.loc[(data.minute >= 0) & (data.minute <= 24 * 60)]
            # Fill missing age for old patients (>200 indicates missing)
            data.loc[(data.variable == 'Age') & (data.value > 200), 'value'] = 91.4
            # Remove Height variable
            data = data.loc[data.variable != 'Height']
        
        if args.dataset == 'physionet_2012':
            # Remove blacklisted time series
            black_list = [140501, 150649, 140936, 143656, 141264, 145611, 142998, 
                         147514, 142731, 150309, 155655, 156254]
            data = data.loc[~data.ts_id.isin(black_list)]
        
        if args.dataset == 'physionet_2019':
            # Filter patients: (1) max observation time <= 60 hours, (2) at least 3 unique timestamps
            patient_stats = data.groupby('ts_id')['minute'].agg(['max', 'nunique'])
            patient_stats['max_hours'] = patient_stats['max'] / 60
            valid_patients = patient_stats[
                (patient_stats['max_hours'] <= 60) & 
                (patient_stats['nunique'] > 3)
            ].index
            data = data.loc[data.ts_id.isin(valid_patients)]
        
        return data
    
    def _rebuild_splits(self, args, data, train_ids, val_ids, test_ids):
        """
        Rebuild train/val/test splits and remove variables not in training set.
        """
        # Remove variables not in training set
        train_ids_set = set(train_ids)
        train_variables = data.loc[data.ts_id.isin(train_ids_set)].variable.unique()
        all_variables = data.variable.unique()
        delete_variables = np.setdiff1d(all_variables, train_variables)
        
        if len(delete_variables) > 0:
            args.logger.write(f'Removing variables not in training set: {delete_variables}')
        
        data = data.loc[data.variable.isin(set(train_variables))]
        
        # Filter splits to only include ts_ids present in filtered data
        curr_ids = set(data.ts_id.unique())
        train_ids = np.array([i for i in train_ids if i in curr_ids])
        val_ids = np.array([i for i in val_ids if i in curr_ids])
        test_ids = np.array([i for i in test_ids if i in curr_ids])
        
        return data, train_ids, val_ids, test_ids
    
    def _log_class_statistics(self, args, y, train_ids, val_ids, test_ids):
        """
        Log class distribution and compute positive class weight.
        
        Args:
            args: Argument namespace
            y: Labels array
            train_ids: Training set IDs
            val_ids: Validation set IDs
            test_ids: Test set IDs
        """
        num_train = len(train_ids)
        num_train_pos = y[self.splits['train']].sum()
        
        # Compute positive class weight for imbalanced data
        args.pos_class_weight = (num_train - num_train_pos) / num_train_pos
        args.logger.write(f'pos class weight: {args.pos_class_weight:.4f}')
        
        # Log class distribution
        num_val = len(val_ids)
        num_test = len(test_ids)
        val_pos_pct = y[self.splits["val"]].sum() / num_val if num_val > 0 else 0
        test_pos_pct = y[self.splits["test"]].sum() / num_test if num_test > 0 else 0
        
        args.logger.write(
            f'% pos class in train, val, test splits: '
            f'[{num_train_pos/num_train:.4f}, {val_pos_pct:.4f}, {test_pos_pct:.4f}]'
        )
    
    def _extract_static_features(self, data):
        """
        Extract static/demographic features and handle missing values.
        
        Args:
            data: pandas DataFrame with time series data
            
        Returns:
            DataFrame with static variables removed
        """
        # Separate static variables from time series
        static_ii = data.variable.isin(self.static_varis)
        static_data = data.loc[static_ii]
        data = data.loc[~static_ii]
        
        # Create variable index mapping
        static_var_to_ind = {v: i for i, v in enumerate(self.static_varis)}
        D = len(static_var_to_ind)
        
        # Initialize with -1 to indicate missing values
        demo = np.full((self.N, D), -1.0)
        
        # Fill in observed static values
        for row in static_data.itertuples():
            var_ind = static_var_to_ind[row.variable]
            demo[row.ts_ind, var_ind] = row.value
        
        # Normalize static data using training statistics
        train_ind = self.splits['train']
        demo_train = demo[train_ind]
        
        # Create mask: observed values are != -1
        mask = (demo != -1).astype(float)
        
        # Compute mean and std from observed values only (exclude -1 missing values)
        means = np.zeros(D)
        stds = np.ones(D)
        for d in range(D):
            obs_vals = demo_train[:, d][demo_train[:, d] != -1]
            if len(obs_vals) > 0:
                means[d] = obs_vals.mean()
                stds[d] = obs_vals.std()
                if stds[d] == 0:
                    stds[d] = 1
        
        # Normalize and apply mask (missing values become 0)
        demo = (demo - means.reshape(1, D)) / stds.reshape(1, D)
        demo = demo * mask
        
        self.args.logger.write(f'# static features: {D}')
        self.args.D = D

        return data, demo

    def _apply_variable_type_drop(self, args, data):
        drop_rate = float(getattr(args, 'variable_drop_rate', 0.0) or 0.0)
        variables = sorted(data.variable.unique())
        num_variables = len(variables)

        drop_info = {
            'enabled': drop_rate > 0.0,
            'drop_rate': drop_rate,
            'original_num_variables': num_variables,
            'dropped_num_variables': 0,
            'remaining_num_variables': num_variables,
            'resolved_seed': None,
            'dropped_variables': [],
        }

        if drop_rate < 0.0 or drop_rate >= 1.0:
            raise ValueError(f'variable_drop_rate must be in [0, 1), got {drop_rate}')

        if drop_rate <= 0.0 or num_variables == 0:
            self.variable_drop_info = drop_info
            args.variable_drop_info = drop_info
            return data

        drop_count = int(num_variables * drop_rate)
        if drop_count >= num_variables:
            raise ValueError(
                f'variable_drop_rate={drop_rate} drops all {num_variables} variables.'
            )

        if drop_count == 0:
            self.args.logger.write(
                f'Variable drop requested at rate {drop_rate:.2f}, '
                f'but 0 of {num_variables} variables were selected.'
            )
            self.variable_drop_info = drop_info
            args.variable_drop_info = drop_info
            return data

        resolved_seed = self._resolve_variable_drop_seed(args, drop_rate)
        rng = np.random.default_rng(resolved_seed)
        dropped_variables = sorted(
            rng.choice(np.array(variables, dtype=object), size=drop_count, replace=False).tolist()
        )

        data = data.loc[~data.variable.isin(set(dropped_variables))].copy()
        drop_info.update({
            'dropped_num_variables': drop_count,
            'remaining_num_variables': num_variables - drop_count,
            'resolved_seed': resolved_seed,
            'dropped_variables': dropped_variables,
        })
        self.args.logger.write(
            'Variable-type drop enabled: '
            f'rate={drop_rate:.2f}, dropped={drop_count}, remaining={num_variables - drop_count}'
        )
        self.args.logger.write(f'Dropped variables: {dropped_variables}')
        self.variable_drop_info = drop_info
        args.variable_drop_info = drop_info
        return data

    def _resolve_variable_drop_seed(self, args, drop_rate):
        base_seed = getattr(args, 'variable_drop_seed', None)
        if base_seed is None:
            base_seed = getattr(args, 'seed', 0)
        token = f'{args.dataset}|{args.fold}|{drop_rate:.8f}|{base_seed}'
        digest = hashlib.sha256(token.encode('utf-8')).digest()
        return int.from_bytes(digest[:8], 'big') % (2 ** 32)

    def get_batch(self, ind=None):
        """
        Get a batch of data.
        
        Args:
            ind: Optional array of indices. If None, uses train_cycler to get next batch.
            
        Returns:
            Dictionary with model-specific batch data
        """
        if ind is None:
            ind = self.train_cycler.get_batch_ind()
        
        return self.adapter.get_batch(ind)


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[2]
    dataset_name = "physionet_2012"

    args = SimpleNamespace(
        root_dir=str(root_dir),
        dataset=dataset_name,
        split_file=f"{dataset_name}_5fold.json",
        fold=1,
        train_batch_size=4,
        model_type="dbvt",
        logger=Logger(),
    )

    dataset = Dataset(args)
    batch = dataset.get_batch(np.arange(2))

    print("N:", dataset.N)
    print("train/val/test:", len(dataset.splits["train"]), len(dataset.splits["val"]), len(dataset.splits["test"]))
    print("observed_data:", batch["observed_data"].shape)
    print("observed_mask:", batch["observed_mask"].shape)
    print("observed_tp:", batch["observed_tp"].shape)
    print("tau:", batch["tau"].shape)
    print("demo:", batch["demo"].shape)
    print("labels:", batch["labels"])
