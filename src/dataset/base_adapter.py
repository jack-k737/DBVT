"""
Base data adapter for model-specific data preparation.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseDataAdapter(ABC):
    """
    Base class for model-specific data adapters.
    
    Each model can have its own adapter that handles:
    - Data preparation (normalization, aggregation, etc.)
    - Batch generation with model-specific formats
    """
    
    def __init__(self, args, logger, y=None, demo=None):
        """
        Initialize the adapter.
        
        Args:
            args: Argument namespace containing model hyperparameters
            logger: Logger instance for writing messages
            y: Labels array [N] (optional, can be set later)
            demo: Static/demographic features array [N, D] (optional, can be set later)
        """
        self.args = args
        self.logger = logger
        self.y = y
        self.demo = demo
    
    @abstractmethod
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        """
        Prepare data in model-specific format.
        
        Args:
            data: pandas DataFrame with columns [ts_id, ts_ind, minute, variable, value]
            N: Total number of time series
            train_ind: List of training time series INDICES (not IDs) for normalization
            var_to_ind: Dictionary mapping variable names to indices
            variables: Array of variable names
        """
        pass
    
    @abstractmethod
    def get_batch(self, ind):
        """
        Get batch in model-specific format.
        
        Args:
            ind: Array of indices for the batch
            
        Returns:
            Dictionary with model-specific batch data
            
        Note:
            Uses self.demo and self.y which should be set during initialization or prepare_data
        """
        pass
    
    # ========== Common utility methods ==========
    
    def normalize_zscore(self, data, train_ids):
        """
        Apply Z-score normalization using training statistics.
        
        Args:
            data: pandas DataFrame with 'value' column
            train_ids: List of training time series IDs
            
        Returns:
            Normalized DataFrame with added 'mean' and 'std' columns
        """
        means_stds = data.loc[data.ts_id.isin(train_ids)].groupby('variable').agg(
                mean=('value', 'mean'),
                std=('value', lambda s: s.std(ddof=0))
        )
        means_stds.loc[means_stds['std'] == 0, 'std'] = 1
        data = data.merge(means_stds.reset_index(), on='variable', how='left')
        data['value'] = (data['value'] - data['mean']) / data['std']
        return data
    
    def pad_sequences(self, sequences, pad_value=0):
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of sequences (lists or arrays)
            pad_value: Value to use for padding
            
        Returns:
            Tuple of (padded_sequences, lengths, pad_lengths)
        """
        lengths = np.array([len(seq) for seq in sequences])
        max_len = max(lengths)
        pad_lens = max_len - lengths
        padded = [list(seq) + [pad_value] * int(pad_len) 
                  for seq, pad_len in zip(sequences, pad_lens)]
        return padded, lengths, pad_lens
    
    def compute_delta_time(self, obs, T):
        """
        Compute time since last observation (delta) for each variable.
        
        Args:
            obs: Observation indicator array [N, T, V]
            T: Number of time steps
            
        Returns:
            Delta array [N, T, V] normalized by T
        """
        N, T, V = obs.shape
        delta = np.zeros((N, T, V))
        delta[:, 0, :] = obs[:, 0, :]

        for t in range(1, T):
            delta[:, t, :] = obs[:, t, :] * 0 + (1 - obs[:, t, :]) * (1 + delta[:, t-1, :])
        
        delta = delta / T
        return delta
    
    def mean_fill_missing(self, values, obs, train_ind):
        """
        Fill missing values with training mean.
        
        Args:
            values: Values array [N, T, V]
            obs: Observation indicator array [N, T, V]
            train_ind: Indices of training samples
            
        Returns:
            Filled values array [N, T, V]
        """
        means = (values[train_ind] * obs[train_ind]).sum(axis=(0, 1)) / (obs[train_ind].sum(axis=(0, 1)) + 1e-8)
        V = values.shape[-1]
        filled = values * obs + (1 - obs) * means.reshape((1, 1, V))
        return filled
    
    def prepare_data_for_branch2(self, data, N, var_to_ind):
        data = data.copy()
        if "hour" in data.columns:
            data["branch2_t"] = np.floor(data["hour"]).astype(int).clip(lower=0)
        elif "timestep" in data.columns:
            data["branch2_t"] = data["timestep"].astype(int).clip(lower=0)
        else:
            data["branch2_t"] = data["minute"].apply(lambda x: max(0, int(x // 60)))

        V = len(var_to_ind)
        T = int(data["branch2_t"].max()) + 1
        # Construct data and mask arrays
        observed_data = np.zeros((N, T, V), dtype=np.float32)
        observed_mask = np.zeros((N, T, V), dtype=np.float32)
        for row in data.itertuples():
            vind = var_to_ind[row.variable]
            tstep = row.branch2_t
            observed_data[row.ts_ind, tstep, vind] = row.value
            observed_mask[row.ts_ind, tstep, vind] = 1.0

        # Construct GRU specific features
        lengths = np.zeros(N, dtype=np.int64)
        for i in range(N):
            obs_times = np.where(observed_mask[i].sum(axis=1) > 0)[0]
            if len(obs_times) > 0:
                lengths[i] = obs_times.max() + 1
            else:
                lengths[i] = 1
        return observed_data, observed_mask, lengths



if __name__ == "__main__":
    import json
    import pickle
    from pathlib import Path

    class _DummyAdapter(BaseDataAdapter):
        def prepare_data(self, data, N, train_ind, var_to_ind, variables):
            return None

        def get_batch(self, ind):
            return None

    dataset = "physionet_2012"
    fold = 1
    root_dir = Path(__file__).resolve().parents[2]
    data_path = root_dir / "data" / "processed" / f"{dataset}.pkl"
    split_path = root_dir / "data" / "splits" / f"{dataset}_5fold.json"

    data, _ = pickle.load(open(data_path, "rb"))
    with open(split_path, "r") as f:
        split_cfg = json.load(f)

    train_ids = split_cfg["folds"][fold - 1]["train_ids"]

    adapter = _DummyAdapter(args=None, logger=None)
    normalized = adapter.normalize_zscore(data.copy(), train_ids)
    train_before = data.loc[data.ts_id.isin(train_ids)]
    train_after = normalized.loc[normalized.ts_id.isin(train_ids)]

    print(f"dataset={dataset}, fold={fold}, train_ids={len(train_ids)}")
    print("\nBefore normalize:")
    print(train_before.groupby("variable")["value"].agg(["mean", "std"]).head())
    print("\nAfter normalize:")
    print(train_after.groupby("variable")["value"].agg(["mean", "std"]).head())
    print("\nSample rows:")
    print(normalized.head(20))
    
