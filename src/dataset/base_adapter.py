"""
Base data adapter for model-specific data preparation.
"""
from abc import ABC, abstractmethod
import torch
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
        means_stds = data.loc[data.ts_id.isin(train_ids)].groupby('variable').agg({'value': ['mean', 'std']})
        means_stds.columns = [col[1] for col in means_stds.columns]
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
    
    def normalize_array(self, values, train_ind):
        """
        Normalize array using training statistics.
        
        Args:
            values: Values array to normalize
            train_ind: Indices of training samples
            
        Returns:
            Normalized array
        """
        means = values[train_ind].mean(axis=(0, 1), keepdims=True)
        stds = values[train_ind].std(axis=(0, 1), keepdims=True)
        stds = np.where(stds == 0, 1, stds)
        normalized = (values - means) / stds
        return normalized

    def normalize_zscore_array(self, values, mask, train_ind):
        """Z-score normalization per variable using training set statistics."""
        N, T, V = values.shape
        
        # Get training data using indices
        data_train = values[train_ind]
        mask_train = mask[train_ind]
        
        # Compute per-variable mean and std from observed values
        means = np.zeros(V, dtype=np.float32)
        stds = np.ones(V, dtype=np.float32)
        for v in range(V):
            obs_vals = data_train[:, :, v][mask_train[:, :, v] > 0]
            if len(obs_vals) > 0:
                means[v] = obs_vals.mean()
                stds[v] = obs_vals.std()
                if stds[v] == 0:
                    stds[v] = 1
        
        # Normalize values (only where observed, missing stays 0)
        values = (values - means.reshape(1, 1, V)) / stds.reshape(1, 1, V)
        values = values * mask  # Zero out unobserved values
        return values
    