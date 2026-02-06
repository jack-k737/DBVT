"""
Data adapter for KEDGN model.
"""
import torch
import numpy as np
from ..base_adapter import BaseDataAdapter

from numba import jit, prange

@jit(nopython=True, parallel=True)
def compute_interval_numba(mask, length, N, T, V):
    """
    Compute interval features using numba for acceleration.
    
    For each observation, compute the average time interval to neighboring observations.
    """
    interval = np.zeros((N, T, V), dtype=np.float32)
    for i in prange(N):
        seq_len = length[i, 0]
        for v in range(V):
            # Find non-zero indices
            count = 0
            for t in range(T):
                if mask[i, t, v] > 0:
                    count += 1
            if count == 0:
                continue
            idx_not_zero = np.zeros(count, dtype=np.int64)
            k = 0
            for t in range(T):
                if mask[i, t, v] > 0:
                    idx_not_zero[k] = t
                    k += 1
            if count == 1:
                interval[i, idx_not_zero[0], v] = seq_len / 2.0
            else:
                for j in range(count):
                    if j == 0:
                        left = idx_not_zero[0]
                    else:
                        left = idx_not_zero[j] - idx_not_zero[j-1]
                    if j == count - 1:
                        right = seq_len - idx_not_zero[j]
                    else:
                        right = idx_not_zero[j+1] - idx_not_zero[j]
                    interval[i, idx_not_zero[j], v] = (left + right) / 2.0
    return interval


class KEDGNAdapter(BaseDataAdapter):
    
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        # Get variable mappings (sorted for consistency)
        variables_sorted = sorted(variables)
        var_to_ind_sorted = {v: i for i, v in enumerate(variables_sorted)}
        V = len(variables_sorted)
        
        self.args.V = V
        self.logger.write(f'# TS variables: {V}')
        
        # Convert minute to hour for KEDGN (hourly aggregation)
        data = data.copy()
        data['hour'] = data['minute'].apply(lambda x: max(0, int(x // 60)))
        T = data.hour.max() + 1
        self.args.T = T
        self.logger.write(f'# intervals (hours): {T}')
        
        # Initialize arrays [N, T, V]
        arr = np.zeros((N, T, V), dtype=np.float32)
        mask = np.zeros((N, T, V), dtype=np.float32)
        time_arr = np.zeros((N, T, V), dtype=np.float32)
        
        t_max = data.hour.max()
        # Fill arrays
        for row in data.itertuples():
            vind = var_to_ind_sorted[row.variable]
            tstep = row.hour
            arr[row.ts_ind, tstep, vind] = row.value
            mask[row.ts_ind, tstep, vind] = 1
            time_arr[row.ts_ind, tstep, vind] = (row.hour / t_max) * 2 - 1  # normalize to [-1, 1]
        
        # Compute sequence lengths
        length = np.zeros((N, 1), dtype=np.int64)
        for i in range(N):
            obs_times = np.where(mask[i].sum(axis=1) > 0)[0]
            if len(obs_times) > 0:
                length[i, 0] = obs_times.max() + 1
            else:
                length[i, 0] = 1
        
        
        # Mean fill and normalize using training data
        means = (arr[train_ind] * mask[train_ind]).sum(axis=(0, 1)) / (mask[train_ind].sum(axis=(0, 1)) + 1e-8)
        arr = arr * mask + (1 - mask) * means.reshape((1, 1, V))
        
        means_norm = arr[train_ind].mean(axis=(0, 1), keepdims=True)
        stds_norm = arr[train_ind].std(axis=(0, 1), keepdims=True)
        stds_norm = (stds_norm == 0) * 1 + (stds_norm > 0) * stds_norm
        arr = (arr - means_norm) / stds_norm
        
        # Compute interval features
        interval = compute_interval_numba(mask, length, N, T, V)
        
        # Store prepared data
        self.arr = arr
        self.mask = mask
        self.time = time_arr
        self.interval = interval
        self.length = length

    
    def get_batch(self, ind):
        """
        Get batch for KEDGN model.
        
        Args:
            ind: Array of indices for the batch
        
        Returns:
            dict with KEDGN-specific batch data
        """
        return {
            'arr': torch.FloatTensor(self.arr[ind]),
            'mask': torch.FloatTensor(self.mask[ind]),
            'time': torch.FloatTensor(self.time[ind]),
            'interval': torch.FloatTensor(self.interval[ind]),
            'length': torch.LongTensor(self.length[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }

