"""
Data adapter for Raindrop model.
"""
import torch
import numpy as np
from ..base_adapter import BaseDataAdapter


class RaindropAdapter(BaseDataAdapter):
    """
    Data adapter for Raindrop model.
    
    Raindrop uses graph neural networks on time series with dense format.
    Expects:
        - src: [T, B, 2*V] - values concatenated with mask
        - times: [T, B] - timestamps
        - static: [B, D] - static features
        - lengths: [B] - sequence lengths
    """
    
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        """
        Prepare data for Raindrop model with dense format.
        
        Args:
            data: pandas DataFrame with columns [ts_id, ts_ind, minute, variable, value]
            N: Total number of time series
            train_ind: List of training time series INDICES (not IDs, for normalization)
            var_to_ind: Dictionary mapping variable names to indices
            variables: Sorted list of variable names
        """
        V = len(variables)
        self.args.V = V
        self.logger.write(f'# TS variables: {V}')
        
        # Use hourly aggregation for manageable sequence length
        data = data.copy()
        data['timestep'] = data['minute'].apply(lambda x: max(0, int(x // 60)))
        T = data.timestep.max() + 1
        
        self.args.T = T
        self.logger.write(f'# timesteps (Raindrop): {T}')
        
        # Initialize arrays [N, T, V]
        values = np.zeros((N, T, V), dtype=np.float32)
        mask = np.zeros((N, T, V), dtype=np.float32)
        
        # Fill arrays
        for row in data.itertuples():
            vind = var_to_ind[row.variable]
            tstep = row.timestep
            if tstep < T:
                values[row.ts_ind, tstep, vind] = row.value
                mask[row.ts_ind, tstep, vind] = 1
        
        # Compute sequence lengths (number of timesteps with at least one observation)
        lengths = np.zeros(N, dtype=np.int64)
        for i in range(N):
            obs_times = np.where(mask[i].sum(axis=1) > 0)[0]
            if len(obs_times) > 0:
                lengths[i] = obs_times.max() + 1
            else:
                lengths[i] = 1
        
        # Z-score normalization using training statistics
        values = self.normalize_zscore_array(values, mask, train_ind)
        
        # Create time array (normalized timestamps)
        time_arr = np.zeros((N, T), dtype=np.float32)
        for i in range(N):
            if lengths[i] > 1:
                time_arr[i, :] = np.arange(T) / (lengths[i] - 1)
        
        # Store prepared data
        self.values = values   # [N, T, V]
        self.mask = mask       # [N, T, V]
        self.times = time_arr  # [N, T]
        self.lengths = lengths # [N]
    
    def get_batch(self, ind):
        """
        Get batch for Raindrop model.
        
        Args:
            ind: Array of indices for the batch
        
        Returns:
            dict with keys:
                - src: [T, B, 2*V] - values and mask concatenated
                - static: [B, D] - static/demographic features
                - times: [T, B] - timestamps
                - lengths: [B] - sequence lengths
                - labels: [B] - binary labels
        """
        # Get data for batch indices
        values = self.values[ind]   # [B, T, V]
        mask = self.mask[ind]       # [B, T, V]
        times = self.times[ind]     # [B, T]
        lengths = self.lengths[ind] # [B]
        
        # Concatenate values and mask: [B, T, 2*V]
        src = np.concatenate([values, mask], axis=-1)
        
        # Transpose to Raindrop format: [T, B, 2*V]
        src = src.transpose(1, 0, 2)
        
        # Transpose times: [T, B]
        times = times.transpose(1, 0)
        
        return {
            'src': torch.FloatTensor(src),
            'static': torch.FloatTensor(self.demo[ind]),
            'times': torch.FloatTensor(times),
            'lengths': torch.LongTensor(lengths),
            'labels': torch.FloatTensor(self.y[ind])
        }
