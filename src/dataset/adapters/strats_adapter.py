"""
Data adapter for STraTS and iSTraTS models.
"""
import torch
import numpy as np
from ..base_adapter import BaseDataAdapter


class StratsAdapter(BaseDataAdapter):
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        # Trim to max observations
        data = data.sample(frac=1)
        data = data.groupby('ts_id').head(self.args.max_obs)
        
        # Convert train indices to train IDs for normalization
        train_ids = data[data.ts_ind.isin(train_ind)]['ts_id'].unique()
        
        # Z-score normalization using training data
        data = self.normalize_zscore(data, train_ids)
        
        V = len(var_to_ind)
        self.args.V = V
        self.logger.write(f'# TS variables: {V}')
        
        # Store max minute for time normalization
        max_minute = data['minute'].max()
        self.max_minute = max_minute
        
        # Normalize time to [-1, 1]
        data['minute'] = data['minute'] / max_minute * 2 - 1
        
        # Initialize containers for each time series
        values = [[] for _ in range(N)]
        times = [[] for _ in range(N)]
        varis = [[] for _ in range(N)]
        
        # Collect observations for each time series
        for row in data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(var_to_ind[row.variable])
        
        self.values = values
        self.times = times
        self.varis = varis
    
    def get_batch(self, ind):
        """
        Get batch for STraTS/iSTraTS models.
        
        Args:
            ind: Array of indices for the batch
        
        Returns:
            dict with keys:
                - values: [B, max_obs] observation values
                - times: [B, max_obs] observation times
                - varis: [B, max_obs] variable indices
                - obs_mask: [B, max_obs] observation mask
                - demo: [B, D] demographics
                - labels: [B] labels
        """
        # Pad to maximum observations in batch
        values_batch = [self.values[i] for i in ind]
        times_batch = [self.times[i] for i in ind]
        varis_batch = [self.varis[i] for i in ind]
        
        num_obs = [len(v) for v in values_batch]
        max_obs = max(num_obs)
        pad_lens = max_obs - np.array(num_obs)
        
        # Pad sequences
        values_padded = [v + [0] * int(pad_len) for v, pad_len in zip(values_batch, pad_lens)]
        times_padded = [t + [0] * int(pad_len) for t, pad_len in zip(times_batch, pad_lens)]
        varis_padded = [v + [0] * int(pad_len) for v, pad_len in zip(varis_batch, pad_lens)]
        
        # Create observation mask
        obs_mask = [[1] * n + [0] * int(pad_len) for n, pad_len in zip(num_obs, pad_lens)]
        
        return {
            'values': torch.FloatTensor(values_padded),
            'times': torch.FloatTensor(times_padded),
            'varis': torch.IntTensor(varis_padded),
            'obs_mask': torch.IntTensor(obs_mask),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }

