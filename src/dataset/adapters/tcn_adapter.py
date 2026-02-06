"""
Data adapter for TCN and SAND models.

Both models use hourly aggregated time series with format [N, T, V*3]:
- values: normalized observation values
- obs: observation mask
- delta: time since last observation
"""
import torch
import numpy as np
from ..base_adapter import BaseDataAdapter


class TCNAdapter(BaseDataAdapter):
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        V = len(variables)
        self.args.V = V
        self.logger.write(f'# TS variables (TCN/SAND): {V}')
        
        # Convert minute to hour for hourly aggregation
        data = data.copy()
        data['hour'] = data['minute'].apply(lambda x: max(0, int(x // 60)))
        T = data.hour.max() + 1
        self.args.T = T
        self.logger.write(f'# timesteps (hours): {T}')
        
        # Initialize arrays [N, T, V]
        values = np.zeros((N, T, V), dtype=np.float32)
        obs = np.zeros((N, T, V), dtype=np.float32)
        
        # Fill arrays
        for row in data.itertuples():
            vind = var_to_ind[row.variable]
            tstep = row.hour
            if tstep < T:
                values[row.ts_ind, tstep, vind] = row.value
                obs[row.ts_ind, tstep, vind] = 1
        
        # Generate delta (time since last observation)
        delta = np.zeros((N, T, V), dtype=np.float32)
        delta[:, 0, :] = obs[:, 0, :]
        for t in range(1, T):
            delta[:, t, :] = obs[:, t, :] * 0 + (1 - obs[:, t, :]) * (1 + delta[:, t-1, :])
        delta = delta / T  # Normalize
        
        # Mean fill missing observations using training statistics
        train_means = (values[train_ind] * obs[train_ind]).sum(axis=(0, 1)) / (obs[train_ind].sum(axis=(0, 1)) + 1e-8)
        values = values * obs + (1 - obs) * train_means.reshape((1, 1, V))
        
        # Z-score normalization using training statistics
        means = values[train_ind].mean(axis=(0, 1), keepdims=True)
        stds = values[train_ind].std(axis=(0, 1), keepdims=True)
        stds = np.where(stds == 0, 1, stds)
        values = (values - means) / stds
        
        # Concatenate to form X: [N, T, V*3]
        self.X = np.concatenate((values, obs, delta), axis=-1).astype(np.float32)
        self.logger.write(f'Input shape: [N, {T}, {V*3}]')
    
    def get_batch(self, ind):
        """
        Get batch for TCN/SAND models.
        
        Args:
            ind: Array of indices for the batch
        
        Returns:
            dict with keys:
                - ts: [B, T, V*3] concatenation of (values, obs_mask, delta)
                - demo: [B, D] static/demographic features
                - labels: [B] binary labels
        """
        return {
            'ts': torch.FloatTensor(self.X[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }
