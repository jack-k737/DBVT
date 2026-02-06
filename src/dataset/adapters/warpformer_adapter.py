"""
Data adapter for Warpformer model.
"""
import torch
import numpy as np
from ..base_adapter import BaseDataAdapter


class WarpformerAdapter(BaseDataAdapter):
    """
    Data adapter for Warpformer model.
    
    Warpformer uses time warping and attention mechanisms for irregular time series.
    Expects:
        - observed_data: [B, L, K] - normalized observation values
        - observed_mask: [B, L, K] - observation mask
        - observed_tp: [B, L] - timestamps (normalized)
        - tau: [B, L, K] - time since last observation for each variable
        - demo: [B, D] - static features
    """
    
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        V = len(variables)
        self.args.V = V
        self.logger.write(f'# TS variables (Warpformer): {V}')
        
        # Convert minute to hour for hourly aggregation
        data = data.copy()
        data['hour'] = data['minute'].apply(lambda x: max(0, int(x // 60)))
        T = data.hour.max() + 1
        self.args.T = T
        self.logger.write(f'# timesteps (Warpformer): {T}')
        
        # Initialize arrays [N, T, V]
        observed_data = np.zeros((N, T, V), dtype=np.float32)
        observed_mask = np.zeros((N, T, V), dtype=np.float32)
        
        # Fill arrays
        for row in data.itertuples():
            vind = var_to_ind[row.variable]
            tstep = row.hour
            if tstep < T:
                observed_data[row.ts_ind, tstep, vind] = row.value
                observed_mask[row.ts_ind, tstep, vind] = 1
        
        # Z-score normalization using training statistics
        observed_data = self.normalize_zscore_array(observed_data, observed_mask, train_ind)
        
        # Create timestamp array (normalized to [0, 1])
        observed_tp = np.zeros((N, T), dtype=np.float32)
        for t in range(T):
            observed_tp[:, t] = t / max(T - 1, 1)
        
        # Compute tau (time since last observation for each variable)
        tau = self._compute_tau(observed_mask, N, T, V)
        
        # Compute sequence lengths (number of timesteps with at least one observation)
        lengths = np.zeros(N, dtype=np.int64)
        for i in range(N):
            obs_times = np.where(observed_mask[i].sum(axis=1) > 0)[0]
            if len(obs_times) > 0:
                lengths[i] = obs_times.max() + 1
            else:
                lengths[i] = 1
        
        # Store prepared data
        self.data = observed_data      # [N, T, V]
        self.mask = observed_mask      # [N, T, V]
        self.tp = observed_tp          # [N, T]
        self.tau = tau                 # [N, T, V]
        self.lengths = lengths         # [N]
    
    def _compute_tau(self, mask, N, T, V):
        """
        Compute time since last observation for each variable.
        
        For each position (i, t, v), tau gives the number of timesteps
        since the last observed value of variable v up to time t.
        """
        tau = np.zeros((N, T, V), dtype=np.float32)
        
        for i in range(N):
            for v in range(V):
                last_obs_time = -1
                for t in range(T):
                    if mask[i, t, v] > 0:
                        if last_obs_time >= 0:
                            tau[i, t, v] = t - last_obs_time
                        else:
                            tau[i, t, v] = 0
                        last_obs_time = t
                    else:
                        if last_obs_time >= 0:
                            tau[i, t, v] = t - last_obs_time
                        else:
                            tau[i, t, v] = t  # No observation yet
        
        # Normalize tau to reasonable range
        tau = tau / max(T, 1)
        return tau
    
    def get_batch(self, ind):
        """
        Get batch for Warpformer model.
        
        Args:
            ind: Array of indices for the batch
        
        Returns:
            dict with keys:
                - observed_data: [B, L, K] - normalized observation values
                - observed_mask: [B, L, K] - binary mask indicating observed values
                - observed_tp: [B, L] - timestamps (normalized)
                - tau: [B, L, K] - time since last observation for each variable
                - demo: [B, D] - static/demographic features
                - labels: [B] - binary labels
        """
        return {
            'observed_data': torch.FloatTensor(self.data[ind]),
            'observed_mask': torch.FloatTensor(self.mask[ind]),
            'observed_tp': torch.FloatTensor(self.tp[ind]),
            'tau': torch.FloatTensor(self.tau[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }
