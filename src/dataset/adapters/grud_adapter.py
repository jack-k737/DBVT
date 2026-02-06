"""
Data adapter for GRU-D model.
Uses hourly aggregation similar to DBVT for stable fixed-size arrays.
"""
import torch
import numpy as np
from ..base_adapter import BaseDataAdapter


class GRUDAdapter(BaseDataAdapter):
    
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        V = len(variables)
        self.args.V = V
        self.logger.write(f'[GRU-D] # TS variables: {V}')
        
        # Convert minute to hour for hourly aggregation
        data = data.copy()
        data['hour'] = data['minute'].apply(lambda x: max(0, int(x // 60)))
        T = data.hour.max() + 1
        self.args.T = T
        self.logger.write(f'[GRU-D] # timesteps (hourly): {T}')
        
        # ========== Create fixed-size arrays [N, T, V] ==========
        observed_data = np.zeros((N, T, V), dtype=np.float32)
        observed_mask = np.zeros((N, T, V), dtype=np.float32)
        
        for row in data.itertuples():
            vind = var_to_ind[row.variable]
            tstep = row.hour
            if tstep < T:
                observed_data[row.ts_ind, tstep, vind] = row.value
                observed_mask[row.ts_ind, tstep, vind] = 1
        
        # Z-score normalization
        observed_data = self.normalize_zscore_array(observed_data, observed_mask, train_ind)
        
        # ========== Compute delta_t (time since last observation) ==========
        delta_t = self._compute_delta_t(observed_mask, N, T, V)
        
        # ========== Compute sequence lengths ==========
        lengths = np.zeros(N, dtype=np.int64)
        for i in range(N):
            obs_times = np.where(observed_mask[i].sum(axis=1) > 0)[0]
            if len(obs_times) > 0:
                lengths[i] = obs_times.max() + 1
            else:
                lengths[i] = 1  # At least 1 timestep
        
        # Store data
        self.x_t = observed_data
        self.m_t = observed_mask
        self.delta_t = delta_t
        self.seq_len = lengths
        
    
    def _compute_delta_t(self, mask, N, T, V):
        delta = np.zeros((N, T, V), dtype=np.float32)
        
        for i in range(N):
            for v in range(V):
                last_obs = 0
                for t in range(T):
                    if t == 0:
                        delta[i, t, v] = 0
                    else:
                        delta[i, t, v] = t - last_obs
                    if mask[i, t, v] > 0:
                        last_obs = t
        
        # Normalize by T for numerical stability
        delta = delta / max(T, 1)
        return delta
    
    def get_batch(self, ind):
        return {
            'x_t': torch.FloatTensor(self.x_t[ind]),
            'm_t': torch.FloatTensor(self.m_t[ind]),
            'delta_t': torch.FloatTensor(self.delta_t[ind]),
            'seq_len': torch.LongTensor(self.seq_len[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }
