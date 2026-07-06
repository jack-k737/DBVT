"""
Data adapter for the naive Transformer model.
"""
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from ..base_adapter import BaseDataAdapter
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from base_adapter import BaseDataAdapter


class TransformerAdapter(BaseDataAdapter):
    """
    Data adapter for Transformer-style irregular time series inputs.

    Expects:
        - observed_data: [B, T, V] - normalized observation values
        - observed_mask: [B, T, V] - binary observation mask
        - observed_tp: [B, T] - normalized timestamps
        - tau: [B, T, V] - time since last observation for each variable
        - demo: [B, D] - static features
    """

    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        V = len(variables)
        self.args.V = V
        self.logger.write(f'# TS variables (Transformer): {V}')

        data = data.copy()
        data['hour'] = data['minute'].apply(lambda x: max(0, int(x // 60)))
        T = int(data['hour'].max()) + 1
        self.args.T = T
        self.logger.write(f'# timesteps (Transformer): {T}')

        train_ids = data.loc[data.ts_ind.isin(train_ind), 'ts_id'].unique()
        data = self.normalize_zscore(data, train_ids)

        observed_data = np.zeros((N, T, V), dtype=np.float32)
        observed_mask = np.zeros((N, T, V), dtype=np.float32)

        for row in data.itertuples():
            vind = var_to_ind[row.variable]
            tstep = row.hour
            if tstep < T:
                observed_data[row.ts_ind, tstep, vind] = row.value
                observed_mask[row.ts_ind, tstep, vind] = 1.0

        observed_tp = np.zeros((N, T), dtype=np.float32)
        for t in range(T):
            observed_tp[:, t] = t / max(T - 1, 1)

        self.observed_data = observed_data
        self.observed_mask = observed_mask
        self.observed_tp = observed_tp
        self.tau = self._compute_tau(observed_mask, N, T, V)

        if self.args.branch2_enable:
            self.x_t, self.m_t, self.seq_len = self.prepare_data_for_branch2(
                data, N, var_to_ind
            )

    def _compute_tau(self, mask, N, T, V):
        """Compute normalized time since last observation for each variable."""
        tau = np.zeros((N, T, V), dtype=np.float32)

        for i in range(N):
            for v in range(V):
                last_obs_time = -1
                for t in range(T):
                    if mask[i, t, v] > 0:
                        tau[i, t, v] = 0 if last_obs_time < 0 else t - last_obs_time
                        last_obs_time = t
                    else:
                        tau[i, t, v] = t if last_obs_time < 0 else t - last_obs_time

        return tau / max(T, 1)

    def get_batch(self, ind):
        batch = {
            'observed_data': torch.FloatTensor(self.observed_data[ind]),
            'observed_mask': torch.FloatTensor(self.observed_mask[ind]),
            'observed_tp': torch.FloatTensor(self.observed_tp[ind]),
            'tau': torch.FloatTensor(self.tau[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind]),
        }

        if self.args.branch2_enable:
            batch.update({
                'x_t': torch.FloatTensor(self.x_t[ind]),
                'm_t': torch.FloatTensor(self.m_t[ind]),
                'seq_len': torch.LongTensor(self.seq_len[ind]),
            })

        return batch
