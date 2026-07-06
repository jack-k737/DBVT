"""
Data adapter for Raindrop model.
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

        data = data.copy()
        data['timestep'] = data['minute'].apply(lambda x: max(0, int(x // 60)))
        T = int(data['timestep'].max()) + 1

        self.args.T = T
        self.logger.write(f'# timesteps (Raindrop): {T}')

        train_ids = data.loc[data.ts_ind.isin(train_ind), 'ts_id'].unique()
        data = self.normalize_zscore(data, train_ids)

        values = np.zeros((N, T, V), dtype=np.float32)
        mask = np.zeros((N, T, V), dtype=np.float32)

        for row in data.itertuples():
            vind = var_to_ind[row.variable]
            tstep = row.timestep
            if tstep < T:
                values[row.ts_ind, tstep, vind] = row.value
                mask[row.ts_ind, tstep, vind] = 1.0

        lengths = np.zeros(N, dtype=np.int64)
        for i in range(N):
            obs_times = np.where(mask[i].sum(axis=1) > 0)[0]
            if len(obs_times) > 0:
                lengths[i] = obs_times.max() + 1
            else:
                lengths[i] = 1

        time_arr = np.zeros((N, T), dtype=np.float32)
        for t in range(T):
            time_arr[:, t] = t / max(T - 1, 1)

        self.values = values
        self.mask = mask
        self.times = time_arr
        self.lengths = lengths

        if self.args.branch2_enable:
            self.x_t, self.m_t, self.seq_len = self.prepare_data_for_branch2(
                data, N, var_to_ind
            )

    def get_batch(self, ind):
        """
        Get batch for Raindrop model.

        Args:
            ind: Array of indices for the batch

        Returns:
            dict with keys:
                - src: [T, B, 2*V] - values and mask concatenated
                - demo: [B, D] - static/demographic features
                - times: [T, B] - timestamps
                - lengths: [B] - sequence lengths
                - labels: [B] - binary labels
        """
        values = self.values[ind]
        mask = self.mask[ind]
        times = self.times[ind]
        lengths = self.lengths[ind]

        src = np.concatenate([values, mask], axis=-1)
        src = src.transpose(1, 0, 2)
        times = times.transpose(1, 0)

        batch = {
            'src': torch.FloatTensor(src),
            'demo': torch.FloatTensor(self.demo[ind]),
            'times': torch.FloatTensor(times),
            'lengths': torch.LongTensor(lengths),
            'labels': torch.FloatTensor(self.y[ind]),
        }

        if self.args.branch2_enable:
            batch.update({
                'x_t': torch.FloatTensor(self.x_t[ind]),
                'm_t': torch.FloatTensor(self.m_t[ind]),
                'seq_len': torch.LongTensor(self.seq_len[ind]),
            })

        return batch
