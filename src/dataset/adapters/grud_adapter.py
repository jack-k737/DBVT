"""Data adapter for GRU-D model."""
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from ..base_adapter import BaseDataAdapter
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from base_adapter import BaseDataAdapter


class GRUDAdapter(BaseDataAdapter):
    """Prepare dense hourly inputs for GRU-D."""

    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        V = len(variables)
        self.args.V = V
        if self.logger is not None:
            self.logger.write(f'# TS variables (GRUD): {V}')

        data = data.copy()
        data['hour'] = data['minute'].apply(lambda x: max(0, int(x // 60)))
        T = int(data['hour'].max()) + 1
        self.args.T = T
        if self.logger is not None:
            self.logger.write(f'# time steps (GRUD): {T}')

        train_ids = data.loc[data.ts_ind.isin(train_ind), 'ts_id'].unique()
        data = self.normalize_zscore(data, train_ids)

        self.x_t, self.m_t, self.seq_len = self.prepare_data_for_branch2(data, N, var_to_ind)
        self.delta_t = self.compute_delta_time(self.m_t, self.x_t.shape[1]).astype(np.float32)

        if self.args.branch2_enable:
            T = self.x_t.shape[1]
            self.observed_data = self.x_t
            self.observed_mask = self.m_t
            self.observed_tp = np.tile(
                np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(1, T),
                (N, 1),
            )
            self.tau = self.delta_t

    def get_batch(self, ind):
        ind = np.asarray(ind)
        if ind.ndim == 0:
            ind = ind[None]

        batch = {
            'x_t': torch.FloatTensor(self.x_t[ind]),
            'm_t': torch.FloatTensor(self.m_t[ind]),
            'delta_t': torch.FloatTensor(self.delta_t[ind]),
            'seq_len': torch.LongTensor(self.seq_len[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind]),
        }

        if self.args.branch2_enable:
            batch.update({
                'observed_data': torch.FloatTensor(self.observed_data[ind]),
                'observed_mask': torch.FloatTensor(self.observed_mask[ind]),
                'observed_tp': torch.FloatTensor(self.observed_tp[ind]),
                'tau': torch.FloatTensor(self.tau[ind]),
            })

        return batch
