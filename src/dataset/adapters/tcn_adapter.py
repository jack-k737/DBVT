from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:
    from ..base_adapter import BaseDataAdapter
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from base_adapter import BaseDataAdapter


class TCNAdapter(BaseDataAdapter):
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        V = len(variables)
        self.args.V = V
        self.logger.write(f'# TS variables (TCN): {V}')
        data = data.copy()
        data['hour'] = data['minute'].apply(lambda x: max(0, int(x // 60)))
        T = int(data.hour.max()) + 1
        self.args.T = T
        self.logger.write(f'# timesteps (hours): {T}')

        values = np.zeros((N, T, V), dtype=np.float32)
        obs = np.zeros((N, T, V), dtype=np.float32)
        for row in data.itertuples():
            vind = var_to_ind[row.variable]
            tstep = row.hour
            if tstep < T:
                values[row.ts_ind, tstep, vind] = row.value
                obs[row.ts_ind, tstep, vind] = 1.0

        delta = self.compute_delta_time(obs, T).astype(np.float32)
        values = self.mean_fill_missing(values, obs, train_ind)
        means = values[train_ind].mean(axis=(0, 1), keepdims=True)
        stds = values[train_ind].std(axis=(0, 1), keepdims=True)
        stds = np.where(stds == 0, 1, stds)
        values = (values - means) / stds
        self.X = np.concatenate((values, obs, delta), axis=-1).astype(np.float32)
        self.logger.write(f'Input shape: [N, {T}, {V * 3}]')

    def get_batch(self, ind):
        return {
            'ts': torch.FloatTensor(self.X[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind]),
        }
