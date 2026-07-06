"""
Data adapter for Neural CDE model.
"""

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from base_adapter import BaseDataAdapter


class CDEAdapter(BaseDataAdapter):
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        V = len(variables)
        self.args.V = V
        self.logger.write(f'# TS variables (CDE): {V}')

        if self.args.dataset == 'mimic_iii':
            history = 24
        elif self.args.dataset == 'physionet_2019':
            history = 60
        else:
            history = 48

        data = data.groupby('ts_id').head(self.args.max_obs).copy()
        data['hour'] = data['minute'].apply(lambda x: max(0, int(x // 60))).clip(0, history - 1)
        self.args.T = history
        self.logger.write(f'# timesteps (CDE): {history}')

        train_ids = data[data.ts_ind.isin(train_ind)]['ts_id'].unique()
        data = self.normalize_zscore(data, train_ids)

        values = np.zeros((N, history, V), dtype=np.float32)
        mask = np.zeros((N, history, V), dtype=np.float32)

        for row in data.itertuples():
            vind = var_to_ind[row.variable]
            tstep = row.hour
            values[row.ts_ind, tstep, vind] = row.value
            mask[row.ts_ind, tstep, vind] = 1.0

        lengths = np.zeros(N, dtype=np.int64)
        for i in range(N):
            obs_times = np.where(mask[i].sum(axis=1) > 0)[0]
            lengths[i] = obs_times.max() + 1 if len(obs_times) > 0 else 1

        time = np.tile(
            np.linspace(0.0, 1.0, history, dtype=np.float32).reshape(1, history),
            (N, 1),
        )

        self.values = values
        self.mask = mask
        self.time = time
        self.lengths = lengths

        if self.args.branch2_enable:
            self.x_t, self.m_t, self.seq_len = self.prepare_data_for_branch2(
                data, N, var_to_ind
            )

    def get_batch(self, ind):
        batch = {
            'x': torch.FloatTensor(self.values[ind]),
            'mask': torch.FloatTensor(self.mask[ind]),
            't': torch.FloatTensor(self.time[ind]),
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
