"""Data adapter for Strats model."""
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from ..base_adapter import BaseDataAdapter
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from base_adapter import BaseDataAdapter


DATASET_HISTORY_MINUTES = {
    'mimic_iii': 24 * 60,
    'physionet_2012': 48 * 60,
    'physionet_2019': 60 * 60,
}


class StratsAdapter(BaseDataAdapter):
    """Prepare event-level padded triplets for Strats."""

    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        V = len(variables)
        self.args.V = V
        history_minutes = DATASET_HISTORY_MINUTES.get(self.args.dataset, 48 * 60)
        self.args.T = history_minutes // 60
        if self.logger is not None:
            self.logger.write(f'# TS variables (Strats): {V}')
            self.logger.write(f'# max observations (Strats): {self.args.max_obs}')

        data = data.sort_values(['ts_ind', 'minute']).groupby('ts_id', sort=False).head(self.args.max_obs).copy()
        train_ids = data.loc[data.ts_ind.isin(train_ind), 'ts_id'].unique()
        data = self.normalize_zscore(data, train_ids)
        data['var_ind'] = data['variable'].map(var_to_ind).astype(np.int64)
        data['t_norm'] = (data['minute'].astype(np.float32) / float(history_minutes)).clip(0.0, 1.0)

        self.series = [
            {
                'values': [],
                'times': [],
                'varis': [],
            }
            for _ in range(N)
        ]

        for row in data.itertuples():
            record = self.series[row.ts_ind]
            record['values'].append(float(row.value))
            record['times'].append(float(row.t_norm))
            record['varis'].append(int(row.var_ind))

        for i in range(N):
            record = self.series[i]
            if not record['values']:
                record['values'] = [0.0]
                record['times'] = [0.0]
                record['varis'] = [0]
            record['values'] = np.asarray(record['values'], dtype=np.float32)
            record['times'] = np.asarray(record['times'], dtype=np.float32)
            record['varis'] = np.asarray(record['varis'], dtype=np.int64)

        if self.args.branch2_enable:
            self.x_t, self.m_t, self.seq_len = self.prepare_data_for_branch2(data, N, var_to_ind)

    def get_batch(self, ind):
        batch_data = [self.series[i] for i in ind]
        lengths = np.asarray([len(item['values']) for item in batch_data], dtype=np.int64)
        max_len = int(lengths.max())
        batch_size = len(batch_data)

        values = np.zeros((batch_size, max_len), dtype=np.float32)
        times = np.zeros((batch_size, max_len), dtype=np.float32)
        varis = np.zeros((batch_size, max_len), dtype=np.int64)
        obs_mask = np.zeros((batch_size, max_len), dtype=np.float32)

        for i, item in enumerate(batch_data):
            seq_len = lengths[i]
            values[i, :seq_len] = item['values']
            times[i, :seq_len] = item['times']
            varis[i, :seq_len] = item['varis']
            obs_mask[i, :seq_len] = 1.0

        batch = {
            'values': torch.FloatTensor(values),
            'times': torch.FloatTensor(times),
            'varis': torch.LongTensor(varis),
            'obs_mask': torch.FloatTensor(obs_mask),
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
