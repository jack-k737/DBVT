import sys
from pathlib import Path

import numpy as np
import torch

try:
    from ..base_adapter import BaseDataAdapter
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from base_adapter import BaseDataAdapter


class GRUAdapter(BaseDataAdapter):
    """
    Data adapter for the standalone GRU branch inputs.

    Expects:
        - x_t: [B, T, V] - normalized observation values
        - m_t: [B, T, V] - binary observation mask
        - seq_len: [B] - effective sequence lengths
        - demo: [B, D] - static features
    """

    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        V = len(variables)
        self.args.V = V
        if self.logger is not None:
            self.logger.write(f"# TS variables (GRU): {V}")

        data = data.copy()
        data["hour"] = data["minute"].apply(lambda x: max(0, int(x // 60)))
        T = int(data["hour"].max()) + 1
        self.args.T = T
        if self.logger is not None:
            self.logger.write(f"# time steps (GRU): {T}")

        train_ids = data.loc[data.ts_ind.isin(train_ind), "ts_id"].unique()
        data = self.normalize_zscore(data, train_ids)

        self.x_t, self.m_t, self.seq_len = self.prepare_data_for_branch2(
            data, N, var_to_ind
        )

        if self.args.branch2_enable:
            T = self.x_t.shape[1]
            self.observed_data = self.x_t
            self.observed_mask = self.m_t
            self.observed_tp = np.tile(
                np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(1, T),
                (N, 1),
            )
            self.tau = self._compute_tau(self.observed_mask, N, T, V)

    def _compute_tau(self, mask, N, T, V):
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
        ind = np.asarray(ind)
        if ind.ndim == 0:
            ind = ind[None]

        batch = {
            'x_t': torch.FloatTensor(self.x_t[ind]),
            'm_t': torch.FloatTensor(self.m_t[ind]),
            'seq_len': torch.LongTensor(self.seq_len[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }

        if self.args.branch2_enable:
            batch.update({
                'observed_data': torch.FloatTensor(self.observed_data[ind]),
                'observed_mask': torch.FloatTensor(self.observed_mask[ind]),
                'observed_tp': torch.FloatTensor(self.observed_tp[ind]),
                'tau': torch.FloatTensor(self.tau[ind]),
            })

        return batch
