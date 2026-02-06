"""
Data adapter for MTM model.
"""
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from ..base_adapter import BaseDataAdapter


class MTMAdapter(BaseDataAdapter):
    """
    Data adapter for MTM (Multi-Scale Token Mixing Transformer) model.
    
    MTM expects compact format with variable-length sequences:
        - x: [B, T] - padded values
        - x_mask: [B, T, C] - observation mask
        - t: [B, T] - timestamps (integer indices)
        - demo: [B, D] - static features
    """
    
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        """
        Prepare data for MTM model in compact format.
        
        Args:
            data: pandas DataFrame with columns [ts_id, ts_ind, minute, variable, value]
            N: Total number of time series
            train_ind: List of training time series INDICES
            var_to_ind: Dictionary mapping variable names to indices
            variables: Sorted list of variable names
        """
        # Trim to max observations
        data = data.sample(frac=1)
        data = data.groupby('ts_id').head(self.args.max_obs)
        
        # Get train IDs for normalization
        train_ids = data[data.ts_ind.isin(train_ind)]['ts_id'].unique()
        
        # Z-score normalization using training data
        data = self.normalize_zscore(data, train_ids)
        
        V = len(var_to_ind)
        self.args.V = V
        self.logger.write(f'# TS variables: {V}')
        
        # Store max minute for time normalization
        max_minute = data['minute'].max()
        self.max_minute = max_minute
        
        # Create hourly timesteps (for reasonable sequence length)
        data['timestep'] = data['minute'].apply(lambda x: max(0, int(x // 60)))
        
        # Initialize containers for compact format
        # We'll store lists of observations per time series
        self.ts_data = [[] for _ in range(N)]
        
        # Collect observations grouped by (ts_ind, timestep)
        for row in data.itertuples():
            ts_ind = row.ts_ind
            timestep = row.timestep
            var_idx = var_to_ind[row.variable]
            value = row.value
            
            self.ts_data[ts_ind].append({
                'timestep': timestep,
                'var': var_idx,
                'value': value
            })
        
        # Convert to compact representation per time series
        self.compact_data = []
        for i in range(N):
            if len(self.ts_data[i]) == 0:
                # Empty time series - add dummy observation
                self.compact_data.append({
                    'values': np.array([0.0], dtype=np.float32),
                    'times': np.array([0], dtype=np.int64),
                    'vars': np.array([0], dtype=np.int64)
                })
                continue
            
            # Group by timestep
            timesteps = {}
            for obs in self.ts_data[i]:
                t = obs['timestep']
                if t not in timesteps:
                    timesteps[t] = {'vars': [], 'values': []}
                timesteps[t]['vars'].append(obs['var'])
                timesteps[t]['values'].append(obs['value'])
            
            # Create compact arrays
            sorted_times = sorted(timesteps.keys())
            values = []
            times = []
            vars_list = []
            
            for t in sorted_times:
                for v, val in zip(timesteps[t]['vars'], timesteps[t]['values']):
                    values.append(val)
                    times.append(t)
                    vars_list.append(v)
            
            self.compact_data.append({
                'values': np.array(values, dtype=np.float32),
                'times': np.array(times, dtype=np.int64),
                'vars': np.array(vars_list, dtype=np.int64)
            })
    
    def get_batch(self, ind):
        """
        Get batch for MTM model.
        
        Follows original MTM implementation by removing timesteps where all variables are missing.
        
        Args:
            ind: Array of indices for the batch
            
        Returns:
            dict with keys:
                - x: [B, T, C] - values (NaN for missing, padded)
                - x_mask: [B, T, C] - observation mask (True = missing, False = observed)
                - t: [B, T] - timestamps (padded with -1)
                - demo: [B, D] - static/demographic features
                - labels: [B] - binary labels
        """
        batch_data = [self.compact_data[i] for i in ind]
        V = self.args.V
        B = len(ind)
        
        # First, create dense arrays for each sample, then remove fully missing timesteps
        processed_xs = []
        processed_ts = []
        
        for data in batch_data:
            # Find unique timesteps for this sample
            unique_times = sorted(np.unique(data['times']).tolist())
            if len(unique_times) == 0:
                # Empty sample - use dummy
                processed_xs.append(torch.tensor(np.zeros((1, V), dtype=np.float32), dtype=torch.float32))
                processed_ts.append(torch.tensor([0], dtype=torch.int64))
                continue
            
            # Create dense array [T, V]
            T_sample = len(unique_times)
            x_sample = np.full((T_sample, V), np.nan, dtype=np.float32)
            
            # Fill in observations
            for val, t, v in zip(data['values'], data['times'], data['vars']):
                t_idx = unique_times.index(t)
                x_sample[t_idx, v] = val
            
            # Remove timesteps where all variables are missing (like original MTM)
            mask = np.isnan(x_sample)  # [T, V], True = missing
            t_mask = mask.all(axis=1)  # [T], True if all vars missing at this timestep
            valid_mask = ~t_mask  # [T], True if at least one var observed
            
            # Only keep timesteps with at least one observation
            if valid_mask.sum() > 0:
                x_sample_clean = x_sample[valid_mask, :]  # [T_valid, V]
                t_sample_clean = np.array([unique_times[i] for i in range(len(unique_times)) if valid_mask[i]], dtype=np.int64)
            else:
                # All timesteps were fully missing - keep at least one dummy timestep
                x_sample_clean = np.zeros((1, V), dtype=np.float32)
                t_sample_clean = np.array([unique_times[0] if len(unique_times) > 0 else 0], dtype=np.int64)
            
            processed_xs.append(torch.tensor(x_sample_clean, dtype=torch.float32))
            processed_ts.append(torch.tensor(t_sample_clean, dtype=torch.int64))
        
        # Pad sequences to same length (like original MTM collate_compact)
        # Use pad_sequence: padding_value=torch.nan for x, padding_value=-1 for t
        xs_padded = pad_sequence(processed_xs, batch_first=True, padding_value=float('nan'))
        ts_padded = pad_sequence(processed_ts, batch_first=True, padding_value=-1)
        
        # Create mask (True = missing, False = observed)
        x_mask = torch.isnan(xs_padded)
        
        # Replace NaN with 0 for input (model expects this)
        x = torch.nan_to_num(xs_padded, nan=0.0)
        
        return {
            'x': x,  # [B, T, V]
            'x_mask': x_mask,  # [B, T, V]
            't': ts_padded,  # [B, T]
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }
