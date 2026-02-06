"""
Data adapter for HiPatch model.
"""
import torch
import numpy as np
from ..base_adapter import BaseDataAdapter


class HiPatchAdapter(BaseDataAdapter):
    
    def prepare_data(self, data, N, train_ind, var_to_ind, variables):
        V = len(variables)
        self.args.V = V
        self.logger.write(f'# TS variables: {V}')
        
        # Convert minute to hours
        data = data.copy()
        data['hour'] = data['minute'] / 60.0
        
        # Get dataset-specific parameters
        if self.args.dataset == 'mimic_iii':
            history = 24  # 24 hours
        elif self.args.dataset == 'physionet_2012':
            history = 48  # 48 hours
        elif self.args.dataset == 'physionet_2019':
            history = 60  # 60 hours
        else:
            history = 48  # default
        
        self.args.history = history
        
        # Patch parameters
        patch_size = getattr(self.args, 'patch_size', 6)  # hours
        stride = getattr(self.args, 'stride', 6)  # hours
        
        self.args.patch_size = patch_size
        self.args.stride = stride
        
        # Calculate number of patches
        npatch = int(np.ceil((history - patch_size) / stride)) + 1
        self.args.npatch = npatch
        
        # Calculate patch_layer (number of hierarchical levels)
        def layer_of_patches(n_patch):
            if n_patch == 1:
                return 1
            if n_patch % 2 == 0:
                return 1 + layer_of_patches(n_patch / 2)
            else:
                return layer_of_patches(n_patch + 1)
        
        self.args.patch_layer = layer_of_patches(npatch)
        self.logger.write(f'# patches: {npatch} | patch_layer: {self.args.patch_layer} | patch_size: {patch_size}h | stride: {stride}h')
        
        # Convert train indices to train IDs for normalization
        train_ids = data[data.ts_ind.isin(train_ind)]['ts_id'].unique()
        
        # Z-score normalization using training data
        data = self.normalize_zscore(data, train_ids)
        
        # Add variable index column for vectorized operations
        data['var_ind'] = data['variable'].map(var_to_ind)
        data['t_norm'] = data['hour'] / history
        
        # Add patch index column: determine which patch each observation belongs to
        data['patch_idx'] = ((data['hour'] / stride).astype(int)).clip(0, npatch - 1)
        
        # Fixed max observations per patch (cap at 20 for efficiency)
        L = 20
        self.logger.write(f'# max obs per patch: {L}')
        
        # Initialize arrays: [N, M, L, V]
        hipatch_values = np.zeros((N, npatch, L, V), dtype=np.float32)
        hipatch_mask = np.zeros((N, npatch, L, V), dtype=np.float32)
        hipatch_time = np.zeros((N, npatch, L, V), dtype=np.float32)
        
        # Convert to numpy arrays for faster access
        ts_inds = data['ts_ind'].values
        patch_idxs = data['patch_idx'].values
        var_inds = data['var_ind'].values.astype(int)
        values = data['value'].values.astype(np.float32)
        t_norms = data['t_norm'].values.astype(np.float32)
        
        # Track slot usage per (ts_ind, patch_idx, var_ind)
        slot_counter = np.zeros((N, npatch, V), dtype=np.int32)
        
        # Vectorized fill using numpy advanced indexing
        for i in range(len(ts_inds)):
            ts_i = ts_inds[i]
            p_i = patch_idxs[i]
            v_i = var_inds[i]
            slot = slot_counter[ts_i, p_i, v_i]
            if slot < L:
                hipatch_values[ts_i, p_i, slot, v_i] = values[i]
                hipatch_mask[ts_i, p_i, slot, v_i] = 1
                hipatch_time[ts_i, p_i, slot, v_i] = t_norms[i]
                slot_counter[ts_i, p_i, v_i] += 1
        
        # Store prepared data
        self.values = hipatch_values  # [N, M, L, V]
        self.mask = hipatch_mask      # [N, M, L, V]
        self.time = hipatch_time      # [N, M, L, V]
    
    def get_batch(self, ind):
        return {
            'X': torch.FloatTensor(self.values[ind]),
            'mask': torch.FloatTensor(self.mask[ind]),
            'time': torch.FloatTensor(self.time[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }
