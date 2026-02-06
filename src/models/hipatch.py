# -*- coding:utf-8 -*-
"""
Hi-Patch: Hierarchical Irregular Time Series Representation Learning via Patch-based Graph Structure
Adapted for the current framework.

Based on the original Hi-Patch implementation for classification tasks.
Note: This adaptation focuses on classification only, not forecasting.
"""

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


def softmax(src, index):
    """
    Custom softmax function for handling node-wise softmax over a graph structure.
    
    Args:
        src: Source tensor
        index: Index tensor for scatter operation
        
    Returns:
        Normalized attention weights
    """
    N = maybe_num_nodes(index)
    global_out = src - src.max()  # Global max normalization for numerical stability
    global_out = global_out.exp()
    global_out_sum = scatter(global_out, index, dim=0, dim_size=N, reduce='sum')[index]
    c = global_out / (global_out_sum + 1e-16)
    return c


class Intra_Inter_Patch_Graph_Layer(MessagePassing):
    """
    Implementation of intra/inter patch graph layer using message passing.
    
    This layer handles three types of edges:
    1. Same time, different variable (edge_same_time_diff_var)
    2. Different time, same variable (edge_diff_time_same_var)
    3. Different time, different variable (edge_diff_time_diff_var)
    """
    
    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, patch_layer=1, res=1, **kwargs):
        super(Intra_Inter_Patch_Graph_Layer, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.patch_layer = patch_layer
        self.res = res
        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_input // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha

        # Define parameters for query, key, and value transformations
        # 3 sets of weights for the 3 edge types
        self.w_k_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_k)) 
            for _ in range(self.n_heads)
        ])
        self.bias_k_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_k)) 
            for _ in range(self.n_heads)
        ])
        for param in self.w_k_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_k_list:
            nn.init.uniform_(param)

        self.w_q_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_q)) 
            for _ in range(self.n_heads)
        ])
        self.bias_q_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_q)) 
            for _ in range(self.n_heads)
        ])
        for param in self.w_q_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_q_list:
            nn.init.uniform_(param)

        self.w_v_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_e)) 
            for _ in range(self.n_heads)
        ])
        self.bias_v_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_e)) 
            for _ in range(self.n_heads)
        ])
        for param in self.w_v_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_v_list:
            nn.init.xavier_uniform_(param)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_input)
        
        # Output projection to ensure dimension matches for residual connection
        self.out_proj = nn.Linear(n_heads * self.d_e, d_input)

    def forward(self, x, edge_index, edge_value, time_nodes, 
                edge_same_time_diff_var, edge_diff_time_same_var, 
                edge_diff_time_diff_var, n_layer):
        residual = x
        x = self.layer_norm(x)
        return self.propagate(
            edge_index, x=x, edges_temporal=edge_value,
            edge_same_time_diff_var=edge_same_time_diff_var,
            edge_diff_time_same_var=edge_diff_time_same_var,
            edge_diff_time_diff_var=edge_diff_time_diff_var,
            n_layer=n_layer, residual=residual
        )

    def message(self, x_j, x_i, edge_index_i, edges_temporal, 
                edge_same_time_diff_var, edge_diff_time_same_var, 
                edge_diff_time_diff_var, n_layer):
        # Attention and message calculation for each attention head
        messages = []
        for i in range(self.n_heads):
            w_k = self.w_k_list[i][n_layer]
            bias_k = self.bias_k_list[i][n_layer]

            w_q = self.w_q_list[i][n_layer]
            bias_q = self.bias_q_list[i][n_layer]

            w_v = self.w_v_list[i][n_layer]
            bias_v = self.bias_v_list[i][n_layer]

            attention = self.each_head_attention(
                x_j, w_k, bias_k, w_q, bias_q, x_i,
                edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var
            )
            attention = torch.div(attention, self.d_sqrt)
            attention = torch.pow(self.alpha, torch.abs(edges_temporal.squeeze())).unsqueeze(-1) * attention
            attention_norm = softmax(attention, edge_index_i)

            sender_stdv = edge_same_time_diff_var * (torch.matmul(x_j, w_v[0]) + bias_v[0])
            sender_dtsv = edge_diff_time_same_var * (torch.matmul(x_j, w_v[1]) + bias_v[1])
            sender_dtdv = edge_diff_time_diff_var * (torch.matmul(x_j, w_v[2]) + bias_v[2])
            sender = sender_stdv + sender_dtsv + sender_dtdv

            message = attention_norm * sender
            messages.append(message)

        # Concatenate messages from all heads
        message_all_head = torch.cat(messages, 1)
        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k, bias_k, w_q, bias_q, x_i, 
                           edge_same_time_diff_var, edge_diff_time_same_var, 
                           edge_diff_time_diff_var):
        x_i_0 = edge_same_time_diff_var * (torch.matmul(x_i, w_q[0]) + bias_q[0])
        x_i_1 = edge_diff_time_same_var * (torch.matmul(x_i, w_q[1]) + bias_q[1])
        x_i_2 = edge_diff_time_diff_var * (torch.matmul(x_i, w_q[2]) + bias_q[2])
        x_i = x_i_0 + x_i_1 + x_i_2

        sender_0 = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender_1 = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_k[1]) + bias_k[1])
        sender_2 = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_k[2]) + bias_k[2])
        sender = sender_0 + sender_1 + sender_2

        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))
        return torch.squeeze(attention, 1)

    def update(self, aggr_out, residual):
        # Project aggregated output back to input dimension for residual connection
        aggr_out = self.out_proj(aggr_out)
        return self.res * residual + F.gelu(aggr_out)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class HiPatch(nn.Module):
    """
    Hi-Patch model for ISMTS (Irregularly Sampled Multivariate Time Series) classification.
    
    This implementation follows the original Hi-Patch paper closely and is adapted
    for the current framework's data format and training pipeline.
    
    Expected input format (from dataset.get_batch_hipatch):
        - X: [B, M, L, N] - values (batch, patches, time points per patch, variables)
        - mask: [B, M, L, N] - observation mask
        - time: [B, M, L, N] - timestamps (normalized)
        - demo: [B, D] - static/demographic features  
        - labels: [B] - binary labels
    """
    
    def __init__(self, args):
        super(HiPatch, self).__init__()
        
        # Model dimensions from args
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.V  # Number of variables
        self.batch_size = None
        self.n_layer = getattr(args, 'num_layers', 2)
        self.alpha = getattr(args, 'alpha', 0.9)
        self.res = getattr(args, 'res', 1)
        self.patch_layer = getattr(args, 'patch_layer', 3)
        self.n_heads = getattr(args, 'num_heads', 2)
        
        # Static feature dimension
        self.d_static = args.D
        self.n_class = 2  # Binary classification
        
        # Class weight for imbalanced classification
        self.pos_class_weight = getattr(args, 'pos_class_weight', 1.0)
        
        # Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        
        # Observation encoder
        self.obs_enc = nn.Linear(1, args.hid_dim)
        
        # Variable embeddings
        self.nodevec = nn.Embedding(self.N, d_model)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Graph convolutional layers for intra/inter patch
        self.gcs = nn.ModuleList()
        for l in range(self.n_layer):
            self.gcs.append(
                Intra_Inter_Patch_Graph_Layer(
                    self.n_heads, d_model, d_model, 
                    self.alpha, self.patch_layer, self.res
                )
            )

        # Query, key, value matrices for aggregation
        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        # Classification head
        if self.d_static != 0:
            self.emb = nn.Linear(self.d_static, self.N)
            self.classifier = nn.Sequential(
                nn.Linear(self.N * 2, 200),
                nn.ReLU(),
                nn.Linear(200, self.n_class)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.N, 200),
                nn.ReLU(),
                nn.Linear(200, self.n_class)
            )

    def LearnableTE(self, tt):
        """Learnable continuous time embeddings using sin/cos."""
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time):
        """
        Process irregular multivariate time series through hierarchical patch structure.
        
        Args:
            x: [B, N, M, L, D] - Input features (batch, vars, patches, time, hidden)
            mask_X: [B, N, M, L, 1] - Observation mask
            x_time: [B, N, M, L, 1] - Time values
            
        Returns:
            Aggregated representations [B, N, D]
        """
        B, N, M, L, D = x.shape

        # Create variable indices tensor
        variable_indices = torch.arange(N).to(x.device)
        cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)
        cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)

        cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
        cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
        cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

        # Generate graph structure
        cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
        cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
        int_max = torch.iinfo(torch.int32).max
        element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

        if element_count > int_max:
            once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
            sd = 0
            ed = once_num
            total_num = math.ceil(B / once_num)
            for k in range(total_num):
                if k == 0:
                    edge_ind = torch.where(cur_adj[sd:ed] == 1)
                    edge_ind_0 = edge_ind[0]
                    edge_ind_1 = edge_ind[1]
                    edge_ind_2 = edge_ind[2]
                    edge_ind_3 = edge_ind[3]
                elif k == total_num - 1:
                    cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                    edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                    edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                    edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                    edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                else:
                    cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                    edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                    edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                    edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                    edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                sd += once_num
                ed += once_num
        else:
            edge_ind = torch.where(cur_adj == 1)

        source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
        target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
        edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

        edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

        edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
        edge_same_time_diff_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()
        edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
        edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
        edge_same_time_diff_var[edge_self] = 0.0

        # Intra Patch Graph Layer
        for gc in self.gcs:
            cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, 
                      edge_same_time_diff_var, edge_diff_time_same_var, 
                      edge_diff_time_diff_var, 0)
        x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

        # Aggregate node states within patches
        # Handle odd number of patches by adding virtual node
        if M > 1 and M % 2 != 0:
            x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
            mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            M = M + 1

        obs_num_per_patch = torch.sum(mask_X, dim=3)
        x_time_per_patch = torch.sum(x_time, dim=3)
        avg_x_time = x_time_per_patch / torch.where(
            obs_num_per_patch == 0, 
            torch.tensor(1, dtype=x.dtype, device=x.device),
            obs_num_per_patch
        )
        avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)
        time_te = self.LearnableTE(x_time)
        Q = torch.matmul(avg_te, self.w_q)
        K = torch.matmul(time_te, self.w_k)
        V = torch.matmul(x, self.w_v)
        attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        attention = torch.div(attention, Q.shape[-1] ** 0.5)
        attention[torch.where(mask_X == 0)] = -1e10
        scale_attention = torch.softmax(attention, dim=-2)
        mask_X = (obs_num_per_patch > 0).float()
        x = torch.sum((V * scale_attention), dim=-2)
        x_time = avg_x_time

        # Inter Patch Graph Layers
        for n_layer in range(1, self.patch_layer):
            B, N, T, D = x.shape

            cur_x = x.reshape(-1, D)
            cur_x_time = x_time.reshape(-1, 1)

            cur_variable_indices = variable_indices.view(1, N, 1, 1)
            cur_variable_indices = cur_variable_indices.expand(B, N, T, 1).reshape(-1, 1)

            patch_indices = torch.arange(T).float().to(x.device)
            cur_patch_indices = patch_indices.view(1, 1, T)
            missing_indices = torch.where(mask_X.reshape(B, -1) == 0)

            cur_patch_indices = cur_patch_indices.expand(B, N, T).reshape(B, -1)

            patch_indices_matrix_1 = cur_patch_indices.unsqueeze(1).expand(B, N * T, N * T)
            patch_indices_matrix_2 = cur_patch_indices.unsqueeze(-1).expand(B, N * T, N * T)

            patch_interval = patch_indices_matrix_1 - patch_indices_matrix_2
            patch_interval[missing_indices[0], missing_indices[1]] = torch.zeros(
                len(missing_indices[0]), N * T, device=x.device
            )
            patch_interval[missing_indices[0], :, missing_indices[1]] = torch.zeros(
                len(missing_indices[0]), N * T, device=x.device
            )

            edge_ind = torch.where(torch.abs(patch_interval) == 1)

            source_nodes = (N * T * edge_ind[0] + edge_ind[1])
            target_nodes = (N * T * edge_ind[0] + edge_ind[2])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = (
                (cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0
            ).float()
            edge_same_time_diff_var = (
                (cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0
            ).float()
            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0

            if edge_index.shape[1] > 0:
                # Propagate through graph
                for gc in self.gcs:
                    cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, 
                              edge_same_time_diff_var, edge_diff_time_same_var,
                              edge_diff_time_diff_var, n_layer)
                x = rearrange(cur_x, '(b n t) c -> b n t c', b=B, n=N, t=T, c=D)

            # Handle odd number of patches
            if T > 1 and T % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(-2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                T = T + 1

            x = x.view(B, N, T // 2, 2, D)
            x_time = x_time.view(B, N, T // 2, 2, 1)
            mask_X = mask_X.view(B, N, T // 2, 2, 1)

            obs_num_per_patch = torch.sum(mask_X, dim=3)
            x_time_per_patch = torch.sum(x_time, dim=3)
            avg_x_time = x_time_per_patch / torch.where(
                obs_num_per_patch == 0, 
                torch.tensor(1, dtype=x.dtype, device=x.device), 
                obs_num_per_patch
            )
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)
            time_te = self.LearnableTE(x_time)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)

            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            x_time = avg_x_time
        
        # Final aggregation: if x still has more than 2 dims [B, N, T, D], 
        # aggregate over time dimension
        if x.dim() == 4:
            # x: [B, N, T, D] -> [B, N, D]
            # Weighted mean by mask
            B, N, T, D = x.shape
            mask_X_flat = mask_X.view(B, N, T)
            mask_sum = mask_X_flat.sum(dim=-1, keepdim=True)
            mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
            x = (x * mask_X_flat.unsqueeze(-1)).sum(dim=2) / mask_sum
        elif x.dim() == 3:
            # x: [B, N, D] - already correct
            pass
        elif x.dim() == 2:
            # x: [N, D] - add batch dim (shouldn't happen normally)
            x = x.unsqueeze(0)
            
        return x

    def forward(self, X, mask, time, demo, labels=None):
        """
        Forward pass for Hi-Patch classification.
        
        Args:
            X: [B, M, L, N] - values (batch, patches, time per patch, variables)
            mask: [B, M, L, N] - observation mask
            time: [B, M, L, N] - timestamps (normalized)
            demo: [B, D] - static/demographic features
            labels: [B] - binary labels (optional, for computing loss)
            
        Returns:
            If labels provided: loss (scalar)
            Otherwise: probabilities [B]
        """
        B, M, L_in, N = X.shape
        self.batch_size = B
        
        # Reshape: [B, M, L, N] -> [B, N, M, L, 1]
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)
        X = self.obs_enc(X)
        
        time = time.permute(0, 3, 1, 2).unsqueeze(-1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)
        
        # Time embedding
        te_his = self.LearnableTE(time)
        
        # Variable embedding
        var_emb = self.nodevec.weight.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        
        # Combine: observation + variable embedding + time embedding
        X = self.relu(X + var_emb + te_his)

        # Process through hierarchical patch model
        h = self.IMTS_Model(X, mask, time)

        # Classification
        if self.d_static != 0 and demo is not None:
            static_emb = self.emb(demo)
            logits = self.classifier(torch.cat([torch.sum(h, dim=-1), static_emb], dim=-1))
        else:
            logits = self.classifier(torch.sum(h, dim=-1))

        if labels is not None:
            # Compute weighted cross entropy loss
            weight = torch.tensor(
                [1.0, self.pos_class_weight], 
                device=logits.device, 
                dtype=logits.dtype
            )
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(logits, labels.long())
            return loss
        else:
            # Return probability of positive class
            probs = F.softmax(logits, dim=-1)
            return probs[:, 1]
