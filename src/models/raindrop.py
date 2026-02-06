# -*- coding:utf-8 -*-
"""
Raindrop: Graph-Guided Network for Irregularly Sampled Multivariate Time Series
Adapted for the current framework.

Original paper: "Raindrop: Graph-Guided Network for Irregularly Sampled Multivariate Time Series"
https://arxiv.org/abs/2110.05357
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor
from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter


class PositionalEncodingTF(nn.Module):
    """Positional encoding using sinusoidal functions for time series."""
    
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def forward(self, P_time):
        """
        Args:
            P_time: [T, B] - timestamps (Raindrop format)
        Returns:
            pe: [T, B, d_model] - positional encodings
        """
        device = P_time.device
        T, B = P_time.shape
        
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)
        timescales = torch.tensor(timescales, dtype=P_time.dtype, device=device)
        
        # P_time: [T, B] -> [T, B, 1]
        times = P_time.unsqueeze(-1)
        
        # scaled_time: [T, B, num_timescales]
        scaled_time = times / timescales[None, None, :]
        
        # pe: [T, B, d_model]
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        
        return pe


class ObservationPropagation(MessagePassing):
    """
    Observation propagation layer using graph message passing.
    This implements the inter-sensor message passing in Raindrop.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int,
                 n_nodes: int, ob_dim: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.n_nodes = n_nodes
        self.ob_dim = ob_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.weight = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.bias_param = Parameter(torch.Tensor(heads * out_channels))
        self.nodewise_weights = Parameter(torch.Tensor(self.n_nodes, heads * out_channels))
        self.increase_dim = Linear(in_channels[1], heads * out_channels * 8)
        self.map_weights = Parameter(torch.Tensor(self.n_nodes, heads * 16))

        self.index = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()
        glorot(self.weight)
        if self.bias_param is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias_param, -bound, bound)
        glorot(self.nodewise_weights)
        glorot(self.map_weights)
        self.increase_dim.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], p_t: Tensor, edge_index: Adj, 
                edge_weights=None, use_beta=False, edge_attr: OptTensor = None, 
                return_attention_weights=None):
        """
        Args:
            x: Node features [n_nodes, features]
            p_t: Positional/time encoding [T, d_pe]
            edge_index: Edge indices [2, num_edges]
            edge_weights: Edge weights [num_edges]
            use_beta: Whether to use beta for edge pruning
            edge_attr: Edge attributes
            return_attention_weights: Whether to return attention weights
        """
        self.edge_index = edge_index
        self.p_t = p_t
        self.use_beta = use_beta

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weights=edge_weights, 
                            edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None
        edge_index = self.edge_index

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        use_beta = self.use_beta
        
        if use_beta:
            n_step = self.p_t.shape[0]
            n_edges = x_i.shape[0]

            h_W = self.increase_dim(x_i).view(-1, n_step, 32)
            w_v = self.map_weights[self.edge_index[1]].unsqueeze(1)
            p_emb = self.p_t.unsqueeze(0)
            aa = torch.cat([w_v.repeat(1, n_step, 1), p_emb.repeat(n_edges, 1, 1)], dim=-1)
            beta = torch.mean(h_W * aa, dim=-1)

        if edge_weights is not None:
            if use_beta:
                gamma = beta * (edge_weights.unsqueeze(-1))
                gamma = torch.repeat_interleave(gamma, self.ob_dim, dim=-1)

                # Edge pruning
                all_edge_weights = torch.mean(gamma, dim=1)
                K = int(gamma.shape[0] * 0.5)
                index_top_edges = torch.argsort(all_edge_weights, descending=True)[:K]
                gamma = gamma[index_top_edges]
                self.edge_index = self.edge_index[:, index_top_edges]
                index = self.edge_index[0]
                x_i = x_i[index_top_edges]
            else:
                gamma = edge_weights.unsqueeze(-1)

        self.index = index
        if use_beta:
            self._alpha = torch.mean(gamma, dim=-1)
        else:
            self._alpha = gamma

        gamma = softmax(gamma, index, ptr, size_i)
        gamma = F.dropout(gamma, p=self.dropout, training=self.training)

        out = F.relu(self.lin_value(x_i)).view(-1, self.heads, self.out_channels)
        
        if use_beta:
            out = out * gamma.view(-1, self.heads, out.shape[-1])
        else:
            out = out * gamma.view(-1, self.heads, 1)
            
        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        index = self.index
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)


class TransformerEncoder(nn.Module):
    """Simple Transformer encoder for temporal modeling."""
    
    def __init__(self, hid_dim, nhid, num_heads, num_layers, dropout=0.2):
        super().__init__()
        self.N = num_layers
        self.d = hid_dim
        self.dff = nhid
        self.attention_dropout = dropout
        self.dropout = dropout
        self.h = num_heads
        self.dk = self.d // self.h
        self.all_head_size = self.dk * self.h

        self.Wq = nn.Parameter(self._init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wk = nn.Parameter(self._init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wv = nn.Parameter(self._init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wo = nn.Parameter(self._init_proj((self.N, self.all_head_size, self.d)), requires_grad=True)
        self.W1 = nn.Parameter(self._init_proj((self.N, self.d, self.dff)), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros((self.N, 1, 1, self.dff)), requires_grad=True)
        self.W2 = nn.Parameter(self._init_proj((self.N, self.dff, self.d)), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros((self.N, 1, 1, self.d)), requires_grad=True)

    def _init_proj(self, shape, gain=1):
        x = torch.rand(shape)
        fan_in_out = shape[-1] + shape[-2]
        scale = gain * np.sqrt(6 / fan_in_out)
        x = x * 2 * scale - scale
        return x

    def forward(self, x, src_key_padding_mask):
        """
        Args:
            x: [T, B, d] - input sequence (time-first format)
            src_key_padding_mask: [B, T] - padding mask (True for padding positions)
        """
        # Transpose to [B, T, d] for processing
        x = x.permute(1, 0, 2)
        bsz, max_len, _ = x.size()
        
        # Convert padding mask to attention mask
        # src_key_padding_mask: [B, T], True means padding (should be masked)
        valid_mask = (~src_key_padding_mask).float()  # [B, T], 1 for valid, 0 for padding
        attn_mask = valid_mask[:, :, None] * valid_mask[:, None, :]
        attn_mask = (1 - attn_mask)[:, None, :, :] * torch.finfo(x.dtype).min
        layer_mask = attn_mask
        
        for i in range(self.N):
            # Multi-head attention
            q = torch.einsum('bld,hde->bhle', x, self.Wq[i])
            k = torch.einsum('bld,hde->bhle', x, self.Wk[i])
            v = torch.einsum('bld,hde->bhle', x, self.Wv[i])
            
            A = torch.einsum('bhle,bhke->bhlk', q, k)
            
            if self.training:
                dropout_mask = (torch.rand_like(A) < self.attention_dropout).float() * torch.finfo(x.dtype).min
                layer_mask = attn_mask + dropout_mask
                
            A = A + layer_mask
            A = torch.softmax(A, dim=-1)
            
            v = torch.einsum('bhkl,bhle->bkhe', A, v)
            all_head_op = v.reshape((bsz, max_len, -1))
            all_head_op = torch.matmul(all_head_op, self.Wo[i])
            all_head_op = F.dropout(all_head_op, self.dropout, self.training)
            
            x = (all_head_op + x) / 2
            
            # FFN
            ffn_op = torch.matmul(x, self.W1[i]) + self.b1[i]
            ffn_op = F.gelu(ffn_op)
            ffn_op = torch.matmul(ffn_op, self.W2[i]) + self.b2[i]
            ffn_op = F.dropout(ffn_op, self.dropout, self.training)
            
            x = (ffn_op + x) / 2
        
        # Transpose back to [T, B, d]
        return x.permute(1, 0, 2)


class Raindrop(nn.Module):
    """
    Raindrop model for irregularly sampled multivariate time series classification.
    
    This implementation follows the original Raindrop paper closely.
    
    Expected input format (from dataset.get_batch_raindrop):
        - src: [T, B, 2*V] - values concatenated with mask
        - static: [B, D] - static/demographic features
        - times: [T, B] - timestamps
        - lengths: [B] - sequence lengths
        - labels: [B] - binary labels
    """
    
    def __init__(self, args):
        super().__init__()
        
        # Model dimensions from args
        self.d_inp = args.V  # Number of input variables (sensors)
        self.d_static = args.D  # Static feature dimension
        self.n_classes = 2  # Binary classification
        
        # Hyperparameters
        self.nhead = getattr(args, 'num_heads', 4)
        self.nlayers = getattr(args, 'num_layers', 2)
        self.dropout = getattr(args, 'dropout', 0.2)
        self.max_len = getattr(args, 'T',  48)
        
        # d_model should be divisible by d_inp for per-sensor processing
        base_d_model = getattr(args, 'hid_dim', 64)
        self.d_ob = max(1, base_d_model // self.d_inp)  # Observation embedding dim per sensor
        self.d_model = self.d_inp * self.d_ob  # Total model dimension
        
        # Positional encoding dimension
        self.d_pe = 16
        
        # Sensor-wise mask flag
        
        # Whether to use static features
        self.use_static = self.d_static > 0
        
        # Class weight for imbalanced classification
        self.pos_class_weight = getattr(args, 'pos_class_weight', 1.0)
        
        # Aggregation method
        self.aggreg = 'mean'
        
        # ========== Model Components ==========
        
        # Encoder: project input to d_model dimensions
        self.encoder = nn.Linear(self.d_inp * self.d_ob, self.d_inp * self.d_ob)
        
        # Static embedding
        if self.use_static:
            self.emb = nn.Linear(self.d_static, self.d_inp)
        
        # Positional encoding for time
        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len=self.max_len, MAX=100)
        
        # R_u: learnable parameter for initial observation embedding
        self.R_u = Parameter(torch.Tensor(1, self.d_inp * self.d_ob))
        
        # Global structure: learnable adjacency matrix for graph
        self.register_buffer('adj', torch.ones(self.d_inp, self.d_inp))
        
        # Observation propagation layers (graph neural network)
        obs_channels = int(self.max_len * self.d_ob)  # Ensure native Python int
        self.ob_propagation = ObservationPropagation(
            in_channels=obs_channels,
            out_channels=obs_channels,
            heads=1,
            n_nodes=self.d_inp,
            ob_dim=self.d_ob,
            dropout=self.dropout
        )
        
        self.ob_propagation_layer2 = ObservationPropagation(
            in_channels=obs_channels,
            out_channels=obs_channels,
            heads=1,
            n_nodes=self.d_inp,
            ob_dim=self.d_ob,
            dropout=self.dropout
        )
        
        # Transformer encoder for temporal modeling
        # if self.sensor_wise_mask:
        #     transformer_dim = self.d_inp * (self.d_ob + self.d_pe)
        # else:
        transformer_dim = self.d_model + self.d_pe
            
        nhid = 2 * transformer_dim
        self.transformer_encoder = TransformerEncoder(
            hid_dim=transformer_dim,
            nhid=nhid,
            num_heads=self.nhead,
            num_layers=self.nlayers,
            dropout=self.dropout
        )
        
        # Final classifier
        if self.use_static:
            d_final = transformer_dim + self.d_inp
        else:
            d_final = transformer_dim
            
        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, self.n_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.use_static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)
    
    def forward(self, src, static, times, lengths, labels=None):
        """
        Forward pass for Raindrop model.
        
        Args:
            src: [T, B, 2*V] - values concatenated with mask
            static: [B, D] - static/demographic features
            times: [T, B] - timestamps
            lengths: [B] - sequence lengths (number of valid timesteps)
            labels: [B] - binary labels (optional, for computing loss)
            
        Returns:
            If labels provided: loss (scalar)
            Otherwise: probabilities [B]
        """
        device = src.device
        maxlen, batch_size = src.shape[0], src.shape[1]
        
        # Split values and mask
        # src: [T, B, 2*V] -> values [T, B, V], mask [T, B, V]
        missing_mask = src[:, :, self.d_inp:int(2 * self.d_inp)]
        src_values = src[:, :, :self.d_inp]
        
        n_sensor = self.d_inp
        n_step = maxlen
        
        # Repeat interleave to expand dimensions: [T, B, V] -> [T, B, V*d_ob]
        src_expanded = torch.repeat_interleave(src_values, self.d_ob, dim=-1)
        
        # Apply R_u: [T, B, V*d_ob]
        h = F.relu(src_expanded * self.R_u)
        
        # Get positional encoding: [T, B, d_pe]
        pe = self.pos_encoder(times)
        
        # Static embedding
        if self.use_static and static is not None:
            emb = self.emb(static)  # [B, d_inp]
        else:
            emb = None
        
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Create padding mask: [B, T], True for padding positions
        mask = torch.arange(maxlen, device=device)[None, :] >= lengths[:, None]
        
        # ========== Step 1: Graph Propagation ==========
        
        # Create adjacency matrix with self-loops
        adj = self.adj.clone()
        adj[torch.eye(self.d_inp, device=device).bool()] = 1
        
        # Get edge index and weights
        edge_index = torch.nonzero(adj).T  # [2, num_edges]
        edge_weights = adj[edge_index[0], edge_index[1]]
        
        # Output tensor: [T, B, V*d_ob]
        output = torch.zeros([n_step, batch_size, self.d_inp * self.d_ob], device=device)
        alpha_all = torch.zeros([edge_index.shape[1], batch_size], device=device)
        
        # Process each sample in batch
        for unit in range(batch_size):
            # Get sample data: [T, V*d_ob]
            stepdata = h[:, unit, :]
            p_t = pe[:, unit, :]  # [T, d_pe]
            
            # Reshape: [T, V*d_ob] -> [T, V, d_ob] -> [V, T, d_ob] -> [V, T*d_ob]
            stepdata = stepdata.view(n_step, self.d_inp, self.d_ob).permute(1, 0, 2)
            stepdata = stepdata.reshape(self.d_inp, n_step * self.d_ob)
            
            # First observation propagation layer
            stepdata, attn_weights = self.ob_propagation(
                stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                use_beta=False, return_attention_weights=True
            )
            
            edge_index_layer2 = attn_weights[0]
            edge_weights_layer2 = attn_weights[1].squeeze(-1)
            
            # Second observation propagation layer
            stepdata, attn_weights = self.ob_propagation_layer2(
                stepdata, p_t=p_t, edge_index=edge_index_layer2, 
                edge_weights=edge_weights_layer2,
                use_beta=False, return_attention_weights=True
            )
            
            # Reshape back: [V, T*d_ob] -> [V, T, d_ob] -> [T, V, d_ob] -> [T, V*d_ob]
            stepdata = stepdata.view(self.d_inp, n_step, self.d_ob)
            stepdata = stepdata.permute(1, 0, 2)
            stepdata = stepdata.reshape(n_step, self.d_inp * self.d_ob)
            
            output[:, unit, :] = stepdata
            alpha_all[:, unit] = attn_weights[1].squeeze(-1)
        
        # ========== Step 2: Combine with PE and Transformer ==========
        
        # if self.sensor_wise_mask:
        #     # Sensor-wise: [T, B, V, d_ob]
        #     extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
        #     # [T, B, V, d_pe]
        #     extended_pe = pe.unsqueeze(2).repeat(1, 1, self.d_inp, 1)
        #     # [T, B, V, d_ob + d_pe] -> [T, B, V*(d_ob+d_pe)]
        #     output = torch.cat([extend_output, extended_pe], dim=-1)
        #     output = output.view(-1, batch_size, self.d_inp * (self.d_ob + self.d_pe))
        # else:
            # Concatenate with positional encoding: [T, B, V*d_ob + d_pe]
        output = torch.cat([output, pe], dim=-1)
        
        # Transformer encoding: [T, B, d]
        r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        
        # ========== Step 3: Aggregation ==========
        
        # Masked mean aggregation
        lengths_expanded = lengths.unsqueeze(1)  # [B, 1]
        mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [T, B, 1]
        
        # if self.sensor_wise_mask:
        #     # Sensor-wise aggregation
        #     output_final = torch.zeros([batch_size, self.d_inp, self.d_ob + self.d_pe], device=device)
        #     extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
        #     for se in range(self.d_inp):
        #         r_out_view = r_out.view(-1, batch_size, self.d_inp, self.d_ob + self.d_pe)
        #         out = r_out_view[:, :, se, :]
        #         length_se = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
        #         out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (length_se + 1)
        #         output_final[:, se, :] = out_sensor
        #     output = output_final.view(-1, self.d_inp * (self.d_ob + self.d_pe))
        # else:
            # Standard masked mean: [B, d]
        output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths_expanded + 1)
        
        # ========== Step 4: Classification ==========
        
        # Add static features
        if self.use_static and emb is not None:
            output = torch.cat([output, emb], dim=1)
        
        # Classifier
        logits = self.mlp_static(output)  # [B, n_classes]
        
        if labels is not None:
            # Compute weighted cross entropy loss
            weight = torch.tensor([1.0, self.pos_class_weight], device=device, dtype=logits.dtype)
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(logits, labels.long())
            return loss
        else:
            # Return probability of positive class
            probs = F.softmax(logits, dim=-1)
            return probs[:, 1]
