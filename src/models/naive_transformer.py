import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor
from argparse import Namespace
from einops import rearrange, repeat

PAD = 0

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""
    
    def __init__(self, d_model, n_head, d_k, d_v, temperature, attn_dropout=0.2):
        super().__init__()
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.n_head = n_head

    def forward(self, q, k, v, mask=None):
        q = rearrange(self.w_qs(q), 'b k l (n d) -> b k n l d', n=self.n_head)
        k = rearrange(self.w_ks(k), 'b k l (n d) -> b k n d l', n=self.n_head)
        v = rearrange(self.w_vs(v), 'b k l (n d) -> b k n l d', n=self.n_head)
        
        attn = torch.matmul(q, k) / self.temperature
        if mask is not None:
            if attn.dim() > mask.dim():
                mask = mask.unsqueeze(2).expand(attn.shape)
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        v = torch.matmul(attn, v)
        v = rearrange(v, 'b k n l d -> b k l (n d)')
        return v, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(d_v * n_head, d_model)
        self.attention = ScaledDotProductAttention(
            d_model, n_head, d_k, d_v,
            temperature=d_k ** 0.5,
            attn_dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        output, attn = self.attention(q, k, v, mask=mask)
        output = self.dropout(self.fc(output))
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward network."""
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


def get_attn_key_pad_mask(mask, transpose=False):
    """Generate attention mask for padding."""
    if transpose:
        seq_q = rearrange(mask, 'b l k -> b k l 1')
        seq_k = rearrange(mask, 'b l k -> b k 1 l')
    else:
        seq_q = rearrange(mask, 'b k l -> b k l 1')
        seq_k = rearrange(mask, 'b k l -> b k 1 l')
    padding_mask = torch.matmul(seq_q, seq_k).eq(PAD)
    return padding_mask


class BidirectionalTransformerLayer(nn.Module):
    """Bidirectional Transformer Layer (temporal + variable attention)."""
    
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.temporal_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.variable_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, non_pad_mask=None):
        tem_mask = get_attn_key_pad_mask(non_pad_mask, transpose=False)
        var_mask = get_attn_key_pad_mask(non_pad_mask, transpose=True)
        
        # Temporal attention
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.temporal_attn(x, x, x, mask=tem_mask)
        x = x + residual
        
        x = rearrange(x, 'b k l d -> b l k d')
        
        # Variable attention
        residual = x
        x = self.layer_norm2(x)
        x, _ = self.variable_attn(x, x, x, mask=var_mask)
        x = x + residual
        
        # Feed-forward
        residual = x
        x = self.layer_norm3(x)
        x = self.pos_ffn(x)
        x = x + residual
        
        x = rearrange(x, 'b l k d -> b k l d')
        return x

class ValueEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = nn.Linear(1, d_model)

    def forward(self, x, mask):
        mask = rearrange(mask, 'b l k -> b l k 1')
        x = rearrange(x, 'b l k -> b l k 1')
        return self.encoder(x) * mask


class TimeEncoder(nn.Module):
    def __init__(self, d_model, num_types):
        super().__init__()
        self.periodic = nn.Linear(1, d_model - 1)
        self.linear = nn.Linear(1, 1)
        self.k_map = nn.Parameter(torch.ones(1, 1, num_types, d_model))

    def forward(self, tt, mask):
        mask = rearrange(mask, 'b l k -> b l k 1')
        if tt.dim() == 3:
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:
            tt = rearrange(tt, 'b l -> b l 1 1')
        out1 = self.linear(tt)
        out2 = torch.sin(self.periodic(tt))
        out = torch.cat([out1, out2], dim=-1)
        return torch.mul(out, self.k_map) * mask


class VariableTypeEncoder(nn.Module):
    def __init__(self, d_model, num_types):
        super().__init__()
        self.embedding = nn.Embedding(num_types + 1, d_model, padding_idx=PAD)
        self.type_ids = None

    def forward(self, batch_size, device):
        if self.type_ids is None or self.type_ids.device != device:
            num_types = self.embedding.num_embeddings - 1
            self.type_ids = torch.arange(1, num_types + 1, device=device)
        emb = self.embedding(self.type_ids)
        emb = rearrange(emb, 'k d -> 1 1 k d')
        return emb.expand(batch_size, -1, -1, -1)


class TauEncoder(nn.Module):
    def __init__(self, d_model, num_types, hid_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, d_model)
        )
        self.k_map = nn.Parameter(torch.ones(1, 1, num_types, d_model))

    def forward(self, tau, mask):
        mask = rearrange(mask, 'b l k -> b l k 1')
        tau = rearrange(tau, 'b l k -> b l k 1')
        out = self.encoder(tau)
        return torch.mul(out, self.k_map) * mask


class AttentionAggregator(nn.Module):
    """Aggregate features using attention."""
    
    def __init__(self, d_model):
        super().__init__()
        self.temporal_attn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.Tanh(),
            nn.Linear(d_model * 2, 1, bias=False)
        )
        self.variable_attn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.Tanh(),
            nn.Linear(d_model * 2, 1, bias=False)
        )

    def forward(self, x, mask):
        # Temporal: [B, K, L, D] -> [B, K, D]
        mask_t = rearrange(mask, 'b k l -> b k l 1')
        attn_t = self.temporal_attn(x)
        attn_t = attn_t.masked_fill(mask_t == 0, -1e9)
        attn_t = F.softmax(attn_t, dim=2)
        x = torch.sum(x * attn_t, dim=2)
        
        # Variable: [B, K, D] -> [B, D]
        attn_v = self.variable_attn(x)
        attn_v = F.softmax(attn_v, dim=1)
        return torch.sum(x * attn_v, dim=1)

class TransformerBranch(nn.Module):
    """Bidirectional Transformer branch that outputs hidden representation."""
    
    def __init__(self, args: Namespace):
        super().__init__()
        
        self.d_model = getattr(args, 'transformer_hid_dim', 64)
        self.d_inner = self.d_model * 2
        self.n_layers = getattr(args, 'transformer_num_layers', 2)
        self.n_head = getattr(args, 'transformer_num_heads', 4)
        self.d_k = self.d_model // self.n_head
        self.d_v = self.d_model // self.n_head
        dropout = getattr(args, 'transformer_dropout', 0.2)
        self.num_types = args.V
        self.last_hid_dim = self.d_model
        
        # Embeddings
        self.value_enc = ValueEncoder(self.d_model)
        self.time_enc = TimeEncoder(self.d_model, self.num_types)
        self.type_enc = VariableTypeEncoder(self.d_model, self.num_types)
        self.tau_enc = TauEncoder(self.d_model, self.num_types)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            BidirectionalTransformerLayer(
                self.d_model, self.d_inner, self.n_head,
                self.d_k, self.d_v, dropout
            ) for _ in range(self.n_layers)
        ])
        
        # Aggregator
        self.aggregator = AttentionAggregator(self.d_model)
        
    def forward(self, observed_data, observed_mask, observed_tp, tau, demo=None):
        """Returns [B, d_model]."""
        B = observed_data.size(0)
        device = observed_data.device
        
        # Embeddings
        value_emb = self.value_enc(observed_data, observed_mask)
        time_emb = self.time_enc(observed_tp, observed_mask)
        type_emb = self.type_enc(B, device)
        tau_emb = self.tau_enc(tau, observed_mask)
        
        h = value_emb + time_emb + type_emb + tau_emb
        h = rearrange(h, 'b l k d -> b k l d')
        mask = rearrange(observed_mask, 'b l k -> b k l')
        
        for layer in self.transformer_layers:
            h = layer(h, mask)
        
        return self.aggregator(h, mask)
