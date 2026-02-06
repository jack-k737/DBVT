"""
SAND (Simply Attend and Diagnose) for time series classification.
References:
    - https://arxiv.org/pdf/1711.03905.pdf
    - https://github.com/khirotaka/SAnD/tree/master
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    def __init__(self, args):
        super().__init__()
        assert args.hid_dim % args.num_heads == 0
        self.dk = args.hid_dim // args.num_heads
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        
        self.Wq = nn.Parameter(torch.empty((args.hid_dim, args.hid_dim)))
        self.Wk = nn.Parameter(torch.empty((args.hid_dim, args.hid_dim)))
        self.Wv = nn.Parameter(torch.empty((args.hid_dim, args.hid_dim)))
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        
        self.Wo = nn.Linear(args.hid_dim, args.hid_dim, bias=False)

    def forward(self, x, mask):
        # x: [B, T, d]
        bsz, T, d = x.size()
        
        queries = torch.matmul(x, self.Wq).view(bsz, T, self.num_heads, self.dk) / np.sqrt(self.dk)
        keys = torch.matmul(x, self.Wk).view(bsz, T, self.num_heads, self.dk)
        values = torch.matmul(x, self.Wv).view(bsz, T, self.num_heads, self.dk)
        
        # Attention scores
        A = torch.einsum('bthd,blhd->bhtl', queries, keys) + mask  # [B, h, T, T]
        A = F.softmax(A, dim=-1)
        A = F.dropout(A, self.dropout, self.training)
        
        # Apply attention
        x = torch.einsum('bhtl,bthd->bhtd', A, values)
        x = self.Wo(x.reshape((bsz, T, d)))
        return x


class FeedForward(nn.Module):
    """Position-wise feed-forward network using 1D convolutions."""
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(args.hid_dim, args.hid_dim * 2, 1),
            nn.ReLU(),
            nn.Conv1d(args.hid_dim * 2, args.hid_dim, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, d, T]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, T, d]
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    def __init__(self, args):
        super().__init__()
        self.mha = MultiHeadAttention(args)
        self.ffn = FeedForward(args)
        self.norm_mha = nn.LayerNorm(args.hid_dim)
        self.norm_ffn = nn.LayerNorm(args.hid_dim)
        self.dropout = args.dropout

    def forward(self, x, mask):
        x2 = F.dropout(self.mha(x, mask), self.dropout, self.training)
        x = self.norm_mha(x + x2)
        x2 = F.dropout(self.ffn(x), self.dropout, self.training)
        x = self.norm_ffn(x + x2)
        return x


class DenseInterpolation(nn.Module):
    """Dense interpolation layer for fixed-size output."""
    def __init__(self, args):
        super().__init__()
        M = getattr(args, 'M', 16)  # Number of output features
        T = args.T
        
        cols = torch.arange(M).reshape((1, M)) / M
        rows = torch.arange(T).reshape((T, 1)) / T
        self.W = (1 - torch.abs(rows - cols)) ** 2
        self.W = nn.Parameter(self.W, requires_grad=False)
        self.M = M

    def forward(self, x):
        # x: [B, T, d]
        bsz = x.size()[0]
        x = torch.matmul(x.transpose(1, 2), self.W)  # [B, d, M]
        return x.reshape((bsz, -1))  # [B, d*M]


class SAND(nn.Module):
    """
    SAND model for irregular time series classification.
    
    Input format:
        - ts: [B, T, V*3] concatenation of (values, obs_mask, delta)
        - demo: [B, D] static/demographic features
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Input embedding
        self.input_embedding = nn.Conv1d(args.V * 3, args.hid_dim, 1)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.empty((1, args.T, args.hid_dim))
        )
        nn.init.normal_(self.positional_encoding)
        
        # Causal attention mask with local window
        r = getattr(args, 'r', 10)  # Attention window size
        indices = torch.arange(args.T)
        # t attends to t-r, ..., t
        mask = torch.logical_and(
            indices[None, :] <= indices[:, None],
            indices[None, :] >= indices[:, None] - r
        ).float()
        self.mask = nn.Parameter(
            (1 - mask) * torch.finfo(mask.dtype).min,
            requires_grad=False
        )
        
        self.dropout = args.dropout
        
        # Transformer layers
        self.transformer = nn.ModuleList([
            TransformerBlock(args) for _ in range(args.num_layers)
        ])
        
        # Dense interpolation
        M = getattr(args, 'M', 16)
        args.M = M
        self.dense_interpolation = DenseInterpolation(args)
        
        # Demographics embedding
        self.demo_emb = nn.Sequential(
            nn.Linear(args.D, args.hid_dim * 2),
            nn.Tanh(),
            nn.Linear(args.hid_dim * 2, args.hid_dim)
        )
        
        # Classification head (hid_dim * M + hid_dim for demo)
        self.binary_head = nn.Linear(args.hid_dim * M + args.hid_dim, 1)
        self.pos_class_weight = torch.tensor(args.pos_class_weight)

    def binary_cls_final(self, logits, labels=None):
        if labels is not None:
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_class_weight
            )
        else:
            return torch.sigmoid(logits)

    def forward(self, ts, demo, labels=None):
        """
        Args:
            ts: [B, T, V*3] time series (values, mask, delta concatenated)
            demo: [B, D] demographics
            labels: [B] binary labels (optional)
        """
        # Input embedding: [B, T, V*3] -> [B, T, hid_dim]
        ts_inp_emb = self.input_embedding(ts.permute(0, 2, 1)).permute(0, 2, 1)
        ts_inp_emb = ts_inp_emb + self.positional_encoding
        
        if self.dropout > 0:
            ts_inp_emb = F.dropout(ts_inp_emb, self.dropout, self.training)
        
        # Transformer layers
        ts_hid_emb = ts_inp_emb
        for layer in self.transformer:
            ts_hid_emb = layer(ts_hid_emb, self.mask)
        
        # Dense interpolation
        ts_emb = self.dense_interpolation(ts_hid_emb)
        
        # Demographics embedding
        demo_emb = self.demo_emb(demo)
        
        # Concatenate and classify
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        logits = self.binary_head(ts_demo_emb)[:, 0]
        
        return self.binary_cls_final(logits, labels)
