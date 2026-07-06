from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.hid_dim % args.num_heads != 0:
            raise ValueError(f'hid_dim={args.hid_dim} must be divisible by num_heads={args.num_heads}')
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, T, d = x.size()
        queries = torch.matmul(x, self.Wq).view(bsz, T, self.num_heads, self.dk) / np.sqrt(self.dk)
        keys = torch.matmul(x, self.Wk).view(bsz, T, self.num_heads, self.dk)
        values = torch.matmul(x, self.Wv).view(bsz, T, self.num_heads, self.dk)
        attn = torch.einsum('bthd,blhd->bhtl', queries, keys) + mask
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, self.dropout, self.training)
        x = torch.einsum('bhtl,blhd->bthd', attn, values)
        x = self.Wo(x.reshape((bsz, T, d)))
        return x


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(args.hid_dim, args.hid_dim * 2, 1),
            nn.ReLU(),
            nn.Conv1d(args.hid_dim * 2, args.hid_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha = MultiHeadAttention(args)
        self.ffn = FeedForward(args)
        self.norm_mha = nn.LayerNorm(args.hid_dim)
        self.norm_ffn = nn.LayerNorm(args.hid_dim)
        self.dropout = args.dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x2 = F.dropout(self.mha(x, mask), self.dropout, self.training)
        x = self.norm_mha(x + x2)
        x2 = F.dropout(self.ffn(x), self.dropout, self.training)
        x = self.norm_ffn(x + x2)
        return x


class DenseInterpolation(nn.Module):
    def __init__(self, args):
        super().__init__()
        M = getattr(args, 'M', 16)
        T = args.T
        cols = torch.arange(M).reshape((1, M)) / M
        rows = torch.arange(T).reshape((T, 1)) / T
        W = (1 - torch.abs(rows - cols)) ** 2
        self.register_buffer('W', W)
        self.M = M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x = torch.matmul(x.transpose(1, 2), self.W)
        return x.reshape((bsz, -1))


class SAND(nn.Module):
    """Project-integrated SAND backbone returning feature vectors."""

    def __init__(self, args):
        super().__init__()
        self.input_embedding = nn.Conv1d(args.V * 3, args.hid_dim, 1)
        self.positional_encoding = nn.Parameter(torch.empty((1, args.T, args.hid_dim)))
        nn.init.normal_(self.positional_encoding)
        r = getattr(args, 'r', 10)
        indices = torch.arange(args.T)
        mask = torch.logical_and(indices[None, :] <= indices[:, None], indices[None, :] >= indices[:, None] - r).float()
        self.register_buffer('mask', (1 - mask) * torch.finfo(mask.dtype).min)
        self.dropout = args.dropout
        self.transformer = nn.ModuleList([TransformerBlock(args) for _ in range(args.num_layers)])
        M = getattr(args, 'M', 16)
        args.M = M
        self.dense_interpolation = DenseInterpolation(args)
        self.demo_emb = None
        if args.D > 0:
            self.demo_emb = nn.Sequential(
                nn.Linear(args.D, args.hid_dim * 2),
                nn.Tanh(),
                nn.Linear(args.hid_dim * 2, args.hid_dim),
            )
        self.last_hid_dim = args.hid_dim * M + (args.hid_dim if args.D > 0 else 0)

    def forward(self, ts: torch.Tensor, demo: torch.Tensor | None = None) -> torch.Tensor:
        ts_inp_emb = self.input_embedding(ts.permute(0, 2, 1)).permute(0, 2, 1)
        ts_inp_emb = ts_inp_emb + self.positional_encoding
        if self.dropout > 0:
            ts_inp_emb = F.dropout(ts_inp_emb, self.dropout, self.training)
        ts_hid_emb = ts_inp_emb
        for layer in self.transformer:
            ts_hid_emb = layer(ts_hid_emb, self.mask)
        ts_emb = self.dense_interpolation(ts_hid_emb)
        if self.demo_emb is None or demo is None:
            return ts_emb
        demo_emb = self.demo_emb(demo)
        return torch.cat((ts_emb, demo_emb), dim=-1)
