# -*- coding:utf-8 -*-
"""
MTM: Multi-Scale Token Mixing Transformer for Irregular Multivariate Time Series
Adapted for the current framework from the original MTM implementation.

Original paper: "A Multi-Scale Token Mixing Transformer for Irregular Multivariate Time Series Classification"
KDD 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from timm.layers import DropPath


def precompute_rpe(dim: int, max_len=640, theta=10000.0):
    """Precompute relative positional encodings using RoPE."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(max_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    rpe = torch.polar(torch.ones_like(freqs), freqs)
    return rpe


def apply_rel_pe_qk(xq, xk, pos, rpe):
    """Apply relative positional encoding to query and key."""
    d_model = xq.shape[-1]
    xq_ = xq.float().reshape(-1, d_model // 2, 2)
    xk_ = xk.float().reshape(-1, d_model // 2, 2)
    pos = pos.reshape(-1)
    
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    xq_ = torch.view_as_real(xq_ * rpe[pos, :])
    xk_ = torch.view_as_real(xk_ * rpe[pos, :])
    return xq_.type_as(xq).flatten(1), xk_.type_as(xk).flatten(1)


def precompute_ape(d_model, max_len=640, theta=10000.0):
    """Precompute absolute positional encodings."""
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(theta) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def apply_abs_pe(x, pos, ape):
    """Apply absolute positional encoding."""
    pe = ape[pos.flatten(), :]
    return x + pe.reshape(*(pos.shape), -1)


PE_QK_FUNC = {'rel': apply_rel_pe_qk}


class LayerScale(nn.Module):
    """Layer scale from CaiT paper."""
    def __init__(self, dim: int, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class MLP(nn.Module):
    """MLP with GELU activation and dropout."""
    def __init__(self, d_model, r_hid, drop=0.1, norm_first=True, layer_scale=True):
        super().__init__()
        if layer_scale:
            ls = LayerScale(d_model)
        else:
            ls = nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * r_hid),
            nn.GELU(),
            nn.Linear(d_model * r_hid, d_model),
            ls,
            DropPath(drop)
        )
        self.norm = nn.LayerNorm(d_model)
        self.norm_first = norm_first

    def forward(self, x, x_mask=None):
        if self.norm_first:
            x = x + self.net(self.norm(x))
        else:
            x = self.norm(x + self.net(x))
        return x


class TemporalAttn(nn.Module):
    """Temporal attention module."""
    def __init__(self, d_model, drop=0.1, norm_first=False, layer_scale=True):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop)
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm_first = norm_first
        if layer_scale:
            self.layer_scale = LayerScale(d_model)
        else:
            self.layer_scale = nn.Identity()
        self.d_head = d_model

    def _attn_block(self, x, x_mask, pos, pe, pe_type='rel'):
        bsz, nt, nc, nd = x.shape

        if pe_type in PE_QK_FUNC:
            xq, xk = PE_QK_FUNC[pe_type](self.wq(x), self.wk(x), pos, pe)
            xq = rearrange(xq, "(b t c) d -> (b c) t d", b=bsz, t=nt)
            xk = rearrange(xk, "(b t c) d -> (b c) t d", b=bsz, t=nt)
        else:
            xq, xk = self.wq(x), self.wk(x)
            xq = rearrange(xq, "b t c d -> (b c) t d")
            xk = rearrange(xk, "b t c d -> (b c) t d")

        xv = rearrange(self.wv(x), "b t c d -> b c t d")

        attn = rearrange(
            torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.d_head),
            "(b c) tq tk -> b c tq tk",
            b=bsz
        )

        mask = x_mask.transpose(1, 2)
        mask = mask[:, :, :, None] | mask[:, :, None, :]
        attn = torch.masked_fill(attn, mask, float('-inf'))
        attn = self.drop(F.softmax(attn, -1).nan_to_num(0))

        out = torch.einsum("bcmn,bcnd->bmcd", attn, xv)
        return out, attn

    def forward(self, x, x_mask, pos, pe, pe_type='rel'):
        if self.norm_first:
            out, attn = self._attn_block(self.layer_norm(x), x_mask, pos, pe, pe_type)
            out = x + self.layer_scale(out)
        else:
            out, attn = self._attn_block(x, x_mask, pos, pe, pe_type)
            out = self.layer_norm(x + self.layer_scale(self.drop(out)))
        return out, attn


class TokenMixingAttn(nn.Module):
    """Token mixing attention with channel-wise importance."""
    def __init__(self, d_model, drop=0.1, norm_first=False, layer_scale=True):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop)
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm_first = norm_first
        if layer_scale:
            self.layer_scale = LayerScale(d_model)
        else:
            self.layer_scale = nn.Identity()
        self.d_head = d_model

    def _attn_block(self, x, x_mask, p_mask, pos, pe, imp, idx_c, pe_type='rel'):
        bsz, nt, nc, nd = x.shape
        imp = repeat(imp, "b t -> b t c", c=nc)
        idx_c = repeat(
            torch.where(x_mask, imp, idx_c[:, :nt, ...]),
            'b t c -> b t c d',
            d=nd
        )
        x = torch.gather(x, 2, idx_c)

        if pe_type in PE_QK_FUNC:
            xq, xk = PE_QK_FUNC[pe_type](self.wq(x), self.wk(x), pos, pe)
            xq = rearrange(xq, "(b t c) d -> (b c) t d", b=bsz, t=nt)
            xk = rearrange(xk, "(b t c) d -> (b c) t d", b=bsz, t=nt)
        else:
            xq, xk = self.wq(x), self.wk(x)
            xq = rearrange(xq, "b t c d -> (b c) t d")
            xk = rearrange(xk, "b t c d -> (b c) t d")

        xv = rearrange(self.wv(x), "b t c d -> b c t d")

        attn = rearrange(
            torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.d_head),
            "(b c) tq tk -> b c tq tk",
            b=bsz
        )

        x_mask = x_mask.transpose(1, 2)
        p_mask = p_mask.transpose(1, 2)
        mask = p_mask[:, :, :, None] | p_mask[:, :, None, :]
        attn = F.softmax(attn.masked_fill(mask, float('-inf')), -1).nan_to_num(0)

        weight = torch.where(x_mask, 1 / nt, 1)
        weighted_attn = attn * weight[:, :, None, :]

        out = torch.einsum("bcmn,bcnd->bmcd", self.drop(weighted_attn), xv)
        return out, attn

    def forward(self, x, x_mask, p_mask, pos, pe, imp, idx_c, pe_type='rel', return_imp=False):
        if self.norm_first:
            out, attn = self._attn_block(self.layer_norm(x), x_mask, p_mask, pos, pe, imp, idx_c, pe_type)
            out = x + self.layer_scale(out)
        else:
            out, attn = self._attn_block(x, x_mask, p_mask, pos, pe, imp, idx_c, pe_type)
            out = self.layer_norm(x + self.layer_scale(self.drop(out)))
        
        if return_imp:
            imp = F.softmax(attn, -1).nan_to_num(0)[:, :, 0, 1:].sum(1)
            return out, imp
        else:
            return out


class ChannelAttn(nn.Module):
    """Channel-wise attention module."""
    def __init__(self, d_model, drop=0.1, norm_first=False, layer_scale=True):
        super().__init__()
        self.d_head = d_model
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop)
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm_first = norm_first
        if layer_scale:
            self.layer_scale = LayerScale(d_model)
        else:
            self.layer_scale = nn.Identity()

    def _attn_block(self, x, x_mask):
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        attn = torch.einsum("btqd,btkd->btqk", xq, xk) / math.sqrt(self.d_head)
        attn_mask = x_mask[:, :, None, :] | x_mask[:, :, :, None]
        attn = torch.masked_fill(attn, attn_mask, float('-inf'))
        attn = self.drop(F.softmax(attn, -1).nan_to_num(0))
        out = torch.einsum("btqk,btkd->btqd", attn, xv)
        return out

    def forward(self, x, x_mask, *args, **kwargs):
        if self.norm_first:
            out = self._attn_block(self.layer_norm(x), x_mask)
            out = x + self.layer_scale(out)
        else:
            out = self._attn_block(x, x_mask)
            out = self.layer_norm(x + self.layer_scale(self.drop(out)))
        return out


class CLSHead(nn.Module):
    """Classification head."""
    def __init__(self, d_model, d_static, num_cls, drop=0.1):
        d_out = d_model + d_static
        super().__init__()
        if d_static > 0:
            self.net = nn.Sequential(
                nn.Linear(d_out, d_model * 4),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d_model * 4, num_cls),
            )
        else:
            self.net = nn.Linear(d_out, num_cls)

    def forward(self, x):
        return self.net(x)


# Downsampling utilities
EPS = 1e-7

def expand(x, x_mask, idx_b, idx_t, idx_c):
    """Expand compressed representation to full timegrid."""
    bsz, num_t, num_c, d_model = x.shape
    t_max = idx_t.max().item() + 1
    size = (bsz, t_max, num_c, d_model)
    idx_b = idx_b[:, :num_t, :]
    idx_c = idx_c[:, :num_t, :]
    indices = torch.stack([idx_b[~x_mask], idx_t[~x_mask], idx_c[~x_mask]])
    x = torch.sparse_coo_tensor(indices, x[~x_mask], size).to_dense()
    x_mask = torch.sparse_coo_tensor(
        indices, 
        x.new_ones((indices.shape[1],)),
        size[:-1]
    ).to_dense()
    x_mask = ~(x_mask.to(torch.bool))
    return x, x_mask


def shrink_t(x, x_mask, x_t, ratio):
    """Shrink temporal dimension by ratio."""
    _, _, num_c, d_model = x.shape
    new_ts = []
    for timestamps in x_t // ratio:
        new_ts.append(torch.unique_consecutive(timestamps[timestamps >= 0]))
    from torch.nn.utils.rnn import pad_sequence
    new_t = pad_sequence(new_ts, batch_first=True, padding_value=-1)
    padding_mask = new_t < 0
    idx_t = torch.where(padding_mask, 0, new_t).long()
    x = torch.gather(x, 1, repeat(idx_t, "b t -> b t c d", c=num_c, d=d_model))
    x.masked_fill_(padding_mask[..., None, None], 0)
    x_mask = torch.gather(x_mask, 1, repeat(idx_t, "b t -> b t c", c=num_c))
    x_mask = x_mask.masked_fill(padding_mask[:, :, None], True)
    return x, x_mask, new_t


class Downsample(nn.Module):
    """Downsample layer with masked concat pooling."""
    def __init__(self, d_model, ratio, mode):
        super().__init__()
        self.mode = mode
        self.ratio = ratio
        if self.mode == 'concat':
            self.lin = nn.Linear(d_model * 2, d_model)

    def forward(self, x, x_mask, idx_b, idx_t, idx_c, imp):
        x, x_mask = expand(x, x_mask, idx_b, idx_t, idx_c)

        num_t = x_mask.shape[1]
        res = num_t % self.ratio
        padding = 0 if res == 0 else self.ratio - res
        x = F.pad(x, (0, 0, 0, 0, 0, padding), 'constant', 0)
        x_mask = F.pad(x_mask, (0, 0, 0, padding), 'constant', True)

        x = rearrange(x, "b (t r) c d -> b t r c d", r=self.ratio)
        x_mask = rearrange(x_mask, "b (t r) c -> b t r c", r=self.ratio)

        if self.mode == 'max':
            x = x.masked_fill(x_mask[..., None], float("-inf")).max(dim=2).values
            x = x.nan_to_num(0, 0, 0)
        elif self.mode == 'avg':
            x_sum = x.masked_fill(x_mask[..., None], 0.).sum(dim=2)
            div = (~x_mask).sum(dim=2).unsqueeze(-1) + EPS
            x = x_sum / div
        elif self.mode == 'concat':
            x_max = x.masked_fill(x_mask[..., None], float("-inf")).max(dim=2).values
            x_max = x_max.nan_to_num(0, 0, 0)
            x_sum = x.masked_fill(x_mask[..., None], 0.).sum(dim=2)
            div = (~x_mask).sum(dim=2).unsqueeze(-1) + EPS
            x_avg = x_sum / div
            x = self.lin(torch.cat([x_max, x_avg], dim=-1))
        else:
            raise ValueError(self.mode)

        x_mask = x_mask.all(dim=2)
        x, x_mask, new_t = shrink_t(x, x_mask, idx_t[:, :, 0], self.ratio)
        idx_t = repeat(new_t, 'b t -> b t c', c=idx_t.shape[-1])

        return x, x_mask, idx_t


class DownsampleLayer(nn.Module):
    """Downsample layer wrapper."""
    def __init__(self, d_model, ratio, mode):
        super().__init__()
        self.mode = mode
        self.ratio = ratio
        if self.mode in ['max', 'avg', 'concat']:
            self.down = Downsample(d_model, ratio, mode)
        else:
            self.down = None

    def forward(self, x, x_mask, idx_b, idx_t, idx_c, imp):
        if self.down is not None:
            x, x_mask, idx_t = self.down(x, x_mask, idx_b, idx_t, idx_c, imp)
        return x, x_mask, idx_t


class TokenMixingLayer(nn.Module):
    """Complete token mixing layer with temporal, mixing, and channel attention."""
    def __init__(self, d_model=64, r_hid=4, drop=0.2, norm_first=False):
        super().__init__()
        self.temporal = TemporalAttn(d_model, drop, norm_first)
        self.mixer = TokenMixingAttn(d_model, drop, norm_first)
        self.channel = ChannelAttn(d_model, drop, norm_first)
        self.mlp3 = MLP(d_model, r_hid, drop, norm_first)

    def forward(self, x, x_mask, cls_tok, pos, pe, idx_c, pe_type='rel'):
        x = torch.concat([cls_tok, x], dim=1)
        x_mask = F.pad(x_mask, (0, 0, 1, 0), 'constant', False)
        p_mask = pos < 0
        pos = F.pad(pos + 1, (0, 0, 1, 0), 'constant', 0)
        p_mask = F.pad(p_mask, (0, 0, 1, 0), 'constant', False)
        x, imp = self.temporal(x, x_mask, pos, pe, pe_type)
        imp = imp[:, :, 0, :].argmax(1)
        x, imp = self.mixer(x, x_mask, p_mask, pos, pe, imp, idx_c, pe_type, True)
        x = self.channel(x, x_mask)
        x = self.mlp3(x)
        return x[:, 1:, :, :], x[:, [0], :, :], imp


class MTM(nn.Module):
    """
    MTM: Multi-Scale Token Mixing Transformer
    
    Args:
        args: Configuration with the following attributes:
            - V: Number of channels/variables
            - D: Static feature dimension
            - hid_dim: Hidden dimension (d_model)
            - r_hid: MLP hidden ratio
            - dropout: Dropout rate
            - ratios: List of downsampling ratios
            - down_mode: Downsampling mode ('concat', 'max', 'avg')
            - device: Device to use
    """
    def __init__(self, args):
        super().__init__()
        self.d_model = args.hid_dim
        self.d_static = args.D
        self.ratios = getattr(args, 'ratios', [3, 3, 3])
        self.down_mode = getattr(args, 'down_mode', 'concat')
        drop = args.dropout
        r_hid = getattr(args, 'r_hid', 4)
        norm_first = getattr(args, 'norm_first', True)
        num_chn = args.V
        num_cls = 2
        
        # Positional encodings
        self.register_buffer('rpe', precompute_rpe(self.d_model))
        self.register_buffer('ape', precompute_ape(self.d_model))
        
        # Channel embeddings and CLS token
        self.chn_emb = nn.Embedding(num_chn, self.d_model)
        self.cls_tok = nn.Parameter(torch.rand(num_chn, self.d_model))

        # Input layer
        self.inp_layer = TokenMixingLayer(self.d_model, r_hid, drop, norm_first)
        
        # Multi-scale layers
        self.mixers = nn.ModuleList()
        self.samplers = nn.ModuleList()
        for r in self.ratios:
            self.mixers.append(TokenMixingLayer(self.d_model, r_hid, drop, norm_first))
            self.samplers.append(DownsampleLayer(self.d_model, r, self.down_mode))

        # Classification head
        self.cls_head = CLSHead(self.d_model, self.d_static, num_cls, drop)
        
        # Loss function
        self.pos_class_weight = getattr(args, 'pos_class_weight', 1.0)
        pos_weight = torch.tensor([self.pos_class_weight], device=args.device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, x, x_mask, t, demo=None, labels=None):
        """
        Forward pass.
        
        Args:
            x: [B, T, C] - values
            x_mask: [B, T, C] - observation mask (1 = observed, 0 = missing)
            t: [B, T] - timestamps (integer indices)
            demo: [B, D] - static/demographic features
            labels: [B] - binary labels
            
        Returns:
            Loss if labels provided, otherwise logits [B, 2]
        """
        bsz, nt, nc = x_mask.shape
        nt = nt + 1  # Add CLS token
        dev = x.device
        
        # Create index tensors
        idx_t = repeat(t, "b t -> b t c", c=nc)
        idx_b = repeat(torch.arange(bsz, device=dev), "b -> b t c", t=nt, c=nc)
        idx_c = repeat(torch.arange(nc, device=dev), "c -> b t c", b=bsz, t=nt)
        
        # Channel embeddings
        c_feat = self.chn_emb(torch.arange(nc, device=dev))
        
        # Apply absolute PE and channel features
        x = apply_abs_pe(x.nan_to_num(0)[..., None] * c_feat, idx_t, self.ape)
        
        # CLS token
        cls_tok = repeat(self.cls_tok, "c d -> b 1 c d", b=bsz)
        
        # Input layer
        x, cls_tok, imp = self.inp_layer(x, x_mask, cls_tok, idx_t, self.rpe, idx_c)

        # Multi-scale layers
        for sampler, mixer in zip(self.samplers, self.mixers):
            x, x_mask, idx_t = sampler(x, x_mask, idx_b, idx_t, idx_c, imp)
            x, cls_tok, imp = mixer(x, x_mask, cls_tok, idx_t, self.rpe, idx_c)

        # Aggregate and classify
        outputs = [reduce(cls_tok, "b 1 c d -> b d", 'max')]
        if self.d_static > 0 and demo is not None:
            outputs.append(demo)
        outputs = self.cls_head(torch.cat(outputs, -1))
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self.loss_fn(outputs[:, 1], labels)
            return loss
        else:
            return outputs[:, 1]
