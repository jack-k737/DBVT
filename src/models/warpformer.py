"""
Warpformer model adapted for the current dataset framework.
Original implementation from: https://github.com/mims-harvard/Warpformer
All dependencies consolidated into this single file.
"""

from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings
from einops import rearrange, repeat

PAD = 0

###############################################################################
# HELPER MODULES
###############################################################################

class Constant(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        self.output_sizes = output_sizes
        self.const = nn.parameter.Parameter(torch.Tensor(1, *output_sizes))

    def forward(self, inp):
        return self.const.expand(inp.shape[0], *((-1,)*len(self.output_sizes)))

    def reset_parameters(self):
        nn.init.uniform_(self.const, -1, 1)

class Square(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return inp*inp

class Abs(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.abs(inp)

class Exp(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.exp(inp)

###############################################################################
# ATTENTION MODULES
###############################################################################

class ScaledDotProductAttention_bias(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, temperature, attn_dropout=0.2):
        super().__init__()
        
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.n_head = n_head

    def forward(self, q, k, v, mask):
        q = rearrange(self.w_qs(q), 'b k l (n d) -> b k n l d', n=self.n_head)
        k = rearrange(self.w_ks(k), 'b k l (n d) -> b k n d l', n=self.n_head)
        v = rearrange(self.w_vs(v), 'b k l (n d) -> b k n l d', n=self.n_head)
        
        attn = torch.matmul(q , k) / self.temperature

        if mask is not None:
            if attn.dim() > mask.dim():
                mask = mask.unsqueeze(2).expand(attn.shape)
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        v = torch.matmul(attn, v)
        v = rearrange(v, 'b k n l d -> b k l (n d)')

        return v, attn

class Attention(nn.Module):
    def __init__(self, hin_d, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, hin_d)
        self.W = nn.Linear(hin_d, 1, bias=False)
        
    def forward(self, x, mask=None, mask_value=-1e30):
        attn = self.W(torch.tanh(self.linear(x)))
        
        if mask is not None:
            attn = mask * attn + (1-mask)*mask_value
            
        attn = F.softmax(attn, dim=-2)
        x = torch.matmul(x.transpose(-1, -2), attn).squeeze(-1)
        
        return x, attn

###############################################################################
# SUB-LAYERS
###############################################################################

class MultiHeadAttention_tem_bias(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, opt=None):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.opt = opt

        self.fc = nn.Linear(d_v * n_head, d_model)
        self.attention = ScaledDotProductAttention_bias(d_model, n_head, d_k, d_v, 
                                                        temperature=d_k ** 0.5, 
                                                        attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        output, attn = self.attention(q, k, v, mask=mask)
        output = self.dropout(self.fc(output))
        return output, attn

class MultiHeadAttention_type_bias(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc = nn.Linear(d_v * n_head, d_model)
        self.attention = ScaledDotProductAttention_bias(d_model, n_head, d_k, d_v, 
                                                        temperature=d_k ** 0.5, 
                                                        attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        output, attn = self.attention(q, k, v, mask=mask)
        output = self.dropout(self.fc(output))
        return output, attn

class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """
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

###############################################################################
# WARPING LAYER
###############################################################################

class Scoring_Layer(nn.Module):
    def __init__(self, d_model, func_type='l2', active='relu'):
        super(Scoring_Layer, self).__init__()
        self.func_type = func_type

        assert func_type in ['l1','l2','l3','random','ones','pool'], \
            print("func_type should in ['l1','l2','l3','random','ones','pool']")

        if active == 'relu':
            activation = nn.ReLU()
        elif active == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            activation = nn.Identity()

        if func_type == 'l1':
            self.reduce_d = nn.Linear(d_model, 1)
        elif func_type == 'l2':
            self.reduce_d = nn.Sequential(
                nn.Linear(d_model, d_model*2),
                activation,
                nn.Linear(d_model*2, 1, bias=False))
        elif func_type == 'l3':
            self.reduce_d = nn.Sequential(
                nn.Linear(d_model, d_model*2),
                activation,
                nn.Linear(d_model*2, d_model*2),
                activation,
                nn.Linear(d_model*2, 1, bias=False))
        elif func_type == 'pool':
            self.reduce_d = F.max_pool1d

    def forward(self, h, mask=None, mask_value=-1e30):
        b, k, l, d = h.shape
        if self.func_type in ['l2', 'l1', "l3"]:
            h = self.reduce_d(h).squeeze(-1)
        elif self.func_type == 'pool':
            h = rearrange(h, 'b k l d -> (b k) l d')
            h = self.reduce_d(h, d).squeeze(-1)
        elif self.func_type == 'random':
            h = torch.rand((b*k, l), device=h.device)
        elif self.func_type == 'ones':
            h = torch.ones((b*k, l), device=h.device)

        if h.dim() == 3:
            h = rearrange(h, 'b k l -> (b k) l')
        return h

class NormalizedIntegral(nn.Module):
    def __init__(self, nonneg):
        super().__init__()
        if nonneg == 'square':
            self.nonnegativity = Square()
        elif nonneg == 'relu':
            warnings.warn('ReLU non-negativity does not necessarily result in a strictly monotonic warping function gamma!', 
                         RuntimeWarning)
            self.nonnegativity = nn.ReLU()
        elif nonneg == 'exp':
            self.nonnegativity = Exp()
        elif nonneg == 'abs':
            self.nonnegativity = Abs()
        elif nonneg == 'sigmoid':
            self.nonnegativity = nn.Sigmoid()
        elif nonneg == 'softplus':
            self.nonnegativity = nn.Softplus()
        else:
            raise ValueError("unknown non-negativity transformation, try: abs, square, exp, relu, softplus, sigmoid")

    def forward(self, input_seq, mask):
        gamma = self.nonnegativity(input_seq)
        mask_mask = torch.ones(gamma.shape).to(input_seq.device)
        mask_mask[:,0] = 0
        mask = mask * mask_mask
        dgamma = mask * gamma
        gamma = torch.cumsum(dgamma, dim=-1) * mask
        gamma_max = torch.max(gamma, dim=1)[0].unsqueeze(1)
        gamma_max[gamma_max==0] = 1
        gamma = gamma / gamma_max
        return gamma

class VanillaWarp(nn.Module):
    def __init__(self, backend, nonneg_trans='abs'):
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise ValueError("backend must be an instance of torch.nn.Module")
        self.backend = backend
        self.normintegral = NormalizedIntegral(nonneg_trans)

    def forward(self, input_seq, mask):
        score = self.backend(input_seq, mask=mask)
        gamma = self.normintegral(score, mask)
        return gamma

class Almtx(nn.Module):
    def __init__(self, opt, K):
        super().__init__()
        self.S = K
        loc_net = Scoring_Layer(opt.d_model, func_type=opt.warpfunc, active=opt.warpact)
        self.warp = VanillaWarp(loc_net, nonneg_trans=opt.nonneg_trans)
        self.only_down = opt.only_down

    def cal_new_bound(self, Rl, Rr, gamma):
        B, S, L = gamma.shape
        mask = (Rr - gamma >= 0)
        vl, _ = torch.max(mask * gamma.detach(), -1)
        new_Rl = torch.min(vl, torch.arange(0, 1, 1/S).to(gamma.device)).unsqueeze(-1).expand(B,S,L)

        mask = (gamma - Rl >= 0)
        mask[mask==False] = 10
        mask[mask==True] = 1
        vr, _ = torch.min(mask * gamma.detach(), -1)
        tmp_Rr = torch.max(vr, torch.arange(1/S, 1+1/S, 1/S).to(gamma.device)).unsqueeze(-1).expand(B,S,L)
        new_Rr = tmp_Rr.clone()
        new_Rr[:,-1] = tmp_Rr[:,-1] + 1e-4

        return new_Rl, new_Rr

    def get_boundary(self, gamma, Rl, Rr, mask):
        if self.only_down:
            new_Rl, new_Rr = Rl, Rr
        else:
            new_Rl, new_Rr = self.cal_new_bound(Rl, Rr, gamma)

        bound_mask = (gamma - new_Rl >= 0) & (new_Rr - gamma > 0)
        bound_mask = mask * bound_mask
        A = torch.threshold(gamma - new_Rl, 0, 0) + torch.threshold(new_Rr - gamma, 0, 0)

        return A, bound_mask

    def forward(self, input_seq, mask):
        mask = rearrange(mask, 'b k l -> (b k) l')
        gamma = self.warp(input_seq, mask)

        mask = repeat(mask, 'b l -> b s l', s=self.S)
        _, L = gamma.shape
        gamma = repeat(gamma, 'b l -> b s l', s=self.S)

        Rl = torch.arange(0, 1, 1/self.S).unsqueeze(0).unsqueeze(-1).expand(1, self.S, L).to(gamma.device)
        Rr = torch.arange(1/self.S, 1+1/self.S, 1/self.S).unsqueeze(0).unsqueeze(-1).expand(1, self.S, L).to(gamma.device)

        A, bound_mask = self.get_boundary(gamma, Rl, Rr, mask)

        A_diag = A * bound_mask
        A_sum = A_diag.sum(dim=-1, keepdim=True)
        A_sum = torch.where(A_sum==0, torch.ones_like(A_sum), A_sum).to(A_sum.device)
        A_norm = A_diag / A_sum

        return bound_mask.float(), A_norm

###############################################################################
# ENCODER LAYERS
###############################################################################

def get_attn_key_pad_mask_K(mask, transpose=False, full_attn=False):
    """ For masking out the padding part of key sequence. """
    if full_attn:
        if transpose:
            mask = rearrange(mask, 'b l k -> b k l')
        padding_mask = repeat(mask, 'b k l1 -> b k l2 l1', l2=mask.shape[-1]).eq(PAD)
    else:
        if transpose:
            seq_q = rearrange(mask, 'b l k -> b k l 1')
            seq_k = rearrange(mask, 'b l k -> b k 1 l')
        else:
            seq_q = rearrange(mask, 'b k l -> b k l 1')
            seq_k = rearrange(mask, 'b k l -> b k 1 l')
        padding_mask = torch.matmul(seq_q, seq_k).eq(PAD)

    return padding_mask

class EncoderLayer(nn.Module):
    """ Compose with two layers """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, opt=None):
        super(EncoderLayer, self).__init__()

        self.full_attn = opt.full_attn

        self.slf_tem_attn = MultiHeadAttention_tem_bias(
            n_head, d_model, d_k, d_v, dropout=dropout, opt=opt)

        self.slf_type_attn = MultiHeadAttention_type_bias(
            n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input, non_pad_mask=None):
        tem_mask = get_attn_key_pad_mask_K(mask=non_pad_mask, transpose=False, full_attn=self.full_attn)
        type_mask = get_attn_key_pad_mask_K(mask=non_pad_mask, transpose=True, full_attn=self.full_attn)
        
        tem_output = self.layer_norm(input)
        tem_output, enc_tem_attn = self.slf_tem_attn(tem_output, tem_output, tem_output, mask=tem_mask)
        tem_output = tem_output + input
        tem_output = rearrange(tem_output, 'b k l d -> b l k d')

        type_output = self.layer_norm(tem_output)
        type_output, enc_type_attn = self.slf_type_attn(type_output, type_output, type_output, mask=type_mask)
        enc_output = type_output + tem_output
        
        output = self.layer_norm(enc_output)
        output = self.pos_ffn(output)
        output = output + enc_output
        output = rearrange(output, 'b l k d -> b k l d')
        output = self.layer_norm(output)

        return output, enc_tem_attn, enc_type_attn

###############################################################################
# EMBEDDING MODULES
###############################################################################

class FFNN(nn.Module):
    def __init__(self, input_dim, hid_units, output_dim):
        super(FFNN, self).__init__()
        self.hid_units = hid_units
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, hid_units)
        self.W = nn.Linear(hid_units, output_dim, bias=False)

    def forward(self, x):
        x = self.linear(x)
        x = self.W(torch.tanh(x))
        return x

class Value_Encoder(nn.Module):
    def __init__(self, hid_units, output_dim, num_type):
        super(Value_Encoder, self).__init__()
        self.hid_units = hid_units
        self.output_dim = output_dim
        self.num_type = num_type
        self.encoder = nn.Linear(1, output_dim)

    def forward(self, x, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        x = rearrange(x, 'b l k -> b l k 1')
        x = self.encoder(x)
        return x * non_pad_mask

class Event_Encoder(nn.Module):
    def __init__(self, d_model, num_types):
        super(Event_Encoder, self).__init__()
        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=PAD)

    def forward(self, event):
        event_emb = self.event_emb(event.long())
        return event_emb

class Time_Encoder(nn.Module):
    def __init__(self, embed_time, num_types):
        super(Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.k_map = nn.Parameter(torch.ones(1,1,num_types,embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:
            tt = rearrange(tt, 'b l -> b l 1 1')
        
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1)
        out = torch.mul(out, self.k_map)
        return out

class MLP_Tau_Encoder(nn.Module):
    def __init__(self, embed_time, num_types, hid_dim=16):
        super(MLP_Tau_Encoder, self).__init__()
        self.encoder = FFNN(1, hid_dim, embed_time)
        self.k_map = nn.Parameter(torch.ones(1,1,num_types,embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:
            tt = rearrange(tt, 'b l -> b l 1 1')
        
        tt = self.encoder(tt)
        tt = torch.mul(tt, self.k_map)
        return tt * non_pad_mask

###############################################################################
# AGGREGATION AND CLASSIFICATION
###############################################################################

class Attention_Aggregator(nn.Module):
    def __init__(self, dim, task):
        super(Attention_Aggregator, self).__init__()
        self.task = task
        self.attention_len = Attention(dim*2, dim)
        self.attention_type = Attention(dim*2, dim)

    def forward(self, ENCoutput, mask):
        """
        input: [B,K,L,D], mask: [B,K,L]
        """
        if self.task == "active":
            mask = rearrange(mask, 'b k l 1 -> b l k 1')
            ENCoutput = rearrange(ENCoutput, 'b k l d -> b l k d')
            ENCoutput, _ = self.attention_type(ENCoutput, mask)
        else:
            ENCoutput, _ = self.attention_len(ENCoutput, mask)
            ENCoutput, _ = self.attention_type(ENCoutput)
        return ENCoutput

class Classifier(nn.Module):
    def __init__(self, dim, type_num, cls_dim, activate=None):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(dim, cls_dim)

    def forward(self, ENCoutput):
        """
        input: [B,L,K,D], mask: [B,L,K]
        """
        ENCoutput = self.linear(ENCoutput)
        return ENCoutput

###############################################################################
# WARPFORMER MODEL
###############################################################################

class Warpformer_Module(nn.Module):
    def __init__(self, new_l, n_head, d_k, d_v, dropout, opt):
        super().__init__()
        self.new_l = new_l
        self.num_types = opt.num_types
        self.opt = opt
        d_inner = opt.d_inner_hid
        d_model = opt.d_model
        
        if not opt.hourly:
            self.get_almtx = Almtx(opt, self.new_l)
        else:
            self.time_split = [i for i in range(self.new_l+1)]

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, opt=opt)
            for _ in range(opt.n_layers)])

    def hour_aggregate(self, event_time, h0, non_pad_mask):
        new_l = len(self.time_split)-1
        b, k, l, dim = h0.shape
        
        event_time_k = repeat(event_time, 'b l -> b k l', k=self.num_types)
        new_event_time = torch.zeros((b,self.num_types,new_l)).to(h0.device)
        new_h0 = torch.zeros((b,self.num_types,new_l, dim)).to(h0.device)
        new_pad_mask = torch.zeros((b,self.num_types,new_l)).to(h0.device)
        almat = torch.zeros((b,l,new_l)).to(h0.device)
        
        for i in range(len(self.time_split)-1):
            idx = (event_time_k.ge(self.time_split[i]) & event_time_k.lt(self.time_split[i+1]))
            total = torch.sum(idx, dim=-1)
            total[total==0] = 1
            
            tmp_h0 = h0 * idx.unsqueeze(-1)
            tmp_h0 = rearrange(tmp_h0, 'b k l d -> (b k) d l')
            tmp_h0 = F.max_pool1d(tmp_h0, tmp_h0.size(-1)).squeeze()
            new_h0[:,:,i,:] = rearrange(tmp_h0, '(b k) d -> b k d', b=b)
            almat[:,:,i] = (event_time.ge(self.time_split[i]) & event_time.lt(self.time_split[i+1]))

            new_event_time[:,:,i] = torch.sum(event_time_k * idx, dim=-1) / total
            new_pad_mask[:,:,i] = torch.sum(non_pad_mask * idx, dim=-1) / total
        
        almat = repeat(almat, 'b l s -> b k l s', k=k)
        return new_h0, new_event_time, new_pad_mask, almat
    
    def almat_aggregate(self, event_time, h0, non_pad_mask):
        b, k, l, dim = h0.shape
        new_event_time = None

        bound_mask, almat = self.get_almtx(h0, mask=non_pad_mask)
        
        almat = rearrange(almat, '(b k) s l -> b k s l', k=k)
        bound_mask = rearrange(bound_mask, '(b k) s l -> b k s l', k=k)
        new_h0 = torch.matmul(almat, h0)
        
        new_pad_mask = torch.sum(bound_mask, dim=-1)
        new_pad_mask[new_pad_mask > 0] = 1
        new_pad_mask = torch.nan_to_num(new_pad_mask)

        return new_h0, new_event_time, new_pad_mask, almat
    
    def forward(self, h0, non_pad_mask, event_time, id_warp=False):
        if id_warp:
            z0 = h0
            new_pad_mask = non_pad_mask
            almat = None
        else:
            if self.opt.hourly:
                z0, _, new_pad_mask, almat = self.hour_aggregate(event_time, h0, non_pad_mask)
            else:
                z0, _, new_pad_mask, almat = self.almat_aggregate(event_time, h0, non_pad_mask)
        
        for enc_layer in self.layer_stack:
            z0, _, _ = enc_layer(z0, non_pad_mask=new_pad_mask)
        
        return z0, new_pad_mask, almat

class Hie_Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """
    def __init__(
            self, opt,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        
        super().__init__()
        self.opt = opt
        self.d_model = d_model
        self.embed_time = d_model

        # event type embedding
        self.event_enc = Event_Encoder(d_model, num_types)
        self.type_matrix = torch.tensor([int(i) for i in range(1,num_types+1)]).to(opt.device)
        self.type_matrix = rearrange(self.type_matrix, 'k -> 1 1 k')
        self.num_types = num_types
        self.task = opt.task
        
        if len(opt.warp_num) > 0:
            warp_num = opt.warp_num
        elif opt.task == "mor" or opt.task == "wbm":
            warp_num = [0,12]
        else:
            warp_num = [0,6]

        if sum(opt.warp_num) == 0:
            self.no_warping = True
        else:
            self.no_warping = False
        self.full_attn = opt.full_attn

        if not opt.hourly and len(opt.warp_num) == 0:
            warp_layer_num = 1
        elif opt.hourly:
            warp_layer_num = 2
        else:
            warp_layer_num = len(warp_num)
            
        print("warp_num: ", str(warp_num), "\t warp_layer_num:", str(warp_layer_num))

        self.warpformer_layer_stack = nn.ModuleList([
            Warpformer_Module(int(warp_num[i]), n_head, d_k, d_v, dropout, opt)
            for i in range(warp_layer_num)])
        
        self.value_enc = Value_Encoder(hid_units=d_inner, output_dim=d_model, num_type=num_types)
        self.learn_time_embedding = Time_Encoder(self.embed_time, num_types)
        self.w_t = nn.Linear(1, num_types, bias=False)
        self.tau_encoder = MLP_Tau_Encoder(self.embed_time, num_types)
        self.agg_attention = Attention_Aggregator(d_model, task=opt.task)
        self.linear = nn.Linear(d_model*warp_layer_num, d_model)

    def forward(self, event_time, event_value, non_pad_mask, tau=None, return_almat=False):
        """ Encode event sequences via masked self-attention. """
        # embedding
        tem_enc_k = self.learn_time_embedding(event_time, non_pad_mask)
        tem_enc_k = rearrange(tem_enc_k, 'b l k d -> b k l d')

        value_emb = self.value_enc(event_value, non_pad_mask)
        value_emb = rearrange(value_emb, 'b l k d -> b k l d')
        
        self.type_matrix = self.type_matrix.to(non_pad_mask.device)
        event_emb = self.type_matrix
        event_emb = self.event_enc(event_emb)
        event_emb = rearrange(event_emb, 'b l k d -> b k l d')

        tau_emb = self.tau_encoder(tau, non_pad_mask)
        tau_emb = rearrange(tau_emb, 'b l k d -> b k l d')
        
        if self.opt.remove_rep == 'abs':
            h0 = value_emb + tau_emb + event_emb
        elif self.opt.remove_rep == 'type':
            h0 = value_emb + tau_emb + tem_enc_k
        elif self.opt.remove_rep == 'rel':
            h0 = value_emb + event_emb + tem_enc_k
        elif self.opt.remove_rep == 'tem':
            h0 = value_emb + event_emb
        else:
            h0 = value_emb + tau_emb + event_emb + tem_enc_k

        amlt_list = []

        if self.opt.input_only:
            z0 = torch.mean(h0, dim=1)
            if self.task != 'active':
                z0 = torch.mean(z0, dim=1)
        elif self.opt.dec_only:
            z0 = self.agg_attention(h0, rearrange(non_pad_mask, 'b l k -> b k l 1'))
        else:
            non_pad_mask = rearrange(non_pad_mask, 'b l k -> b k l')
            z0 = None
            idwarp=True
            
            for i, enc_layer in enumerate(self.warpformer_layer_stack):
                if enc_layer.new_l == 0:
                    idwarp=True
                else:
                    idwarp=False
                if i > 0 and self.no_warping and self.full_attn:
                    non_pad_mask = torch.ones_like(non_pad_mask).to(non_pad_mask.device)

                h0, non_pad_mask, almat = enc_layer(h0, non_pad_mask, event_time, id_warp=idwarp)
                
                output = self.agg_attention(h0, rearrange(non_pad_mask, 'b k l -> b k l 1'))

                if z0 is not None and z0.shape == output.shape:
                    z0 = z0 + output
                else:
                    z0 = output

                if almat is not None:
                    amlt_list.append(almat.detach().cpu())
            
        if return_almat:
            return z0, amlt_list
        else:
            return z0

###############################################################################
# WARPFORMER CONFIG AND MODEL
###############################################################################

class WarpformerConfig:
    """Configuration class for Warpformer model."""
    def __init__(self, args: Namespace):
        # Core model dimensions
        self.d_model = getattr(args, 'hid_dim', 32)
        self.d_inner_hid = self.d_model * 2
        self.n_layers = getattr(args, 'num_layers', 2)
        self.n_head = getattr(args, 'num_heads', 4)
        self.d_k = self.d_model // self.n_head
        self.d_v = self.d_model // self.n_head
        self.dropout = getattr(args, 'dropout', 0.2)
        
        # Number of variable types
        self.num_types = args.V
        
        # Warping layer configuration
        self.warp_num = getattr(args, 'warp_num', [0, 12])
        self.warpfunc = getattr(args, 'warpfunc', 'l2')
        self.warpact = getattr(args, 'warpact', 'relu')
        self.nonneg_trans = getattr(args, 'nonneg_trans', 'abs')
        
        # Other options
        self.hourly = getattr(args, 'warp_hourly', False)
        self.full_attn = getattr(args, 'warp_full_attn', False)
        self.only_down = getattr(args, 'warp_only_down', False)
        self.input_only = getattr(args, 'warp_input_only', False)
        self.dec_only = getattr(args, 'warp_dec_only', False)
        self.remove_rep = getattr(args, 'warp_remove_rep', None)
        
        # Task type (for attention aggregation)
        self.task = 'mor'  # mortality prediction (binary classification)
        
        # Device
        self.device = args.device
        
        # Number of output classes
        self.n_classes = 2


class Warpformer(nn.Module):
    """
    Warpformer model for irregular time series classification.
    
    Adapted to work with the current dataset framework.
    
    Expected input format:
        - observed_data: [B, L, K] - normalized values
        - observed_mask: [B, L, K] - observation mask
        - observed_tp: [B, L] - timestamps (normalized to [0, 1] or similar)
        - tau: [B, L, K] - time since last observation for each variable
        - demo: [B, D] - static/demographic features
    """
    
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        
        # Create Warpformer config from args
        self.config = WarpformerConfig(args)
        
        # Initialize Warpformer encoder
        self.encoder = Hie_Encoder(
            opt=self.config,
            num_types=self.config.num_types,
            d_model=self.config.d_model,
            d_inner=self.config.d_inner_hid,
            n_layers=self.config.n_layers,
            n_head=self.config.n_head,
            d_k=self.config.d_k,
            d_v=self.config.d_v,
            dropout=self.config.dropout
        )
        
        # Classification head
        self.classifier = Classifier(
            dim=self.config.d_model,
            type_num=self.config.num_types,
            cls_dim=self.config.n_classes
        )
        
        # Demographics embedding
        self.demo_emb = nn.Sequential(
            nn.Linear(args.D, self.config.d_model * 2),
            nn.Tanh(),
            nn.Linear(self.config.d_model * 2, self.config.d_model)
        )
        
        # Final binary classification head
        self.binary_head = nn.Linear(self.config.d_model * 2, 1)
        self.pos_class_weight = torch.tensor(args.pos_class_weight)
        
    def binary_cls_final(self, logits, labels=None):
        """Binary classification loss or prediction."""
        if labels is not None:
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_class_weight
            )
        else:
            return torch.sigmoid(logits)
    
    def forward(self, observed_data, observed_mask, observed_tp, tau, demo, labels=None):
        """
        Forward pass of Warpformer.
        
        Args:
            observed_data: [B, L, K] - normalized observation values
            observed_mask: [B, L, K] - binary mask indicating observed values
            observed_tp: [B, L] - timestamps (normalized)
            tau: [B, L, K] - time since last observation for each variable
            demo: [B, D] - static/demographic features
            labels: [B] - binary labels (optional, for training)
            
        Returns:
            loss if labels provided, else predictions
        """
        # Encoder forward pass
        enc_output = self.encoder(
            event_time=observed_tp,
            event_value=observed_data,
            non_pad_mask=observed_mask,
            tau=tau
        )
        
        # Demographics embedding
        demo_emb = self.demo_emb(demo)
        
        # Concatenate encoder output with demographics
        combined = torch.cat([enc_output, demo_emb], dim=-1)
        
        # Binary classification
        logits = self.binary_head(combined)[:, 0]
        
        return self.binary_cls_final(logits, labels)