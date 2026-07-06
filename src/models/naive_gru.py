
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor
from argparse import Namespace
from einops import rearrange, repeat


# ============================================================================
# GRU-D Components
# ============================================================================

class GRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, dropout):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W_gamma = nn.Parameter(torch.zeros(1, input_size), requires_grad=True)
        self.b_gamma = nn.Parameter(torch.zeros(1, input_size), requires_grad=True)
        self.W_gamma_h = nn.Parameter(torch.zeros(input_size, hidden_size), requires_grad=True)
        self.b_gamma_h = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

        self.W_z = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.W_r = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.W = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.U_z = nn.Parameter(torch.empty((hidden_size, hidden_size)), requires_grad=True)
        self.U_r = nn.Parameter(torch.empty((hidden_size, hidden_size)), requires_grad=True)
        self.U = nn.Parameter(torch.empty((hidden_size, hidden_size)), requires_grad=True)
        self.V_z = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.V_r = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.V = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.b_z = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        self.b_r = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

        nn.init.xavier_uniform_(self.W_z)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V_z)
        nn.init.xavier_uniform_(self.V_r)
        nn.init.xavier_uniform_(self.V)
        nn.init.orthogonal_(self.U_z)
        nn.init.orthogonal_(self.U_r)
        nn.init.orthogonal_(self.U)

        self.reset_dropout_masks()

    def reset_dropout_masks(self):
        self._dropout_mask = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        self._recurrent_dropout_mask = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        self._masking_dropout_mask = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]

    @jit.script_method
    def forward(self, input, state):
        # type: (Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        x_t, m_t = input
        h_tm1, xprev = state
        # Update last observed value record
        xprev = xprev * (1 - m_t) + m_t * x_t

        if self.dropout > 0 and self._dropout_mask[0].ndim == 0:
            self._dropout_mask = [F.dropout(torch.ones_like(x_t), 
                                  self.dropout, self.training) for i in range(3)]
            self._recurrent_dropout_mask = [F.dropout(torch.ones_like(h_tm1),
                                             self.dropout, self.training) for i in range(3)]
            self._masking_dropout_mask = [F.dropout(torch.ones_like(m_t), 
                                            self.dropout, self.training) for i in range(3)]
            
        # gamma_t = torch.exp(-torch.clip(self.W_gamma * delta_t + self.b_gamma, min=0))
        # x_t = m_t * x_t + (1 - m_t) * gamma_t * xprev
        # gamma_ht = torch.exp(-torch.clip(torch.matmul(delta_t, self.W_gamma_h) + self.b_gamma_h, min=0))
        # h_tm1 = gamma_ht * h_tm1
        x_t = m_t * x_t + (1 - m_t) * xprev

        if self.dropout > 0:
            x_t_z = self._dropout_mask[0] * x_t
            x_t_r = self._dropout_mask[1] * x_t
            x_t_h = self._dropout_mask[2] * x_t
            m_t_z = self._masking_dropout_mask[0] * m_t
            m_t_r = self._masking_dropout_mask[1] * m_t
            m_t_h = self._masking_dropout_mask[2] * m_t
            h_tm1_z = self._recurrent_dropout_mask[0] * h_tm1
            h_tm1_r = self._recurrent_dropout_mask[1] * h_tm1
            h_tm1_h = self._recurrent_dropout_mask[2] * h_tm1
        else:
            x_t_z = x_t_r = x_t_h = x_t
            m_t_z = m_t_r = m_t_h = m_t
            h_tm1_z = h_tm1_r = h_tm1_h = h_tm1

        z_t = torch.sigmoid(torch.matmul(x_t_z, self.W_z)
                            + torch.matmul(h_tm1_z, self.U_z)
                            + torch.matmul(m_t_z, self.V_z) + self.b_z)
        r_t = torch.sigmoid(torch.matmul(x_t_r, self.W_r)
                            + torch.matmul(h_tm1_r, self.U_r)
                            + torch.matmul(m_t_r, self.V_r) + self.b_r)
        h_t = torch.tanh(torch.matmul(x_t_h, self.W)
                         + torch.matmul(r_t * h_tm1_h, self.U)
                         + torch.matmul(m_t_h, self.V) + self.b)
        h_t = (1 - z_t) * h_tm1 + z_t * h_t

        return h_t, (h_t, xprev)


class NaiveGRU(jit.ScriptModule):
    def __init__(self, *cell_args):
        super(NaiveGRU, self).__init__()
        self.cell = GRUCell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]) -> Tensor
        x_t = input[0].unbind(1)
        m_t = input[1].unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        self.cell.reset_dropout_masks()
        for i in range(len(x_t)):
            out, state = self.cell((x_t[i], m_t[i]), state)
            outputs += [out]
        return torch.stack(outputs).transpose(0, 1)

# ============================================================================
# Embedding Modules for Transformer Branch
# ============================================================================

# ============================================================================
# GRU Branch
# ============================================================================

class GRUBranch(nn.Module):
    """GRU branch that outputs hidden representation."""
    
    def __init__(self, args: Namespace):
        super().__init__()
        hid_dim = getattr(args, 'gru_hid_dim', 128)
        dropout = getattr(args, 'gru_dropout', 0.3)
        
        self.gru = NaiveGRU(args.V, hid_dim, dropout)
        self.V = args.V
        self.hid_dim = hid_dim
        self.last_hid_dim = hid_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.demo_emb = nn.Sequential(
            nn.Linear(args.D, hid_dim * 2),
            nn.Tanh(),
            nn.Linear(hid_dim * 2, hid_dim)
        )
    
    def forward(self, x_t, m_t, seq_len, demo):
        """Returns [B, hid_dim * 2]."""
        bsz = x_t.size()[0]
        device = x_t.device
        initial_state = (torch.zeros((bsz, self.hid_dim), device=device), 
                         torch.zeros((bsz, self.V), device=device)) 
        ts_emb = self.gru((x_t, m_t), initial_state)
        
        bsz, max_len, d = ts_emb.size()
        index = (seq_len - 1)[:, None, None].repeat((1, 1, d))
        ts_emb = torch.gather(ts_emb, 1, index)[:, 0, :]
        ts_emb = self.dropout(ts_emb)

        demo_emb = self.demo_emb(demo)
        return torch.cat((ts_emb, demo_emb), dim=-1)

# ============================================================================
# Bidirectional Transformer Branch
# ============================================================================


# ============================================================================
# DBVT: Dual-Branch Variable-Temporal Network
# ============================================================================

# class DBVT(nn.Module):
#     """
#     DBVT: Dual-Branch Variable-Temporal Network.
    
#     A bi-branch model for capturing Variable correlation and Temporal dependency
#     in irregular medical time series.
    
#     Combines:
#     - GRU Branch: Handles temporal decay and missing value imputation (Temporal dependency)
#     - Bidirectional Transformer Branch: Captures temporal and cross-variable patterns (Variable correlation)
    
#     Both branches run in parallel using CUDA Streams for efficiency.
    
#     Input format:
#         GRU: x_t, m_t, delta_t, seq_len
#         Transformer: observed_data, observed_mask, observed_tp, tau
#         Shared: demo, labels
#     """
    
#     def __init__(self, args: Namespace):
#         super().__init__()
        
#         # Initialize branches
#         self.gru_branch = GRUBranch(args)
#         self.transformer_branch = TransformerBranch(args)
        
#         # Get dimensions
#         gru_hid_dim = getattr(args, 'gru_hid_dim', 128)
#         transformer_hid_dim = getattr(args, 'transformer_hid_dim', 64)
        
#         gru_dim = gru_hid_dim * 2  # ts_emb + demo_emb
#         transformer_dim = transformer_hid_dim
#         combined_dim = gru_dim + transformer_dim
        
#         # Fusion layer
#         fusion_hid_dim = getattr(args, 'fusion_hid_dim', 128)
#         fusion_dropout = getattr(args, 'fusion_dropout', 0.3)
        
#         self.fusion = nn.Sequential(
#             nn.Linear(combined_dim, fusion_hid_dim),
#             nn.ReLU(),
#             nn.Dropout(fusion_dropout),
#             nn.Linear(fusion_hid_dim, fusion_hid_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(fusion_dropout)
#         )
#         self.classifier = nn.Linear(fusion_hid_dim // 2, 1)

#         self.aux_classifier_gru = nn.Linear(gru_dim, 1)
#         self.aux_classifier_transformer = nn.Linear(transformer_dim, 1)
        
#         # 投影层：将两个分支表征投影到相同维度以计算余弦相似度
#         proj_dim = min(gru_dim, transformer_dim)
#         self.proj_gru = nn.Linear(gru_dim, proj_dim)
#         self.proj_transformer = nn.Linear(transformer_dim, proj_dim)
        
#         # 损失权重
#         self.lambda_aux = getattr(args, 'lambda_aux', 0.1)  # 辅助损失权重 (0.1~0.5)
#         self.lambda_cos = getattr(args, 'lambda_cos', 0.05)  # 余弦相似度惩罚权重 (0.01~0.1)
        
#         # 注册为 buffer，确保设备自动同步
#         self.register_buffer('pos_class_weight', torch.tensor(args.pos_class_weight))
        
#         # CUDA streams for parallel execution
#         self._gru_stream = None
#         self._transformer_stream = None
#         self.use_parallel = getattr(args, 'use_parallel', True)
    
#     def _ensure_streams(self, device):
#         if self._gru_stream is None and device.type == 'cuda':
#             self._gru_stream = torch.cuda.Stream(device=device)
#             self._transformer_stream = torch.cuda.Stream(device=device)
        
#     def binary_cls_final(self, logits, labels=None):
#         if labels is not None:
#             return F.binary_cross_entropy_with_logits(
#                 logits, labels, pos_weight=self.pos_class_weight
#             )
#         else:
#             return torch.sigmoid(logits)

#     def cosine_similarity_loss(self, h1, h2):
#         """
#         计算余弦相似度惩罚，让两个分支表征互补而不重复。
#         L_cos = E[(h1·h2 / (||h1|| * ||h2||))^2]
#         用平方是为了"正负相关都惩罚"。
#         """
#         h1_norm = F.normalize(h1, p=2, dim=-1)
#         h2_norm = F.normalize(h2, p=2, dim=-1)
#         cos_sim = (h1_norm * h2_norm).sum(dim=-1)  # [B]
#         return (cos_sim ** 2).mean()
    
#     def _forward_sequential(self, x_t, m_t, delta_t, seq_len,
#                            observed_data, observed_mask, observed_tp, tau, demo):
#         gru_repr = self.gru_branch(x_t, m_t, delta_t, seq_len, demo)
#         transformer_repr = self.transformer_branch(observed_data, observed_mask, observed_tp, tau)
#         return gru_repr, transformer_repr
    
#     def _forward_parallel(self, x_t, m_t, delta_t, seq_len,
#                          observed_data, observed_mask, observed_tp, tau, demo):
#         device = x_t.device
#         self._ensure_streams(device)
        
#         current_stream = torch.cuda.current_stream(device)
#         ready_event = torch.cuda.Event()
#         ready_event.record(current_stream)
        
#         # GRU branch
#         self._gru_stream.wait_event(ready_event)
#         with torch.cuda.stream(self._gru_stream):
#             gru_repr = self.gru_branch(x_t, m_t, delta_t, seq_len, demo)
        
#         # Transformer branch (parallel!)
#         self._transformer_stream.wait_event(ready_event)
#         with torch.cuda.stream(self._transformer_stream):
#             transformer_repr = self.transformer_branch(observed_data, observed_mask, observed_tp, tau)
        
#         # Sync
#         gru_done = torch.cuda.Event()
#         transformer_done = torch.cuda.Event()
#         gru_done.record(self._gru_stream)
#         transformer_done.record(self._transformer_stream)
        
#         current_stream.wait_event(gru_done)
#         current_stream.wait_event(transformer_done)
        
#         return gru_repr, transformer_repr
    
#     def forward(self,
#                 # GRU inputs
#                 x_t, m_t, delta_t, seq_len,
#                 # Transformer inputs  
#                 observed_data, observed_mask, observed_tp, tau,
#                 # Shared
#                 demo, labels=None):
#         """Forward pass with parallel branch execution."""
        
#         if self.use_parallel and x_t.device.type == 'cuda':
#             gru_repr, transformer_repr = self._forward_parallel(
#                 x_t, m_t, delta_t, seq_len,
#                 observed_data, observed_mask, observed_tp, tau, demo
#             )
#         else:
#             gru_repr, transformer_repr = self._forward_sequential(
#                 x_t, m_t, delta_t, seq_len,
#                 observed_data, observed_mask, observed_tp, tau, demo
#             )
        
#         # Combine and classify
#         combined = torch.cat([gru_repr, transformer_repr], dim=-1)
#         fused = self.fusion(combined)
#         logits = self.classifier(fused)[:, 0]
        
#         if labels is not None:
#             # ========== 训练模式：计算所有损失 ==========
#             # 主任务损失
#             L_task = F.binary_cross_entropy_with_logits(
#                 logits, labels, pos_weight=self.pos_class_weight
#             )
            
#             # 辅助分类损失（深监督，防止分支躺平）
#             aux_logits_gru = self.aux_classifier_gru(gru_repr)[:, 0]
#             aux_logits_transformer = self.aux_classifier_transformer(transformer_repr)[:, 0]
#             L_aux = (
#                 F.binary_cross_entropy_with_logits(aux_logits_gru, labels, pos_weight=self.pos_class_weight) +
#                 F.binary_cross_entropy_with_logits(aux_logits_transformer, labels, pos_weight=self.pos_class_weight)
#             )
            
#             # 余弦相似度惩罚（让表征互补而不重复）
#             # 先投影到相同维度再计算
#             gru_proj = self.proj_gru(gru_repr)
#             transformer_proj = self.proj_transformer(transformer_repr)
#             L_cos = self.cosine_similarity_loss(gru_proj, transformer_proj)
            
#             # 总损失
#             loss = L_task + self.lambda_aux * L_aux + self.lambda_cos * L_cos
#             return loss
#         else:
#             # ========== 推理模式：只返回预测 ==========
#             return torch.sigmoid(logits)
