from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cde import NeuralCDE
from .grud import GRUDBranch
from .hipatch import HiPatch
from .kedgn import KEDGN
from .mtm import MTM
from .naive_gru import GRUBranch, NaiveGRU
from .raindrop import Raindrop
from .sand import SAND
from .strats import StratsBranch
from .tcn import TCN
from .warpformer import Warpformer
from .naive_transformer import (
    AttentionAggregator,
    BidirectionalTransformerLayer,
    TauEncoder,
    TimeEncoder,
    TransformerBranch,
    ValueEncoder,
    VariableTypeEncoder,
)

BACKBONE_REGISTRY = {
    "cde": NeuralCDE,
    "gru": GRUBranch,
    "grud": GRUDBranch,
    "hipatch": HiPatch,
    "kedgn": KEDGN,
    "mtm": MTM,
    "raindrop": Raindrop,
    "sand": SAND,
    "strats": StratsBranch,
    "tcn": TCN,
    "transformer": TransformerBranch,
    "warpformer": Warpformer,
}


SINGLE_PLUS_FUSION_DEFAULTS = {
    "physionet_2012": {
        "lambda_aux": 0.5,
        "gru_hid_dim": 96,
        "gru_dropout": 0.2,
        "transformer_hid_dim": 32,
        "transformer_num_layers": 4,
        "transformer_num_heads": 8,
        "transformer_dropout": 0.3,
        "fusion_hid_dim": 96,
        "fusion_dropout": 0.3,
    },
    "physionet_2019": {
        "lambda_aux": 0.5,
        "gru_hid_dim": 128,
        "gru_dropout": 0.2,
        "transformer_hid_dim": 32,
        "transformer_num_layers": 4,
        "transformer_num_heads": 4,
        "transformer_dropout": 0.15,
        "fusion_hid_dim": 128,
        "fusion_dropout": 0.2,
    },
    "mimic_iii": {
        "lambda_aux": 0.5,
        "gru_hid_dim": 128,
        "gru_dropout": 0.3,
        "transformer_hid_dim": 32,
        "transformer_num_layers": 2,
        "transformer_num_heads": 2,
        "transformer_dropout": 0.15,
        "fusion_hid_dim": 128,
        "fusion_dropout": 0.1,
    },
}


def _is_single_plus_variant(args) -> bool:
    config_model_type = getattr(args, 'config_model_type', getattr(args, 'output_model_type', ''))
    return config_model_type.endswith('+') and not config_model_type.endswith('++')


def _apply_single_plus_fusion_defaults(args) -> None:
    """Fill missing fusion hyperparameters for single-plus variants only."""
    if not _is_single_plus_variant(args):
        return

    dataset_defaults = SINGLE_PLUS_FUSION_DEFAULTS.get(getattr(args, 'dataset', ''), {})
    for name, default_value in dataset_defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, default_value)


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x, mask):
        scores = self.scorer(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = attn * mask.float()
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn = attn / denom
        return torch.sum(x * attn.unsqueeze(-1), dim=1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, context, query_mask, context_mask):
        attn_out, _ = self.attn(
            query=self.query_norm(query),
            key=self.context_norm(context),
            value=self.context_norm(context),
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        x = query + self.dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x * query_mask.unsqueeze(-1).float()


class GRUSequenceEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        hid_dim = getattr(args, 'gru_hid_dim', 128)
        dropout = getattr(args, 'gru_dropout', 0.3)
        self.V = args.V
        self.hid_dim = hid_dim
        self.gru = NaiveGRU(args.V, hid_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.demo_emb = nn.Sequential(
            nn.Linear(args.D, hid_dim * 2),
            nn.Tanh(),
            nn.Linear(hid_dim * 2, hid_dim),
        )
        self.state_pool = MaskedAttentionPool(hid_dim, hidden_dim=hid_dim, dropout=dropout)
        self.state_gate = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.Sigmoid(),
        )
        self.last_hid_dim = hid_dim * 2

    def forward(self, x_t, m_t, seq_len, demo):
        batch_size = x_t.size(0)
        device = x_t.device
        initial_state = (
            torch.zeros((batch_size, self.hid_dim), device=device),
            torch.zeros((batch_size, self.V), device=device),
        )
        seq_states = self.gru((x_t, m_t), initial_state)
        max_len = seq_states.size(1)
        time_index = torch.arange(max_len, device=device).unsqueeze(0)
        valid_mask = time_index < seq_len.unsqueeze(1)
        valid_mask = valid_mask | (time_index == 0)
        last_index = (seq_len.clamp_min(1) - 1).view(batch_size, 1, 1).expand(-1, 1, self.hid_dim)
        last_state = torch.gather(seq_states, 1, last_index)[:, 0, :]
        last_state = self.dropout(last_state)
        demo_state = self.demo_emb(demo)
        attn_state = self.state_pool(seq_states, valid_mask)
        gate_input = torch.cat([last_state, attn_state, demo_state], dim=-1)
        state_gate = self.state_gate(gate_input)
        fused_state = state_gate * last_state + (1.0 - state_gate) * attn_state
        pooled = torch.cat([fused_state, demo_state], dim=-1)
        return seq_states, valid_mask, pooled


class TransformerTokenEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = getattr(args, 'transformer_hid_dim', 64)
        self.d_inner = self.d_model * 2
        self.n_layers = getattr(args, 'transformer_num_layers', 2)
        self.n_head = getattr(args, 'transformer_num_heads', 4)
        self.d_k = self.d_model // self.n_head
        self.d_v = self.d_model // self.n_head
        self.num_types = args.V
        self.last_hid_dim = self.d_model
        dropout = getattr(args, 'transformer_dropout', 0.2)
        tau_hid_dim = getattr(args, 'tau_hid_dim', 16)

        self.value_enc = ValueEncoder(self.d_model)
        self.time_enc = TimeEncoder(self.d_model, self.num_types)
        self.type_enc = VariableTypeEncoder(self.d_model, self.num_types)
        self.tau_enc = TauEncoder(self.d_model, self.num_types, hid_dim=tau_hid_dim)
        self.transformer_layers = nn.ModuleList([
            BidirectionalTransformerLayer(
                self.d_model,
                self.d_inner,
                self.n_head,
                self.d_k,
                self.d_v,
                dropout,
            )
            for _ in range(self.n_layers)
        ])
        self.aggregator = AttentionAggregator(self.d_model)

    def forward(self, observed_data, observed_mask, observed_tp, tau):
        batch_size = observed_data.size(0)
        device = observed_data.device

        value_emb = self.value_enc(observed_data, observed_mask)
        time_emb = self.time_enc(observed_tp, observed_mask)
        type_emb = self.type_enc(batch_size, device)
        tau_emb = self.tau_enc(tau, observed_mask)

        h = value_emb + time_emb + type_emb + tau_emb
        h = h.permute(0, 2, 1, 3)
        mask = observed_mask.permute(0, 2, 1)

        for layer in self.transformer_layers:
            h = layer(h, mask)

        pooled = self.aggregator(h, mask)
        counts = mask.sum(dim=1, keepdim=False).clamp_min(1.0).unsqueeze(-1)
        time_tokens = (h * mask.unsqueeze(-1)).sum(dim=1) / counts
        return h, mask, time_tokens, pooled


class CrossGatedBranchFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.register_buffer('pos_class_weight', torch.tensor(float(args.pos_class_weight)))
        self.gru_encoder = GRUSequenceEncoder(args)
        self.transformer_encoder = TransformerTokenEncoder(args)

        common_dim = getattr(args, 'cross_hid_dim', 128)
        cross_heads = getattr(args, 'cross_num_heads', 4)
        fusion_dropout = getattr(args, 'fusion_dropout', 0.3)
        fusion_hid_dim = getattr(args, 'fusion_hid_dim', 128)
        pool_hidden_dim = getattr(args, 'pool_hid_dim', common_dim)
        self.aux_weight = getattr(args, 'aux_weight', 0.0)

        self.gru_seq_proj = nn.Linear(self.gru_encoder.hid_dim, common_dim)
        self.trans_seq_proj = nn.Linear(self.transformer_encoder.d_model, common_dim)
        self.gru_base_proj = nn.Linear(self.gru_encoder.last_hid_dim, common_dim)
        self.trans_base_proj = nn.Linear(self.transformer_encoder.last_hid_dim, common_dim)
        self.demo_proj = nn.Sequential(
            nn.Linear(args.D, common_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
        )

        self.gru_to_trans = CrossAttentionBlock(common_dim, cross_heads, fusion_dropout)
        self.trans_to_gru = CrossAttentionBlock(common_dim, cross_heads, fusion_dropout)
        self.gru_pool = MaskedAttentionPool(common_dim, hidden_dim=pool_hidden_dim, dropout=fusion_dropout)
        self.trans_pool = MaskedAttentionPool(common_dim, hidden_dim=pool_hidden_dim, dropout=fusion_dropout)

        gate_in_dim = common_dim * 5
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(common_dim, common_dim),
            nn.Sigmoid(),
        )

        head_in_dim = common_dim * 4
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, fusion_hid_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hid_dim, fusion_hid_dim // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
        )
        self.classifier = nn.Linear(fusion_hid_dim // 2, 1)

        if self.aux_weight > 0:
            self.aux_gru = nn.Linear(common_dim, 1)
            self.aux_trans = nn.Linear(common_dim, 1)

    def forward(self, labels=None, **batch):
        demo = batch['demo']
        gru_seq, gru_mask, gru_pooled = self.gru_encoder(
            batch['x_t'],
            batch['m_t'],
            batch['seq_len'],
            demo,
        )
        _, trans_var_mask, trans_time_tokens, trans_pooled = self.transformer_encoder(
            batch['observed_data'],
            batch['observed_mask'],
            batch['observed_tp'],
            batch['tau'],
        )
        trans_mask = trans_var_mask.any(dim=1)

        gru_seq_common = self.gru_seq_proj(gru_seq)
        trans_seq_common = self.trans_seq_proj(trans_time_tokens)
        gru_base = self.gru_base_proj(gru_pooled)
        trans_base = self.trans_base_proj(trans_pooled)
        demo_feat = self.demo_proj(demo)

        gru_cross = self.gru_to_trans(gru_seq_common, trans_seq_common, gru_mask, trans_mask)
        trans_cross = self.trans_to_gru(trans_seq_common, gru_seq_common, trans_mask, gru_mask)
        gru_cross_pooled = self.gru_pool(gru_cross, gru_mask)
        trans_cross_pooled = self.trans_pool(trans_cross, trans_mask)

        gate_input = torch.cat([
            gru_base,
            trans_base,
            gru_cross_pooled,
            trans_cross_pooled,
            demo_feat,
        ], dim=-1)
        gate = self.gate(gate_input)
        fused = gate * gru_cross_pooled + (1.0 - gate) * trans_cross_pooled

        head_input = torch.cat([fused, gru_base, trans_base, demo_feat], dim=-1)
        logits = self.classifier(self.head(head_input))[:, 0]

        if labels is None:
            return torch.sigmoid(logits)

        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_class_weight)
        if self.aux_weight > 0:
            aux_gru_logits = self.aux_gru(gru_base)[:, 0]
            aux_trans_logits = self.aux_trans(trans_base)[:, 0]
            aux_loss = (
                F.binary_cross_entropy_with_logits(aux_gru_logits, labels, pos_weight=self.pos_class_weight)
                + F.binary_cross_entropy_with_logits(aux_trans_logits, labels, pos_weight=self.pos_class_weight)
            )
            loss = loss + self.aux_weight * aux_loss
        return loss


class GatedSumFusionHead(nn.Module):
    """Gated sum fusion used by all top-level `+` model variants."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        mid_dim = max(hidden_dim // 2, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, branch1_common: torch.Tensor, branch2_common: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([branch1_common, branch2_common], dim=-1)
        gate = self.gate(gate_input)
        fused = gate * branch1_common + (1.0 - gate) * branch2_common
        return self.classify_features(fused)

    def classify_features(self, fused: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused)[:, 0]


class ConcatFusionHead(nn.Module):
    """Simple concatenation fusion used by the `no_fusion` ablation."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        mid_dim = max(hidden_dim, 1)
        low_dim = max(hidden_dim // 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, low_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(low_dim, 1),
        )

    def forward(self, branch1_common: torch.Tensor, branch2_common: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([branch1_common, branch2_common], dim=-1)
        return self.classifier(fused)[:, 0]


class DualBranchGatedSumModel(nn.Module):
    """Generic dual-branch wrapper that mirrors lab4 gated_sum behavior."""

    def __init__(self, args, primary_model_type: str) -> None:
        super().__init__()
        if primary_model_type not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown model_type: {args.model_type}")

        _apply_single_plus_fusion_defaults(args)

        self.primary_model_type = primary_model_type
        self.branch1_model = BACKBONE_REGISTRY[primary_model_type](args)
        self.register_buffer('pos_class_weight', torch.tensor(float(args.pos_class_weight)))

        if primary_model_type in {'gru', 'grud'}:
            if primary_model_type == 'gru':
                self.branch1_keys = ['x_t', 'm_t', 'seq_len', 'demo']
            else:
                self.branch1_keys = ['x_t', 'm_t', 'delta_t', 'seq_len', 'demo']
            self.branch2_keys = ['observed_data', 'observed_mask', 'observed_tp', 'tau', 'demo']
            self.branch2_model = TransformerBranch(args)
        else:
            self.branch1_keys = None
            self.branch2_keys = ['x_t', 'm_t', 'seq_len', 'demo']
            self.branch2_model = GRUBranch(args)

        branch1_dim = self.branch1_model.last_hid_dim
        branch2_dim = self.branch2_model.last_hid_dim
        fusion_hid_dim = getattr(args, 'fusion_hid_dim', 128)
        fusion_dropout = getattr(args, 'fusion_dropout', 0.3)

        self.branch1_proj = nn.Linear(branch1_dim, fusion_hid_dim)
        self.branch2_proj = nn.Linear(branch2_dim, fusion_hid_dim)
        self.branch1_aux = nn.Linear(branch1_dim, 1)
        self.branch2_aux = nn.Linear(branch2_dim, 1)
        self.lambda_aux = float(getattr(args, 'lambda_aux', 0.2))
        self.plus_ablation_mode = getattr(args, 'plus_ablation_mode', 'full')
        valid_ablation_modes = {'full', 'no_gru', 'no_transformer', 'no_fusion'}
        if self.plus_ablation_mode not in valid_ablation_modes:
            raise ValueError(
                f"Unsupported plus_ablation_mode={self.plus_ablation_mode!r}. "
                f"Expected one of {sorted(valid_ablation_modes)}."
            )
        self.fusion_head = GatedSumFusionHead(fusion_hid_dim, fusion_dropout)
        self.concat_fusion_head = ConcatFusionHead(fusion_hid_dim, fusion_dropout)
        self._apply_ablation_freeze()

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def _apply_ablation_freeze(self) -> None:
        if self.plus_ablation_mode == 'no_gru':
            if self.primary_model_type in {'gru', 'grud'}:
                self._freeze_module(self.branch1_model)
            else:
                self._freeze_module(self.branch2_model)
        elif self.plus_ablation_mode == 'no_transformer':
            if self.primary_model_type in {'gru', 'grud'}:
                self._freeze_module(self.branch2_model)
            else:
                self._freeze_module(self.branch1_model)

    def _build_branch1_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.branch1_keys is not None:
            return {key: batch[key] for key in self.branch1_keys}
        return {
            key: value for key, value in batch.items()
            if key not in {'x_t', 'm_t', 'seq_len'}
        }

    def forward(self, labels=None, **batch):
        branch1_batch = self._build_branch1_batch(batch)
        branch2_batch = {key: batch[key] for key in self.branch2_keys}

        branch1_features = self.branch1_model(**branch1_batch)
        branch2_features = self.branch2_model(**branch2_batch)
        branch1_common = self.branch1_proj(branch1_features)
        branch2_common = self.branch2_proj(branch2_features)
        branch1_logits = self.branch1_aux(branch1_features)[:, 0]
        branch2_logits = self.branch2_aux(branch2_features)[:, 0]

        if self.plus_ablation_mode == 'no_fusion':
            logits = self.concat_fusion_head(branch1_common, branch2_common)
        else:
            logits = self.fusion_head(branch1_common, branch2_common)

        if labels is None:
            return torch.sigmoid(logits)

        task_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_class_weight,
        )
        if self.plus_ablation_mode == 'no_fusion' or self.lambda_aux <= 0:
            return task_loss

        aux_loss = F.binary_cross_entropy_with_logits(
            branch1_logits,
            labels,
            pos_weight=self.pos_class_weight,
        ) + F.binary_cross_entropy_with_logits(
            branch2_logits,
            labels,
            pos_weight=self.pos_class_weight,
        )
        return task_loss + self.lambda_aux * aux_loss


class ModelWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_type = args.model_type.lower()
        self.branch2_enable = getattr(args, 'branch2_enable', False)
        self.model_type = model_type

        if self.branch2_enable:
            self.dual_branch_model = DualBranchGatedSumModel(args, model_type)
            return

        if model_type not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown model_type: {args.model_type}")

        self.backbone = BACKBONE_REGISTRY[model_type](args)
        self.register_buffer('pos_class_weight', torch.tensor(args.pos_class_weight))
        last_hid_dim = self.backbone.last_hid_dim
        self.classifier = nn.Sequential(
            nn.Linear(last_hid_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )

    def forward(self, labels=None, **batch):
        if self.branch2_enable:
            return self.dual_branch_model(labels=labels, **batch)

        features = self.backbone(**batch)
        logits = self.classifier(features)[..., -1]
        if labels is not None:
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_class_weight
            )
        return torch.sigmoid(logits)
