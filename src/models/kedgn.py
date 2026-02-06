# -*- coding:utf-8 -*-
"""
KEDGN: Knowledge-Enhanced Dynamic Graph Network
Adapted for STraTS framework
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import os


class Value_Encoder(nn.Module):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(Value_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = rearrange(x, 'b l k -> b l k 1')
        x = self.encoder(x)
        return x


class Time_Encoder(nn.Module):
    def __init__(self, embed_time, var_num):
        super(Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.var_num = var_num
        self.linear = nn.Linear(1, 1)

    def forward(self, tt):
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:  # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')

        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1)  # [B,L,1,D]
        return out


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class MLP_Param(nn.Module):
    def __init__(self, input_size, output_size, query_vector_dim):
        super(MLP_Param, self).__init__()
        self.W_1 = nn.Parameter(torch.FloatTensor(query_vector_dim, input_size, output_size))
        self.b_1 = nn.Parameter(torch.FloatTensor(query_vector_dim, output_size))

        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.b_1)

    def forward(self, x, query_vectors):
        W_1 = torch.einsum("nd, dio->nio", query_vectors, self.W_1)
        b_1 = torch.einsum("nd, do->no", query_vectors, self.b_1)
        x = torch.squeeze(torch.bmm(x.unsqueeze(1), W_1)) + b_1
        return x


class AGCRNCellWithMLP(nn.Module):
    def __init__(self, input_size, query_vector_dim):
        super(AGCRNCellWithMLP, self).__init__()
        self.update_gate = MLP_Param(2 * input_size + 1, input_size, query_vector_dim)
        self.reset_gate = MLP_Param(2 * input_size + 1, input_size, query_vector_dim)
        self.candidate_gate = MLP_Param(2 * input_size + 1, input_size, query_vector_dim)

    def forward(self, x, h, query_vectors, adj, nodes_ind):
        combined = torch.cat([x, h], dim=-1)
        combined = torch.matmul(adj, combined)
        r = torch.sigmoid(self.reset_gate(combined[nodes_ind], query_vectors))
        u = torch.sigmoid(self.update_gate(combined[nodes_ind], query_vectors))
        h[nodes_ind] = r * h[nodes_ind]
        combined_new = torch.cat([x, h], dim=-1)
        candidate_h = torch.tanh(self.candidate_gate(combined_new[nodes_ind], query_vectors))
        return (1 - u) * h[nodes_ind] + u * candidate_h


class VSDGCRNN(nn.Module):
    def __init__(self, d_in, d_model, num_of_nodes, var_plm_rep_tensor, 
                 rarity_alpha=0.5, query_vector_dim=5, node_emb_dim=8, plm_rep_dim=768):
        super(VSDGCRNN, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_of_nodes = num_of_nodes
        self.gated_update = AGCRNCellWithMLP(d_model, query_vector_dim)
        self.rarity_alpha = rarity_alpha
        self.rarity_W = nn.Parameter(torch.randn(num_of_nodes, num_of_nodes))
        self.relu = nn.ReLU()
        self.projection_f = MLP(plm_rep_dim, 2 * d_model, query_vector_dim)
        self.projection_g = MLP(plm_rep_dim, 2 * d_model, node_emb_dim)
        # Register PLM representations as buffer (not trainable)
        self.register_buffer('var_plm_rep_tensor', var_plm_rep_tensor)

    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, observed_mask, lengths, avg_interval):
        batch, steps, nodes, features = obs_emb.size()
        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)
        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)
        output = torch.zeros_like(h)
        nodes_initial_mask = torch.zeros(batch, nodes).to(device)

        var_total_obs = torch.sum(observed_mask, dim=1)
        var_plm_rep_tensor = repeat(self.var_plm_rep_tensor, "n d -> b n d", b=batch)

        query_vectors = self.projection_f(var_plm_rep_tensor)

        node_embeddings = self.projection_g(var_plm_rep_tensor)
        normalized_node_embeddings = F.normalize(node_embeddings, p=2, dim=2)
        adj = torch.softmax(torch.bmm(normalized_node_embeddings, normalized_node_embeddings.permute(0, 2, 1)), dim=-1)

        for step in range(int(torch.max(lengths).item())):

            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)
            cur_obs = obs_emb[:, step]
            cur_mask = observed_mask[:, step]
            cur_obs_var = torch.where(cur_mask)
            nodes_initial_mask[cur_obs_var] = 1
            nodes_need_update = cur_obs_var
            cur_avg_interval = avg_interval[:, step]
            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)
            rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))

            if nodes_need_update[0].shape[0] > 0:
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                h[nodes_need_update] = self.gated_update(
                    torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                    h, query_vectors[nodes_need_update], cur_adj, nodes_need_update)

            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(lengths).item()) - 1:
                return output

        return output


class KEDGN(nn.Module):
    """
    KEDGN model adapted for STraTS framework.
    
    Expected input format (from dataset.get_batch_kedgn):
        - arr: [B, T, V] - time series values
        - mask: [B, T, V] - observation mask
        - time: [B, T, V] - timestamps
        - interval: [B, T, V] - average interval features
        - length: [B, 1] - sequence lengths
        - demo: [B, D] - static/demographic features
        - labels: [B] - binary labels
    """
    def __init__(self, args):
        super(KEDGN, self).__init__()
        
        hidden_dim = args.hid_dim
        num_of_variables = args.V
        num_of_timestamps = args.T
        d_static = args.D
        query_vector_dim = getattr(args, 'query_vector_dim', 5)
        node_emb_dim = getattr(args, 'node_emb_dim', 16)
        rarity_alpha = getattr(args, 'rarity_alpha', 1.0)
        plm_rep_dim = getattr(args, 'plm_rep_dim', 768)
        node_enc_layer = getattr(args, 'node_enc_layer', 2)
        
        # Load or create PLM representations
        var_plm_rep_path = f'../data/kedgn_embeddings/{args.dataset}_var_rep.pt'   
        if os.path.exists(var_plm_rep_path):
            var_plm_rep_tensor = torch.load(var_plm_rep_path, map_location='cpu',weights_only=True).clone()
        else:
            # Initialize random PLM representations if not provided
            var_plm_rep_tensor = torch.randn(num_of_variables, plm_rep_dim)
        
        self.num_of_variables = num_of_variables
        self.num_of_timestamps = num_of_timestamps
        self.hidden_dim = hidden_dim
        self.pos_class_weight = args.pos_class_weight
        
        self.adj = nn.Parameter(torch.ones(size=[num_of_variables, num_of_variables]))
        self.value_enc = Value_Encoder(output_dim=hidden_dim)
        self.abs_time_enc = Time_Encoder(embed_time=hidden_dim, var_num=num_of_variables)
        self.obs_tp_enc = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                                 num_layers=node_enc_layer, batch_first=True, bidirectional=False)
        self.obs_enc = nn.Sequential(
            nn.Linear(in_features=6 * hidden_dim, out_features=hidden_dim),
            nn.ReLU()
        )
        self.type_emb = nn.Embedding(num_of_variables, hidden_dim)
        self.GCRNN = VSDGCRNN(d_in=hidden_dim, d_model=hidden_dim,
                              num_of_nodes=num_of_variables, 
                              var_plm_rep_tensor=var_plm_rep_tensor,
                              rarity_alpha=rarity_alpha,
                              query_vector_dim=query_vector_dim, 
                              node_emb_dim=node_emb_dim,
                              plm_rep_dim=plm_rep_dim)
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.d_static = d_static
        
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_variables)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_variables * 2, 200),
                nn.ReLU(),
                nn.Linear(200, 2))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_variables, 200),
                nn.ReLU(),
                nn.Linear(200, 2))

    def forward(self, arr, mask, time, interval, length, demo, labels=None):
        """
        Forward pass for KEDGN model.
        
        Args:
            arr: [B, T, V] - time series values (normalized)
            mask: [B, T, V] - observation mask (1 = observed, 0 = missing)
            time: [B, T, V] - timestamps for each observation
            interval: [B, T, V] - average interval features
            length: [B, 1] - sequence lengths
            demo: [B, D] - static/demographic features
            labels: [B] - binary labels (optional, for computing loss)
        
        Returns:
            If labels provided: loss (scalar)
            Otherwise: logits [B, 2]
        """
        b, t, v = arr.shape
        observed_data = arr
        observed_mask = mask
        
        value_emb = self.value_enc(observed_data) * observed_mask.unsqueeze(-1)
        abs_time_emb = self.abs_time_enc(time) * observed_mask.unsqueeze(-1)
        type_emb = repeat(self.type_emb.weight, 'v d -> b v d', b=b)
        structure_input_encoding = (value_emb + abs_time_emb + repeat(type_emb, 'b v d -> b t v d', t=t)) * observed_mask.unsqueeze(-1)

        last_hidden_state = self.GCRNN(structure_input_encoding, observed_mask, length, interval)

        if self.d_static != 0:
            static_emb = self.emb(demo)
            logits = self.classifier(torch.cat([torch.sum(last_hidden_state, dim=-1), static_emb], dim=-1))
        else:
            logits = self.classifier(torch.sum(last_hidden_state, dim=-1))
        
        if labels is not None:
            # Compute weighted cross entropy loss
            weight = torch.tensor([1.0, self.pos_class_weight], device=labels.device, dtype=logits.dtype)
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(logits, labels.long())
            return loss
        
        # Return probability of positive class (for evaluation)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]

