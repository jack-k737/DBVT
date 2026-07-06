"""
Lightweight Neural CDE backbone without external torchcde dependency.

The model uses piecewise-linear controls over hourly aggregated observations and
Euler integration:
    z_{t+1} = z_t + f(z_t) dX_t
where X_t concatenates time, zero-imputed values, and observation masks.
"""

import torch
import torch.nn as nn


class VectorField(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, vf_hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.extend([
                nn.Linear(in_dim, vf_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = vf_hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim * input_dim))
        self.net = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size = hidden.shape[0]
        out = self.net(hidden)
        return out.view(batch_size, self.hidden_dim, self.input_dim)


class NeuralCDE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_value_dim = args.V
        self.demo_dim = getattr(args, "D", 0)

        hidden_dim = getattr(args, "hid_dim", 64)
        num_layers = getattr(args, "num_layers", 2)
        vf_hidden_dim = getattr(args, "vf_hid_dim", hidden_dim)
        dropout = getattr(args, "dropout", 0.1)

        control_dim = 1 + 2 * self.input_value_dim
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim

        self.initial = nn.Sequential(
            nn.Linear(control_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.vector_field = VectorField(
            hidden_dim=hidden_dim,
            input_dim=control_dim,
            vf_hidden_dim=vf_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        if self.demo_dim > 0:
            self.demo_proj = nn.Sequential(
                nn.Linear(self.demo_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.last_hid_dim = hidden_dim * 2
        else:
            self.demo_proj = None
            self.last_hid_dim = hidden_dim

    def forward(self, x, mask, t, demo):
        # Control path: [time, values, mask]
        control = torch.cat([t.unsqueeze(-1), x * mask, mask], dim=-1)
        hidden = self.initial(control[:, 0, :])

        if control.shape[1] > 1:
            delta_control = control[:, 1:, :] - control[:, :-1, :]
            lengths = (mask.sum(dim=-1) > 0).sum(dim=1).clamp(min=1)
            for step in range(delta_control.shape[1]):
                active = (step < (lengths - 1)).float().unsqueeze(-1)
                vf = self.vector_field(hidden)
                increment = torch.bmm(vf, delta_control[:, step, :].unsqueeze(-1)).squeeze(-1)
                updated = hidden + increment
                hidden = active * updated + (1.0 - active) * hidden

        if self.demo_proj is not None:
            demo_repr = self.demo_proj(demo)
            hidden = torch.cat([hidden, demo_repr], dim=-1)

        return hidden
