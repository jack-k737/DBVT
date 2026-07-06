from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout,
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCN(nn.Module):
    """Project-integrated TCN backbone returning feature vectors."""

    def __init__(self, args):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=args.V * 3,
            num_channels=[args.hid_dim] * args.num_layers,
            kernel_size=getattr(args, 'kernel_size', 3),
            dropout=args.dropout,
        )
        self.last_hid_dim = args.hid_dim * 2 if args.D > 0 else args.hid_dim
        self.demo_emb = None
        if args.D > 0:
            self.demo_emb = nn.Sequential(
                nn.Linear(args.D, args.hid_dim * 2),
                nn.Tanh(),
                nn.Linear(args.hid_dim * 2, args.hid_dim),
            )

    def forward(self, ts: torch.Tensor, demo: torch.Tensor | None = None) -> torch.Tensor:
        ts = ts.permute(0, 2, 1)
        ts_emb = self.tcn(ts)[:, :, -1]
        if self.demo_emb is None or demo is None:
            return ts_emb
        demo_emb = self.demo_emb(demo)
        return torch.cat((ts_emb, demo_emb), dim=-1)
