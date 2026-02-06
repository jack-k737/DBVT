"""
TCN (Temporal Convolutional Network) for time series classification.
Reference: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Remove trailing elements to maintain causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal convolutions and residual connection."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of temporal blocks with exponentially increasing dilation."""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    """
    TCN model for irregular time series classification.
    
    Input format:
        - ts: [B, T, V*3] concatenation of (values, obs_mask, delta)
        - demo: [B, D] static/demographic features
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # TCN backbone
        self.tcn = TemporalConvNet(
            num_inputs=args.V * 3,
            num_channels=[args.hid_dim] * args.num_layers,
            kernel_size=getattr(args, 'kernel_size', 3),
            dropout=args.dropout
        )
        
        # Demographics embedding
        self.demo_emb = nn.Sequential(
            nn.Linear(args.D, args.hid_dim * 2),
            nn.Tanh(),
            nn.Linear(args.hid_dim * 2, args.hid_dim)
        )
        
        # Classification head
        self.binary_head = nn.Linear(args.hid_dim * 2, 1)
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
        # TCN expects [B, C, T]
        ts = ts.permute(0, 2, 1)  # [B, V*3, T]
        ts_emb = self.tcn(ts)[:, :, -1]  # Take last timestep: [B, hid_dim]
        
        # Demographics embedding
        demo_emb = self.demo_emb(demo)
        
        # Concatenate and classify
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        logits = self.binary_head(ts_demo_emb)[:, 0]
        
        return self.binary_cls_final(logits, labels)
