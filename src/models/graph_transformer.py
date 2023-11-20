import torch
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import global_add_pool


class TransformerShell(nn.Module):
    def __init__(self, n_heads: int, n_channel: int, dropout: float = 0.1):
        super(TransformerShell, self).__init__()
        self.transformer = TransformerConv(
            n_channel,
            n_channel,
            heads=n_heads,
            dropout=dropout,
            edge_dim=1,
            concat=False,
        )
        self.linear = nn.Linear(n_channel, n_channel)
        self.norm0 = nn.LayerNorm(n_channel)
        self.norm1 = nn.LayerNorm(n_channel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = x + self.dropout(self.norm0(self.transformer(x, edge_index, edge_attr)))
        x = x + self.dropout(self.norm1(self.linear(x)))
        return x


class GraphTransformer(nn.Module):
    def __init__(self, in_channel, n_channel, n_heads, n_layers):
        super(GraphTransformer, self).__init__()
        self.precondition = nn.Sequential(
            nn.Linear(in_channel, n_channel),
            nn.LeakyReLU(),
            nn.Linear(n_channel, n_channel),
            nn.LeakyReLU(),
            nn.Linear(n_channel, n_channel),
        )
        self.trans = nn.ModuleList(
            [TransformerShell(n_heads, n_channel) for _ in range(n_layers)]
        )
        self.charge = nn.Sequential(
            nn.Linear(n_channel, n_channel),
            nn.LeakyReLU(),
            nn.Linear(n_channel, n_channel),
            nn.LeakyReLU(),
            nn.Linear(n_channel, 1),
        )
        self.G = nn.Sequential(
            nn.Linear(n_channel, n_channel),
            nn.LeakyReLU(),
            nn.Linear(n_channel, n_channel),
            nn.LeakyReLU(),
            nn.Linear(n_channel, 1),
        )
        self.gap = nn.Sequential(
            nn.Linear(n_channel, n_channel),
            nn.LeakyReLU(),
            nn.Linear(n_channel, n_channel),
            nn.LeakyReLU(),
            nn.Linear(n_channel, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.int64).to(x.device)
        x = self.precondition(x)
        for trans in self.trans:
            x = trans(x, edge_index, edge_attr)
        charge = self.charge(x)
        G = global_add_pool(self.G(x), batch)
        gap = global_add_pool(self.gap(x), batch)

        return G, gap, charge
