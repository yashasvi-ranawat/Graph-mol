import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
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
        x += self.dropout(self.norm0(self.transformer(x, edge_index, edge_attr)))
        x += self.dropout(self.norm1(self.linear(x)))
        return x


class GraphMol(nn.Module):
    def __init__(self, in_channel, n_channel, n_heads, n_layers):
        super(GraphMol, self).__init__()
        assert n_channel % n_heads == 0
        self.precondition = nn.Sequential(
            nn.Linear(in_channel, n_channel),
            nn.LeakyReLU(),
            nn.Linear(n_channel, n_channel),
            nn.LeakyReLU(),
        )
        self.trans = [TransformerShell(n_heads, n_channel) for _ in range(n_layers)]
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
        n_size = x.shape[0]
        x = self.precondition(x)
        for trans in self.trans:
            x = trans(x, edge_index, edge_attr)
        charge = self.charge(x)
        G = global_add_pool(self.G(x), batch)
        gap = global_add_pool(self.gap(x), batch)

        return G, gap, charge


class GraphModel(pl.LightningModule):
    def __init__(self, in_channel, n_channel, n_heads, n_layers, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.graph = GraphMol(in_channel, n_channel, n_heads, n_layers)
        self.example_input_array = (
            torch.zeros(2, in_channel).type("torch.FloatTensor"),
            torch.tensor([[0], [1]], dtype=torch.int64),
            torch.ones(1, 1).type("torch.FloatTensor"),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        z = self.graph(x, edge_index, edge_attr, batch)
        return z

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=0.97
        )  # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch):
        G, gap, charge = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        G_loss = (G - batch.G).abs().mean()
        gap_loss = (gap - batch.gap).abs().mean()
        c_loss = (charge - batch.c).abs().mean()

        loss = G_loss + gap_loss + c_loss

        # Logging
        self.log("loss", loss)
        self.log("G_loss", G_loss)
        self.log("gap_loss", gap_loss)
        self.log("c_loss", c_loss)
        return loss

    def validation_step(self, batch):
        G, gap, charge = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        G_loss = (G - batch.G).abs().mean()
        gap_loss = (gap - batch.gap).abs().mean()
        c_loss = (charge - batch.charge).abs().mean()

        loss = G_loss + gap_loss + c_loss

        self.log("val_loss", loss)
        self.log("val_G_loss", G_loss)
        self.log("val_gap_loss", gap_loss)
        self.log("val_c_loss", c_loss)

    def test_step(self, batch):
        G, gap, charge = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        G_loss = (G - batch.G).abs().mean()
        gap_loss = (gap - batch.gap).abs().mean()
        c_loss = (charge - batch.charge).abs().mean()

        loss = G_loss + gap_loss + c_loss

        self.log("test_loss", loss)
