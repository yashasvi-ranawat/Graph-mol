import torch
import torch.optim as optim
import pytorch_lightning as pl
from .graph_transformer import GraphTransformer


MODELS = {
    "graph_transformer": {
        "model": GraphTransformer,
        "parameters": ["in_channel", "n_channel", "n_heads", "n_layers"],
    }
}


def get_model(model_name: str, **kwargs):
    parameters = {}
    for params in MODELS[model_name]["parameters"]:
        parameters[params] = kwargs.get(params, None)

    return MODELS[model_name]["model"](**parameters)


class GraphModel(pl.LightningModule):
    def __init__(self, model_name: str, in_channel: int, lr=1e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.graph = get_model(model_name, in_channel=in_channel, **kwargs)
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

        G_loss = torch.mean(torch.abs(G - batch.G))
        gap_loss = torch.mean(torch.abs(gap - batch.gap))
        c_loss = torch.mean(torch.abs(charge - batch.c))

        loss = G_loss + gap_loss + c_loss

        # Logging
        self.log("loss", loss, batch_size=batch.G.shape[0])
        self.log("G_loss", G_loss, batch_size=batch.G.shape[0])
        self.log("gap_loss", gap_loss, batch_size=batch.G.shape[0])
        self.log("c_loss", c_loss, batch_size=batch.G.shape[0])
        return loss

    def validation_step(self, batch):
        G, gap, charge = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        G_loss = torch.mean(torch.abs(G - batch.G))
        gap_loss = torch.mean(torch.abs(gap - batch.gap))
        c_loss = torch.mean(torch.abs(charge - batch.c))

        loss = G_loss + gap_loss + c_loss

        self.log("val_loss", loss, batch_size=batch.G.shape[0])
        self.log("val_G_loss", G_loss, batch_size=batch.G.shape[0])
        self.log("val_gap_loss", gap_loss, batch_size=batch.G.shape[0])
        self.log("val_c_loss", c_loss, batch_size=batch.G.shape[0])

    def test_step(self, batch):
        G, gap, charge = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        G_loss = torch.mean(torch.abs(G - batch.G))
        gap_loss = torch.mean(torch.abs(gap - batch.gap))
        c_loss = torch.mean(torch.abs(charge - batch.c))

        loss = G_loss + gap_loss + c_loss

        self.log("test_loss", loss, batch_size=batch.G.shape[0])
