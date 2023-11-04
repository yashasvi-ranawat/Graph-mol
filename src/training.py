import os

import torch
from data.datamodule import DataModule
from models.graph_mol import GraphModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)


def main():
    # Select the device for training (use GPU if you have one)
    CHECKPOINT_PATH = "ckpts"
    MODEL_PATH = "app"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    data_module = DataModule(batch_size=32)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "model"),
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=60,
        # gradient_clip_val=0.1,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
            LearningRateMonitor("epoch"),
            TQDMProgressBar(),
        ],
    )
    # Check whether pretrained model exists. If yes, load it
    pretrained_filename = os.path.join(MODEL_PATH, "model.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(142)
        model = GraphModel(in_channel=5, n_channel=16, n_heads=8, n_layers=10, lr=1e-4)

    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
