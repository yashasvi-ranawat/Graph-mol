import os
import argparse

from data.datamodule import DataModule
from models.graph_mol import GraphModel, MODELS
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)


MODEL_PARAMETERS = {
    "graph_transformer": {"n_channel": 32, "n_heads": 10, "n_layers": 10}
}


def main(model_name: str):
    # Select the device for training (use GPU if you have one)
    CHECKPOINT_PATH = "ckpts"
    MODEL_PATH = "app"

    data_module = DataModule(batch_size=32)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, f"model-{model_name}"),
        accelerator="gpu",
        max_epochs=60,
        # gradient_clip_val=0.1,
        callbacks=[
            ModelCheckpoint(mode="min", monitor="val_loss"),
            LearningRateMonitor("epoch"),
            TQDMProgressBar(),
        ],
    )
    # Check whether pretrained model exists. If yes, load it
    pretrained_filename = os.path.join(MODEL_PATH, f"model-{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(142)
        model = GraphModel(
            model_name, in_channel=5, lr=1e-4, **MODEL_PARAMETERS[model_name]
        )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument("model_name", type=str, choices=list(MODELS.keys()))
    args = parser.parse_args()
    main(args.model_name)
