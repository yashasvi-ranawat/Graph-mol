import os
import pickle
import torch
import numpy as np
from torch.utils import data
import pytorch_lightning as pl


class Dataset(data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, list_IDs, targ_label: str) -> None:
        #  Initialization
        self.list_IDs: str = list_IDs
        self.targ_label: str = targ_label

    def __len__(self) -> int:
        # Denotes the total number of samples
        return len(self.list_IDs)

    def __getitem__(self, indx: int) -> [torch.tensor, torch.tensor]:
        # Generates one sample of data
        # Select sample

        index = self.list_IDs[indx]
        with open(f"data/processed/{index}_inp.pkl", "rb") as fio:
            x = pickle.load(fio)
        with open(f"data/processed/{index}_targ.pkl", "rb") as fio:
            y = pickle.load(fio)[self.targ_label]

        return x.type("torch.FloatTensor"), y.type("torch.FloatTensor")


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        test_ratio: float = 0.1,  # ratio out of total samples for testing
        train_ratio: float = 1.0,  # ratio out of non-test samples for training
        val_ratio: float = 0.2,  # ratio out of training samples for validation
        batch_size: int = 32,
        num_workers: int = 4,
     ) -> None:
        super().__init__()

        # Parameters
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        # Datasets
        if "partition.pkl" in os.listdir("data"):
            with open("data/partition.pkl", "rb") as f:
                partition: dict[str, list[str]] = pickle.load(f)
        else:
            ls: list[str] = os.listdir("data/processed")
            ids: list[str] = []
            for i in ls:
                if i.endswith("-inp.pkl"):
                    _ = i.split("-")[:-1]
                    ids.append("-".join(_))

            ids = np.random.permutation(ids).tolist()
            test_ind: int = int(len(ids) * (1 - test_ratio))
            train_ind: int = int(test_ind * train_ratio)
            val_ind: int = int(train_ind * (1 - val_ratio))
            partition: dict[str, list[str]] = {
                "train": ids[:val_ind],
                "val": ids[val_ind:train_ind],
                "test": ids[test_ind:],
            }
            with open("data/partition.pkl", "wb") as f:
                pickle.dump(partition, f)

        partition["train"] = partition["train"][
          :int(len(partition["train"]) * train_ratio)
        ]
        partition["val"] = partition["val"][: int(len(partition["val"]) * train_ratio)]
        dataset_sizes = {x: len(partition[x]) * 6 for x in ["train", "val", "test"]}

        print(
            "# Data: train = {}\n"
            "#       validation = {}\n"
            "#       test = {}".format(
                dataset_sizes["train"], dataset_sizes["val"], dataset_sizes["test"]
            )
        )

        self.data_set: dict[str, Dataset] = {
            x: Dataset(partition[x]) for x in ["train", "val", "test"]
        }

    def train_dataloader(self):
        return data.DataLoader(
            self.data_set["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.data_set["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.data_set["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
