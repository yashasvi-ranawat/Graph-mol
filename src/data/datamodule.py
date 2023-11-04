import os
import pickle
import torch
import numpy as np
from torch.utils import data
from torch.nn.functional import one_hot
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl


class Dataset(data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, list_IDs, folder: str = "data") -> None:
        #  Initialization
        self.list_IDs: str = list_IDs
        self.folder: str = folder

    def __len__(self) -> int:
        # Denotes the total number of samples
        return len(self.list_IDs)

    def __getitem__(self, indx: int) -> [torch.tensor, torch.tensor]:
        # Generates one sample of data
        # Select sample
        class_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

        index = self.list_IDs[indx]
        with open(f"{self.folder}/processed/{index}", "rb") as fio:
            dict_ = pickle.load(fio)

        x = torch.concat(
            [one_hot(torch.tensor(class_map[i]), 5).unsqueeze(0) for i in dict_["z"]],
            dim=0,
        ).type("torch.FloatTensor")

        edge_index = torch.tensor(dict_["bonds"])[:, :2].t().to(torch.int64)

        edge_attr = 1 / torch.tensor(dict_["bonds"])[:, 2:].type(torch.FloatTensor) ** 2

        y = {}
        y["G"] = torch.tensor([dict_["G"]]).unsqueeze(0).type(torch.FloatTensor)
        y["gap"] = torch.tensor([dict_["gap"]]).unsqueeze(0).type(torch.FloatTensor)
        y["c"] = torch.tensor(dict_["c"]).unsqueeze(1).type(torch.FloatTensor)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            G=y["G"],
            gap=y["gap"],
            c=y["c"],
        )
        return T.ToUndirected()(data)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        test_ratio: float = 0.1,  # ratio out of total samples for testing
        train_ratio: float = 1.0,  # ratio out of non-test samples for training
        val_ratio: float = 0.2,  # ratio out of training samples for validation
        batch_size: int = 32,
        num_workers: int = 4,
        folder: str = "data",
    ) -> None:
        super().__init__()

        # Parameters
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        # Datasets
        if "partition.pkl" in os.listdir(folder):
            with open(f"{folder}/partition.pkl", "rb") as f:
                partition: dict[str, list[str]] = pickle.load(f)
        else:
            ls: list[str] = os.listdir(f"{folder}/processed")
            ids: list[str] = []
            for i in ls:
                if i.endswith(".pkl"):
                    ids.append(i)

            ids = np.random.permutation(ids).tolist()
            test_ind: int = int(len(ids) * (1 - test_ratio))
            train_ind: int = int(test_ind * train_ratio)
            val_ind: int = int(train_ind * (1 - val_ratio))
            partition: dict[str, list[str]] = {
                "train": ids[:val_ind],
                "val": ids[val_ind:train_ind],
                "test": ids[test_ind:],
            }
            with open(f"{folder}/partition.pkl", "wb") as f:
                pickle.dump(partition, f)

        # partition["train"] = partition["train"][
        #     : int(len(partition["train"]) * train_ratio)
        # ]
        # partition["val"] = partition["val"][: int(len(partition["val"]) * train_ratio)]
        dataset_sizes = {x: len(partition[x]) for x in ["train", "val", "test"]}

        print(
            "# Data: train = {}\n"
            "#       validation = {}\n"
            "#       test = {}".format(
                dataset_sizes["train"], dataset_sizes["val"], dataset_sizes["test"]
            )
        )

        self.data_set: dict[str, Dataset] = {
            x: Dataset(partition[x], folder) for x in ["train", "val", "test"]
        }

    def train_dataloader(self):
        return DataLoader(
            self.data_set["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_set["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_set["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
