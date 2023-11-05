import sys

sys.path.append("../src/")
import torch
from models.graph_mol import GraphModel
from data.prepare_data import read_xyz, prepare_data
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch.nn.functional import one_hot


class_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}


def predict(file_: list[str]):
    """
    Args:
        :param file_: xyz file as a list of lines as read by readlines()
    """
    # graph
    atoms = read_xyz(None, file_)
    dict_ = prepare_data(atoms)
    x = torch.concat(
        [one_hot(torch.tensor(class_map[i]), 5).unsqueeze(0) for i in dict_["z"]],
        dim=0,
    ).type("torch.FloatTensor")

    edge_index = torch.tensor(dict_["bonds"])[:, :2].t().type(torch.int64)

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
    data = T.ToUndirected()(data)

    # model
    model = GraphModel.load_from_checkpoint("model.ckpt")
    model.eval()

    # predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data.to(device)
    with torch.no_grad():
        G, gap, charge = model(data.x, data.edge_index, data.edge_attr)
        G = G.detach().cpu()
        gap = gap.detach().cpu()
        charge = charge.detach().cpu()

    # errors
    G_loss = torch.mean(torch.abs(G - y["G"])).tolist()
    gap_loss = torch.mean(torch.abs(gap - y["gap"])).tolist()
    charge_loss = torch.mean(torch.abs(charge - y["c"])).tolist()

    return {
        "G": G.tolist()[0][0],
        "gap": gap.tolist()[0][0],
        "charge": charge.squeeze(1).numpy(),
        "G_loss": G_loss,
        "gap_loss": gap_loss,
        "charge_loss": charge_loss,
    }
