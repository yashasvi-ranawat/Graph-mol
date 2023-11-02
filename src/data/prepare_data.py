import os
from ase.atoms import Atoms
from tqdm import tqdm
from ase.neighborlist import NeighborList, natural_cutoffs
import pickle
import numpy as np


def read_xyz(filename: str) -> Atoms:
    with open(filename, "r") as fio:
        data = fio.readlines()

    symbols: list[str] = []
    positions: list[list[float, float, float]] = []
    charge: list[float] = []

    tot_atoms = int(data[0].strip())
    comment = data[1].split()
    gap = float(comment[9])
    free_energy = float(comment[15])
    for i in range(tot_atoms):
        line = data[2 + i].replace("*^", "e").split()
        symbols.append(line[0])
        positions.append([float(line[1]), float(line[2]), float(line[3])])
        charge.append(float(line[4]))

    return Atoms(
        symbols=symbols,
        positions=positions,
        charges=charge,
        info={"gap": gap, "G": free_energy},
    )


def prepare_data(raw_dir: str, indx: int) -> dict:
    filename = f"dsgdb9nsd_{indx:0>6}.xyz"
    atoms: Atoms = read_xyz(os.path.join(raw_dir, filename))
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, skin=0.2, self_interaction=False, bothways=True)
    nl.update(atoms)
    neighbor_matrix = nl.get_connectivity_matrix().toarray()
    bonds = []
    for i, array in enumerate(neighbor_matrix):
        neighbor_list = np.where(array)[0]
        for j in neighbor_list:
            if i < j:
                dist = np.sqrt(
                    np.sum((atoms.positions[i, :] - atoms.positions[j, :]) ** 2)
                )
                bonds.append([i, j, dist])
    return {
        "G": atoms.info["G"],
        "gap": atoms.info["gap"],
        "c": atoms.get_initial_charges(),
        "bonds": bonds,
        "z": atoms.numbers,
    }


def main() -> None:
    raw_dir = os.path.join("data", "raw")
    folder = os.path.join("data", "processed")
    indices = [_ for _ in range(1, 133886)]
    for indx in tqdm(indices):
        dict_ = prepare_data(raw_dir, indx)
        with open(os.path.join(folder, f"{indx:0>6}.pkl"), "wb") as fio:
            pickle.dump(dict_, fio)


if __name__ == "__main__":
    main()
