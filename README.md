# Graph-mol

<p align="left">
  <img src="https://github.com/yashasvi-ranawat/Graph-mol/actions/workflows/python-package.yml/badge.svg" alt="Package test status">
  <img src="https://github.com/yashasvi-ranawat/Graph-mol/actions/workflows/lint.yml/badge.svg" alt="Lint test status">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license">
</p>

> **_NOTE:_**  The repository is a work in progress.

Transformer Graph network for molecule property prediction. Uses [134k dataset](https://www.nature.com/articles/sdata201422/) to
train for free energy, band gap, and atomic charges.

## Training

Graph-mol uses Makefile to perform training. Data is fetched and prepared using:


```sh
make prepare-data
```

Training is performed using:

```sh
make train
```

The partition for testing, training and validation, and seed for training is preserved for consistent training.

## Jupyter notebooks

Jupyter notebooks to understand the Datamodules and the Graph model are present in `notebooks`. Jupyter lab can be run as:

```sh
make jupyter
```

## Deploy

The network can be deployed as a docker contatiner as:

```sh
make docker-build
make docker-run
```

## Contributing & TODO

To make contributions to the repository, check [Contributing.md](https://github.com/yashasvi-ranawat/Graph-mol/blob/master/Contributing.md)
