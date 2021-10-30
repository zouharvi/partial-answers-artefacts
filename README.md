# Partial Answers as Artefacts

Work-in-Progress exploration of Fusion of Partial Answers as Artefacts for Multi-Task Classification.
Authors: Vilém Zouhar, Edu Vallejo Arguinzoniz.

## Running the code

### Preparation

- Install dependencies with `pip3 install -r requirements.txt`
- Run `make data_all` to load the data into `data/final/clean.json` and prepare all other crafted data (may require up to 20GB of disk space)

### Misc.

#### T-SNE

TODO

#### Variable Distribution

TODO

#### Feature Dependency

TODO

### Models

#### Main Model

TODO

#### Baseline Models

TODO

#### Meta Model

TODO

## TODO:
- document everything
- move as many scripts to makefile

## Repository structure

The source code is located in `src/`.
Scripts that produce figures are located in `src/figures/`.
Directory tree inside `data/` is automatically created when running `make data_all`.