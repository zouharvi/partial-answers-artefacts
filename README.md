# Partial Answers as Artefacts

Work-in-Progress exploration of Fusion of Partial Answers as Artefacts for Multi-Task Classification.
Authors: Vilém Zouhar, Edu Vallejo Arguinzoniz.

## Running the code

### Preparation

- With Python `>=3.9`, install dependencies as `pip3 install -r requirements.txt`
- Run `make data_all` to load the data into `data/final/clean.json` and prepare all other crafted data (may require up to 20GB of disk space)

### Misc.

#### T-SNE

To visualize BERT embeddings, run `make tsne`.
If you get out of memory errors reduce the batch size in `Makefile` or run it on your CPU (via masking your CUDA device).
Note that this may take from a few minutes up to an hour, depending on your configuration, to complete.
The end result should be a graph similar to this one:

![TSNE graph Newspaper country](data/figures/tsne_bert_512_ncountry.png)

#### Variable Distribution

Run `make balance` to generate overview of class distributions (in LaTeX table formatting).

#### Feature Dependency

Run `make feature_dependency` to model the relationship between variables using logistic regression: $\hat{z} = p(-,\xi;\, \theta),\,\, \xi \in y$.
The final output should look like:

![Variable dependency Logistic Regression](data/figures/feature_dependency_lr.png)

### Models

If the system has a compatible CUDA device visible to PyTorch, it will be used.
Otherwise the model will train on CPU/RAM.
You may enforce the script to use CPU/RAM (e.g. because of memory limitations) by prefixing the launching command with `CUDA_VISIBLE_DEVICES=;`.
This can also be used select which device you would like to utilize (if there are multiple), e.g. `CUDA_VISIBLE_DEVICES=2;` will use the thirds GPU device.

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