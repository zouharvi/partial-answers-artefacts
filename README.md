# Partial Answers as Artefacts

Initial (work-in-progress) exploration of Fusion of Partial Answers as Artefacts for Multi-Output Classification.
Authors: Vilém Zouhar, Edu Vallejo Arguinzoniz.

## Running the code

### Preparation

- With Python `>=3.9`, install dependencies as `pip3 install -r requirements.txt`. The code is expected to be run in a Linux environment with the standard tools installed (bash, autotools).
- Run `make data_all` to load the data into `data/final/clean.json` and prepare all other crafted data (may require up to 20GB of disk space)
- Run the following command to create a new test data `clean_COP25.json` based on `~/Downloads/COP25.filt3.sub.json`. It is necessary to have access to all the data in order to provide consistent filtering.
`./src/data/prepare_clean.py --data-in data/final/all.json --data-out data/final/clean_COP25.json --extra ~/Downloads/COP25.filt3.sub.json `

The filtering is explained in the accompanying paper.
In essence, it is about considering only non-sparse labels and samples with at least one label in the _subject_ and _geographic_ variables.

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

Run `make feature_dependency` to model the relationship between variables using logistic regression: `z = p(-,\xi; \theta), \xi \in y`.
The final output should look similar to:

![Variable dependency Logistic Regression](data/figures/feature_dependency_lr.png)

### Models

If the system has a compatible CUDA device visible to PyTorch, it will be used.
Otherwise, the model will train on CPU/RAM.
You may enforce the script to use CPU/RAM (e.g. because of memory limitations) by prefixing the launching command with `CUDA_VISIBLE_DEVICES=;`.
This can also be used to select which device you would like to utilize (if there are multiple), e.g. `CUDA_VISIBLE_DEVICES=2;` will use the third GPU device.

All the scripts use a fixed split into train/dev/test (-1/1000/1000) that is randomly sampled in-time.
Results for most of the experiments are reported with class-specific metrics (precision, recall, F1) as well as their aggregation (macro-/micro-avg, accuracy) in JSONs in [data/eval/](data/eval/).
The paper reports only accuracies for reasons explained there.

#### Main Model

The main model is the central piece of our exploration, note that it is not the goal of this paper to produce the best model. 
The model consists of a pretrained language model (BERT by default) that we fine-tune in slight variations of news article classification tasks. 
We explore two dimensions: single target classification vs joint classification and the effect of partial answers in classification performance. 

There are ready-to-use "make targets" to train and evaluate the model.
To train a model in a given classification target one should run `make train_{target}_0v1` where `{target}` is the classification target which to train the model for\*.
In the same way, to evaluate on a given target (after training) we provide "make targets" `make eval_{target}_0v1`.
We also provide "make targets" to train and eval on all classification targets, these are `make train_0v1` and `make eval_0v1` respectively.
Finally, there is `make traineval_0v1` which is a combination of the last two (it trains all models and evaluates them).

\* The available classification targets are: `newspaper`, `ncompas`, `ncountry`, `month`, `year`, `subject` and `geographic` for single
target learning, and `all` for joint learning.

The "make targets" make use of two python scripts under the hood which may be used for fine-grain control:
`src/model_main/train_lm_classifier.py` to train the model and `src/model_main/eval_lm_classifier.py` to evaluate it.
Both scripts take a formatted dataset (that `make data_all` generates) and the name of the classification target as input.
The training script produces a checkpoint as output (in `data/models` by default) which is then used by the eval script. 
The eval script produces a JSON file with the results (in `data/eval` by default).
For added detail refer to the help in those scripts (by running them with the `--help` attribute).

##### Joint model

Additionally, we provide the possibility to download a checkpoint for the joint model (instead of training it with `make train_all_0v1`).
To do that run `make get_all_0v1`

After obtaining the model one may want to evaluate the model on new data to do that run the following script for evaluation of the joint model on prepared test data:

```
python3 src/model_main/eval_lm_classifier.py \
    --input data/final/clean_COP25.json \
    -mp data/models/bert_cls_1_uniform_2_shallow_512_body_all_clean.pt \
    -to all \
    -ts all
```

Remember that the new data need to be adequately pre-processed (see [Preparation](#preparation)).

#### Baseline Models

We provide four kinds of baseline models: Naive Bayes, Logistic Regression, SVM-based and LSTM-based.
The first three are invoked via the following scripts.
The input can be either `headline`, `body` or `both`.
The output is one of the variables (single).
If the output variable is multi-class then the models are automatically adapted to support scoring based on probabilities.
The vectorizer can be either `bow` for the absolute baseline or `tfidf` for a better adapted version.
The following models are supported `nb` (naive Bayes, no multioutput), `lr` (logistic regression), `svc` (support vector machine with RBF kernel), `linear_svc` (support vector machine with the linear kernel).
No that `svc` may take up to an hour to run and also the same applies for models for predicting _subject_ or _geographic_ because internally many more models need to be trained.

```
python3 ./src/model_baseline/svm.py --model linear_svc -ti both --vectorizer tfidf
```

The LSTM based model utilizes word embeddings of the words in the headline joined with the first 20 words of the body.
This is processed by a layer of bidirectional LSTM and joined with a heavily filtered (dropout 75%) TF-IDF vector of the article body.
To run, you need to have a prepared GloVe embedding.
This can be done automatically via `make get_glove` (we are using 200-dimensional ones).

```
python3 ./src/model_baseline/lstm.py -to ncompas
```

#### Meta Model

The meta-model first needs generated data from the main model.
Currently, the model can work only with the _month_ variable which is also explored in the paper.
In theory, there are no limitations to the variable selection and _month_ is chosen because of the highest improvement with artefact fusion.
Multi-output is not implemented.

The following code trains a model which takes artefact signature on input together with true posterior and hidden state.

```
# train primary model on 50% dropped out artefacts data
python3 ./src/model_main/train_lm_classifier.py -to month -ti craft -i data/final/Rv1_50_m.json
# get data from primary model
python3 ./src/model_main/eval_lm_classifier_artefacts.py -mp data/models/bert_cls_1_2_shallow_256_craft_month_Rv1_50_m.pt -to month -o data/misc/meta_{}.pkl
# train & evaluate meta-model
python3 ./src/model_meta/main.py -i data/misc/meta_month.pkl -cm indiv -pm true --model standard
```

The following model combinations are valid.
See the paper for an explanation:

- `-cm {no,indiv,multi} -pm {true,frozen} --model standard`
- `-cm {no,indiv} -pm frozen --model joint`

## Repository structure

The source code is located in `src/`.
Scripts that produce figures are located in `src/figures/`.
All publicly exposed source files contain help messages which can be invoked with `-h` or `--help`.
Directory tree inside `data/` is automatically created when running `make data_all`.
Result reports for our experiments are in `data/eval/`.