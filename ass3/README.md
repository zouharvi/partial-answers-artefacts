This document provides a very brief introduction on how the code is structured, how to run the experiments and plot results.

## Experiments
This project provides the following experiments and they are all run through `./main.py --experiment {EXP_NAME}`.
We list the available experiment names together with their description (can be found in code in `experiments/`).

- `main`:
  Perform an extensive gridsearch on the SVM parameter space.
  Associated plotting file is `fig_grid_search.py`.
- `examples`:
  Generate a few examples with color, examine noise coefficients and craft an adversial example
  that changes the polarity of the classification.
- `errors`:
  Examines the relation between correct and incorrect classification and review length.
  Associated file for figures is `fig_errors.py`.
- `features`:
  Examine the SVM coefficients associated with input features (tokens or n-grams).
  Relevant plotting file is `fig_features.py`.
- `size`:
  Examine the effect of training data size and the number of input features on linear SVM performance.
  Associated plotting files are `fig_max_features.py` and `fig_train_size.py` (they both use this script's output).
- `confidence`:
  Examine how the model confidence varies among train/test and with prediction patterns,
  e.g. correct/incorrect classification.

Output of many of these experiments is a figure.
The design is such, that running these experiments with `--data-out {PATH_TO_OUTPUT_FILE}` will store the results which can then be used by a separate plotting script found in `figures/`.
For example:

```
./main.py --experiment features --tf-idf -i data/reviews.txt --data-out xfeatures_tfidf_m_10k.pkl --max-features 10000 --ngrams
./main.py --experiment features -i data/reviews.txt --data-out xfeatures_bow_m_10k.pkl --max-features 10000 --ngrams
./figures/fig_features.py --data-bow features_bow_m_10k.pkl --data-tfidf features_tfidf_m_10k.pkl -m
```

Most plotting scripts require a single argument, `--data-in`.
Please see the corresponding `--help` call to see what the specific requirements are. 

## Misc.

Run `./main.py --help` to get the following parameter overview:

```
usage: main.py [-h] [-i INPUT_FILE] [-t] [-tp TEST_PERCENTAGE] [--experiment EXPERIMENT] [-sh] [--seed SEED] [--max-features MAX_FEATURES] [--ngrams] [--data-out DATA_OUT]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        Input file with all data
  -t, --tf-idf          Use the TF-IDF vectorizer instead of Bag of Words
  -tp TEST_PERCENTAGE, --test-percentage TEST_PERCENTAGE
                        Percentage of the data that is used for the test set
  --experiment EXPERIMENT
                        Which experiment to run: main, examples, errors, confidence, size, features
  -sh, --shuffle        Shuffle data set before splitting in train/test
  --seed SEED           Seed used for shuffling
  --max-features MAX_FEATURES
                        Maximum number of features in the vectorizer
  --ngrams              Use ngrams for feature exploration
  --data-out DATA_OUT   Where to store experiment data
```