from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.svm
from sklearn.metrics import accuracy_score
import pickle

def experiment_size(X_full, Y_full, data_out, test_percentage, shuffle, seed):
    """
    Examine the effect of training data size and the number of input features on linear SVM performance.
    Associated plotting files are `fig_max_features.py` and `fig_train_size.py` (they both use this script's output).
    """
    assert(data_out is not None)

    data = {}

    # iterate over the space TFIDF-stopwords-{trainsize,max_features}
    # scikit's gridsearch could be used here though this seems more straightforward
    for tf_idf in [False, True]:
        for stop_words in [None, "english"]:
            # split the data
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_full, Y_full,
                test_size=test_percentage,
                random_state=seed,
                shuffle=shuffle,
            )
            data_vec = {"train": [], "features": []}

            # iterate over max features
            for max_features in [100, 500, 1000, 5*1000, 10*1000, 50*1000, 100*1000]:
                print("vec:", "tfidf" if tf_idf else "bow", "| max_features:", max_features)

                # define, fit and evaluate the model
                model = Pipeline([
                    ("vec",
                    TfidfVectorizer(
                        preprocessor=lambda x: x,
                        tokenizer=lambda x:x,
                        max_features=max_features,
                        stop_words=stop_words,
                    )
                    if tf_idf else
                    CountVectorizer(
                        preprocessor=lambda x: x,
                        tokenizer=lambda x:x,
                        max_features=max_features,
                        stop_words=stop_words,
                    ),
                    ),
                    ("svm", sklearn.svm.SVC(kernel="linear")),
                ])
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                accuracy = accuracy_score(Y_test, Y_pred)

                # store the result
                data_vec["features"].append((max_features, accuracy))

            # iterate over train size data
            for train_size in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5400]:
                print("vec:", "tfidf" if tf_idf else "bow", "| train_size:", train_size)
                if train_size > len(X_full)*(1-test_percentage):
                    raise Exception("Attempting to use training data which were reserved for testing")

                # define the model and fit it on the truncated data
                model = Pipeline([
                    ("vec",
                    TfidfVectorizer(
                        preprocessor=lambda x: x,
                        tokenizer=lambda x:x,
                        max_features=10000,
                        stop_words=stop_words,
                    )
                    if tf_idf else
                    CountVectorizer(
                        preprocessor=lambda x: x,
                        tokenizer=lambda x:x,
                        max_features=10000,
                        stop_words=stop_words,
                    ),
                    ),
                    ("svm", sklearn.svm.SVC(kernel="linear")),
                ])
                model.fit(X_train[:train_size], Y_train[:train_size])
                # evaluate on test and store the results
                Y_pred = model.predict(X_test)
                accuracy = accuracy_score(Y_test, Y_pred)
                data_vec["train"].append((train_size, accuracy))

            data[("tfidf" if tf_idf else "bow", stop_words)] = data_vec
    
    # store all the results in a file
    with open(data_out, "wb") as f:
        pickle.dump(data, f)