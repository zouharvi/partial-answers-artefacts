from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.svm
from sklearn.metrics import accuracy_score
import pickle

def experiment_size(X_full, Y_full, data_out, seed):
    data = {}
    for tf_idf in [False, True]:
        for stop_words in [None, "english"]:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_full, Y_full,
                test_size=0.1,
                random_state=seed,
                shuffle=True
            )
            data_vec = {"train": [], "features": []}
            for max_features in [100, 500, 1000, 5*1000, 10*1000, 50*1000, 100*1000]:
                print("vec:", "tfidf" if tf_idf else "bow", "| max_features:", max_features)
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
                data_vec["features"].append((max_features, accuracy))

            for train_size in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5400]:
                print("vec:", "tfidf" if tf_idf else "bow", "| train_size:", train_size)
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
                Y_pred = model.predict(X_test)
                accuracy = accuracy_score(Y_test, Y_pred)
                data_vec["train"].append((train_size, accuracy))

            data[("tfidf" if tf_idf else "bow", stop_words)] = data_vec
    
    with open(data_out, "wb") as f:
        pickle.dump(data, f)