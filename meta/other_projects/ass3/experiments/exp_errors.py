from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.svm
import numpy as np
import pickle

def experiment_errors(X_full, Y_full, tf_idf, max_features, data_out, test_percentage, shuffle, seed):
    """
    Examines the relation between correct and incorrect classification and review length.
    Associated file for figures is `fig_errors.py`.
    """
    
    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_full, Y_full,
        test_size=test_percentage,
        random_state=seed,
        shuffle=shuffle,
    )

    # define, train the model and make predictions on test
    model = Pipeline([
        ("vec",
         TfidfVectorizer(
             preprocessor=lambda x: x,
             tokenizer=lambda x:x, max_features=max_features,
         )
         if tf_idf else
         CountVectorizer(
             preprocessor=lambda x: x,
             tokenizer=lambda x:x, max_features=max_features,
         ),
         ),
        ("svm", sklearn.svm.SVC(kernel="linear")),
    ])
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # compute average lengths
    len_avg_t = np.average([len(x) for x,y_pred,y_true in zip(X_test, Y_pred, Y_test) if y_pred == y_true])
    len_avg_f = np.average([len(x) for x,y_pred,y_true in zip(X_test, Y_pred, Y_test) if y_pred != y_true])
    print(f"Average review length of a correct prediction:    {len_avg_t:.2f}")
    print(f"Average review length of an incorrect prediction: {len_avg_f:.2f}")

    # store predictions
    with open(data_out, "wb") as f:
        pickle.dump(list(zip(X_test, Y_test, Y_pred)), f)
