from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.svm
import numpy as np

def experiment_confidence(X_full, Y_full, tf_idf, max_features, test_percentage, seed):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_full, Y_full,
        test_size=test_percentage,
        random_state=seed,
        shuffle=True
    )

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
    conf_test = np.average(np.abs(model.decision_function(X_test)))
    conf_train = np.average(np.abs(model.decision_function(X_train)))
    print("Average confidence test ", f"{conf_test:.2f}")
    print("Average confidence train", f"{conf_train:.2f}")

    Y_pred = model.predict(X_full)
    conf_full = np.abs(model.decision_function(X_full))

    conf_full_tt = np.average(
        [c for c, y_true, y_pred in zip(
            conf_full, Y_full, Y_pred) if y_true and y_pred]
    )
    conf_full_tf = np.average(
        [c for c, y_true, y_pred in zip(
            conf_full, Y_full, Y_pred) if y_true and not y_pred]
    )
    conf_full_ff = np.average(
        [c for c, y_true, y_pred in zip(
            conf_full, Y_full, Y_pred) if not y_true and not y_pred]
    )
    conf_full_ft = np.average(
        [c for c, y_true, y_pred in zip(
            conf_full, Y_full, Y_pred) if not y_true and y_pred]
    )
    print(f"TT {conf_full_tt:.2f}, TF {conf_full_tf:.2f}, FT {conf_full_ft:.2f}, FF {conf_full_ff:.2f}")