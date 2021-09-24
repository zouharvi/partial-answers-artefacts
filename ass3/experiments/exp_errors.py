from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.svm
import pickle

def experiment_errors(X_full, Y_full, tf_idf, max_features, data_out, test_percentage, seed):
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
    Y_pred = model.predict(X_test)
    
    with open(data_out, "wb") as f:
        pickle.dump(list(zip(X_test, Y_test, Y_pred)), f)
