import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from utils.report_utils import complete_scoring, report_score, avg_dict, dict_op

def experiment_cv_eval(
        X_full, y_full,
        table_format="default"):
    """
        Compute 10 fold cross-validated metrics of the best performing model (according to GS).
    """
    
    X_full, y_full = map(np.array,(X_full,y_full))
    
    vec = TfidfVectorizer(max_features=100000, ngram_range=(1,3),
        preprocessor=lambda x: x, tokenizer=lambda x: x)
    svm = SVC(kernel="rbf",C=2,gamma=0.6) # Best parameters found in grid-search
    
    
    k_fold = KFold(n_splits=10)
    
    scores = []
    for train_i, test_i in k_fold.split(X_full):
        # Reinstantiate to forget previous fold training
        classifier = Pipeline([('vec', vec), ('cls', svm)])

        # Perform the split with the corresponding indices
        X_train, Y_train = X_full[train_i], y_full[train_i]
        X_test, Y_test = X_full[test_i], y_full[test_i]

        # Fit classifier and make predictions
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)

        # Compute metrics and store them
        score = complete_scoring(Y_test, Y_pred)
        scores.append(score)
    
    #Average scores
    score = avg_dict(*scores)
    
    # Report scores
    report_score(score,["Negative","Positive"],table_format=table_format)