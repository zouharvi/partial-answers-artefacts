import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from utils.report_utils import complete_scoring, report_score

def experiment_main(
        X_train, y_train, 
        X_test, y_test,
        table_format="default"):
    """
        Compute performance metrics for the best performing model (according to GS)
        on a train-test split setting.
    """
    
    vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    vec.fit(X_train)
    
    # Vecotrize inputs
    X_train = vec.transform(X_train)
    X_test = vec.transform(X_test)
    
    # Fit model
    svm = SVC(kernel="rbf",C=2) # Best parameters found in grid-search
    svm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test)
    
    # Compute scores
    score = complete_scoring(y_pred, y_test)
    report_score(score,["Negative","Positive"],table_format=table_format)
    
    
    
    