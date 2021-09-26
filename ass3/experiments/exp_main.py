import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC


def experiment_main(X, y, data_out):
    vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    X = vec.fit_transform(X)
    
    svm = SVC()
    l_svm = LinearSVC(max_iter=100000)
    
    gs_svm = GridSearchCV(
        svm,
        param_grid=[
            dict(
                C=np.exp2(np.linspace(-4,4,9)),
                kernel=("linear","rbf","sigmoid"),
            ),
            dict(
                C=np.exp2(np.linspace(-4,4,9)),
                kernel=("poly",),
                degree=(2,3,4,5))],
        verbose=2,
        n_jobs=-1,
        cv=10)
    
    gs_l_svm = GridSearchCV(
        l_svm,
        param_grid=[
            dict(
                C=np.exp2(np.linspace(-4,4,9)),
                penalty=("l2",),
                loss=("hinge","squared_hinge")),
            dict(
                C=np.exp2(np.linspace(-4,4,9)),
                penalty=("l1",),
                loss=("squared_hinge",),
                dual=(False,)),
        ],
        verbose=2,
        n_jobs=-1,
        cv=10)
    
    gs_svm.fit(X,y)
    gs_l_svm.fit(X,y)
    
    results = pd.DataFrame(gs_svm.cv_results_)
    results_l = pd.DataFrame(gs_l_svm.cv_results_)
    
    results = results.append(results_l)
    if data_out:
        results.to_csv(data_out, index=False)