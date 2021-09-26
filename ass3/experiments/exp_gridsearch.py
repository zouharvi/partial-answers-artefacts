import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, make_scorer


## Grid search params
C_PARAMS = dict(cls__C=np.exp2(np.linspace(-4,4,9)))

SVM_PARAMS = dict(cls__kernel=("linear","rbf","sigmoid"),**C_PARAMS)
SVM_PARAMS_POLY = dict(cls__kernel=("poly",),cls__degree=(2,3,4),**C_PARAMS)

L_SVM_PARAMS_L2 = dict(
    cls__penalty=("l2",),
    cls__loss=("hinge","squared_hinge"),
    **C_PARAMS)

L_SVM_PARAMS_L1 = dict(
    cls__penalty=("l1",),
    cls__loss=("squared_hinge",),
    cls__dual=(False,),
    **C_PARAMS)

L_SVR_PARAMS = dict(
    cls__loss=("epsilon_insensitive","squared_epsilon_insensitive"),
    **C_PARAMS)
    
## SVR scoring
def svr_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true,dtype=np.bool8)
    y_pred = np.asarray(y_pred >= 0.5,dtype=np.bool8)
    
    return accuracy_score(y_true=y_true,y_pred=y_pred)
    
    
SVR_SCORING = dict(
    acc=make_scorer(svr_accuracy,greater_is_better=True),
    mse=make_scorer(mean_squared_error,greater_is_better=False),
    mae=make_scorer(mean_absolute_error,greater_is_better=False))

def experiment_gridsearch(X, y, data_out):
    """
        Run gridsearch over the 4 SVM models (LinearSVC,LinearSVR,SVC and SVR) and
        relevant parameters for each (C regularization coefficient, kernel, loss function, 
        regularization function).
    """
    
    # Mumeric targets for regression
    y_num = np.array(y,dtype=np.int32)
    
    vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    
    # All base models
    svc = Pipeline([("vec",vec),("cls",SVC())])
    l_svc = Pipeline([("vec",vec),("cls",LinearSVC(max_iter=100000))])
    
    svr = Pipeline([("vec",vec),("cls",SVR())])
    l_svr = Pipeline([("vec",vec),("cls",LinearSVR(max_iter=100000))])
    
    # Grid search definitions
    gs_svc = GridSearchCV(
        svc,
        param_grid=[SVM_PARAMS,SVM_PARAMS_POLY],
        verbose=2,
        n_jobs=-1,
        cv=10)
    
    gs_l_svc = GridSearchCV(
        l_svc,
        param_grid=[L_SVM_PARAMS_L2,L_SVM_PARAMS_L1],
        verbose=2,
        n_jobs=-1,
        cv=10)
    
    gs_svr = GridSearchCV(
        svr,
        param_grid=[SVM_PARAMS,SVM_PARAMS_POLY],
        scoring=SVR_SCORING,
        refit="acc",
        verbose=2,
        n_jobs=-1,
        cv=10)
    
    gs_l_svr = GridSearchCV(
        l_svr,
        param_grid=L_SVR_PARAMS,
        scoring=SVR_SCORING,
        refit="acc",
        verbose=2,
        n_jobs=-1,
        cv=10)
    
    # Fit grid search
    gs_svc.fit(X,y)
    gs_l_svc.fit(X,y)
    gs_svr.fit(X,y)
    gs_l_svr.fit(X,y)
    
    # Merge results
    results = pd.DataFrame(gs_svc.cv_results_)
    results = results.append(pd.DataFrame(gs_l_svc.cv_results_))
    results = results.append(pd.DataFrame(gs_svr.cv_results_))
    results = results.append(pd.DataFrame(gs_l_svr.cv_results_))
    
    # Export results
    if data_out:
        results.to_csv(data_out, index=False)
    else:
        print(results)