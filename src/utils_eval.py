from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import itertools as it

def complete_evaluation(y_true, y_logits,evaluation_targets,target_names=list()):
    # Put label first
    y_true = y_true.T
    
    evaluations = it.starmap(eval_from_logits,
                             it.zip_longest(y_true,y_logits,target_names))
        
    return dict(zip(evaluation_targets,evaluations))
        
        
def eval_from_logits(y_true,logits,target_names=None):
    y_pred = np.argmax(logits,axis=1)
    return classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True,
        target_names=target_names)