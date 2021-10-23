import utils

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import itertools as it
import collections as col
import functools as ft
import operator as op

def complete_evaluation(evaluation_targets : list , y_true, y_logits, target_names=list()):
    # Put label first
    y_true = y_true.T
    
    t_counts = col.Counter(evaluation_targets)
    
    # Single target
    single_targets = {k for k,v in t_counts.items() if v == 1}
    single_target_ids = list(map(evaluation_targets.index,single_targets))
    
    single_targets_y = list(it.zip_longest(y_true,y_logits,target_names))
    single_targets_y = list(map(single_targets_y.__getitem__,single_target_ids))
    
    single_evaluations = it.starmap(eval_from_logits,single_targets_y)
    evaluations = dict(zip(single_targets,single_evaluations))
    
    # Multi target
    multi_targets = set(t_counts.keys()) - single_targets
    for mt in multi_targets:
        mt_ids = [ i for i,et in enumerate(evaluation_targets) if et == mt]
        mt_y_pred = list(map(y_logits.__getitem__,mt_ids))
        mt_y_true = y_true[mt_ids]
        
        evaluations[mt] = r_precission_from_logits(mt_y_true,mt_y_pred)
    
    return evaluations
        
        
def eval_from_logits(y_true,logits,target_names=None):
    y_pred = np.argmax(logits,axis=1)
    return classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True,
        target_names=target_names)


def r_precission_from_logits(y_true,y_logits):
    # Batch first
    y_true = y_true.T
    y_logits = np.array([list(map(op.itemgetter(1),yl)) for yl in y_logits]).T
    
    return utils.rprec(y_true, y_logits)
    