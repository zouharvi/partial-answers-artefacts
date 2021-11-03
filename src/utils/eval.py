import utils

from sklearn.metrics import classification_report, accuracy_score
import numpy as np

import collections as col
import operator as op


def complete_evaluation(evaluation_targets: list, y_true, y_logits, target_names=list()):
    """
    @TODO comment
    """
    # Put label first
    y_true = y_true.T

    t_counts = col.Counter(evaluation_targets)

    # Single target
    single_targets = {k for k, v in t_counts.items() if v == 1}
    single_target_ids = [evaluation_targets.index(x) for x in single_targets]

    single_targets_y = list(zip(y_true, y_logits, target_names))
    single_targets_y = [single_targets_y[x] for x in single_target_ids]

    single_evaluations = [eval_from_logits(*x) for x in single_targets_y]
    evaluations = dict(zip(single_targets, single_evaluations))

    # Multi target
    multi_targets = set(t_counts.keys()) - single_targets
    for mt in multi_targets:
        mt_ids = [i for i, et in enumerate(evaluation_targets) if et == mt]
        mt_y_pred = [y_logits[x] for x in mt_ids]
        mt_y_true = y_true[mt_ids]

        evaluations[mt] = r_precission_from_logits(mt_y_true, mt_y_pred)

    return evaluations


def eval_from_logits(y_true, logits, target_names=None):
    """
    @TODO comment
    """
    y_pred = np.argmax(logits, axis=1)
    classification_dict = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True,
        zero_division=0,
        labels=list(range(len(target_names))),
        target_names=target_names,
    )

    # in some corner cases classification report does not return accuracy
    # this way we can force it
    classification_dict["accuracy"] = accuracy_score(y_true, y_pred)
    return classification_dict

def r_precission_from_logits(y_true, y_logits):
    """
    @TODO comment
    """
    # Batch first
    y_true = y_true.T
    y_logits = np.array([list(map(op.itemgetter(1), yl)) for yl in y_logits]).T

    return utils.rprec(y_true, y_logits)
