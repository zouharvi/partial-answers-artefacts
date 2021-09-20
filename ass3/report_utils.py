
# Utility libraries
from itertools import starmap
import operator
from typing import Any

# Scientific / Numeric libraries
import numpy as np


def format_auto_matrix(mat: np.array, labels: np.array, format_: str = "default") -> str:
    """Given a matrix that compares elements from different classes between themselves (e.g. a
    confusion matrix, correlation matrix, covariance matrix...). Produce a string table to display 
    on stdout or to embed into a latex document.

    Parameters
    ==========
        - "mat": Confusion matrix to format, must be a square matrix
        - "labels": list of labels that correspond to the rows and columns of the matrix.
                    must be the same length as the side of matrix "mat"
        - "format": How to format the table.
                        "default": Create a nice table to display to stdout
                        "latex": Create a table with latex syntax

    Returns
    =======
        A string object containing the matrix formated according to the parametets.
    """
    mat = mat.astype(str)
    mat = mat.tolist()

    # Padding with labels
    mat = [[*labels]] + mat
    mat = list(zip(*mat))  # Transpose
    mat = [["", *labels]] + mat
    mat = list(zip(*mat))  # Transpose

    # Generate a format template depending on "format_"
    if format_ == "default":
        format_str = "{:>10}" * len(mat)
        line_join = "\n"
    elif format_ == "latex":
        format_str = "{:>10}" + " & {:>10}" * len(mat[:-1])
        line_join = "\\\\\n"

    # Fill template with values
    mat = line_join.join(starmap(format_str.format, mat))

    return mat


def format_report(report_dict: dict, digits=2, format_="default") -> str:
    """Given a dict that contains different kinds of classification metric scores
    (as returned by sklearn.classification_report). Produce a string table to display 
    on stdout or to embed into a latex document.

    Parameters
    ==========
        - "report_dict": dictionary returned by "sklearn.classification_report"
        - "digits": Number of decimal digits to round to.
        - "format": How to format the table.
                        "default": Create a nice table to display to stdout
                        "latex": Create a table with latex syntax

    Returns
    =======
        A string object containing the data formated in a table according to the parametets.
    """

    # Table constants
    headers = ["precision", "recall", "f1-score", "support"]
    average_types = {"accuracy", "macro avg", "weighted avg"}

    # Generate format templates
    if format_ == "default":
        row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
        row_fmt_accuracy = '{:>{width}s} ' + \
            ' {:>9.{digits}}' * 2 + ' {:>9.{digits}f}' + \
            ' {:>9}\n'
        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers) + "\n\n"
    elif format_ == "latex":
        row_fmt = '{:>{width}s} &' + ' {:>9.{digits}f} &' * 3 + ' {:>9}\\\\\n'
        row_fmt_accuracy = '{:>{width}s} &' + \
            ' {:>9.{digits}} &' * 2 + ' {:>9.{digits}f} &' + \
            ' {:>9}\\\\\n'
        head_fmt = '{:>{width}s} &' + ' {:>9} &' * \
            len(headers[:-1]) + ' {:>9}\\\\\n'

    # Per class scores
    target_names = set(report_dict.keys()) - average_types
    scores = map(report_dict.__getitem__, target_names)
    scores = map(lambda x: tuple(map(x.__getitem__, headers)), scores)
    scores = zip(*scores)  # Transpose
    rows = zip(target_names, *scores)

    longest_last_line_heading = 'weighted avg'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), digits)

    report = head_fmt.format('', *headers, width=width)
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)

    if format_ == "latex":
        report += "&" * 4 + "\\" * 2
    report += '\n'

    # accuracy
    report += row_fmt_accuracy.format(
        "accuracy", '', '',
        report_dict["accuracy"],
        report_dict["macro avg"]["support"],
        width=width,
        digits=digits)

    # macro avg and weighted avg
    for avg in ("macro avg", "weighted avg"):
        report += row_fmt.format(avg, *
                                 report_dict[avg].values(), width=width, digits=digits)

    return report

# Data manipulation functions


def dict_extract_tensor(d: dict) -> list[Any]:  # Not used
    """Utility function for manipulating dicts. Converts dictionary into a nested list
    structure ignoring keys but preserving structure.

    Parameters
    ==========
        - "d": any dictionary
    Returns
    =======
        A list of possibly more nested lists or values that mimics the structure of the dict "d"
    """
    if not isinstance(d, dict):
        return d

    return list(map(dict_extract_tensor, d.values()))


def dict_op(op: callable, *d: dict) -> dict:
    """Utility function for manipulating dicts. Applies a n-arity operator on n 
    dicts elementwise. Dicts must have the same structure.

    Parameters
    ==========
        - "op": a operator of arbitrary number of parameters.
        - "*d": a list of dictionaries which to operate elementwise
    Returns
    =======
        A dictionary with the same structure of any of the dicts "d" but with each element being
        the result of applying operator "op" on the items in the same place within the nested
        structure.
    """
    d0 = d[0]
    if not isinstance(d0, dict):
        return op(*d)

    vals = []
    for k in d0.keys():
        d_new = map(operator.itemgetter(k), d)
        vals.append(dict_op(op, *d_new))

    return dict(zip(d0.keys(), vals))


def avg_dict(*d):
    """Utility function for averaging all the values of dicts elementwise.

    Parameters
    ==========
        - "*d": a list of identically structured dictionaries which to average elementwise
    Returns
    =======
        A dictionary with the same structure of any of the dicts "d" but with each element being
        the result of averaging the items in the same place within the nested structure.
    """

    def _avg_args(*args):
        return np.mean(args, axis=0)

    return dict_op(_avg_args, *d)
