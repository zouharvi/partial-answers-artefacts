from itertools import starmap
import operator

# Data formating
def format_auto_matrix(mat,labels):
    # TODO document
    mat = mat.astype(str)
    mat = mat.tolist()

    # Padding with labels
    mat = [[*labels]] + mat
    mat = list(zip(*mat)) # Transpose
    mat = [["",*labels]] + mat
    mat = list(zip(*mat)) # Transpose

    format_str = "{:10}"*len(mat)
    mat = "\n".join(starmap(format_str.format,mat))

    return mat

def format_report(report_dict,digits=2):
    # TODO document
    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'

    headers = ["precision", "recall", "f1-score", "support"]
    average_types = {"accuracy","macro avg","weighted avg"}

    # Per class scores
    target_names = set(report_dict.keys()) - average_types
    scores = map(report_dict.__getitem__,target_names)
    scores = map(lambda x: tuple(map(x.__getitem__,headers)),scores)
    scores = zip(*scores) # Transpose
    rows = zip(target_names,*scores)

    longest_last_line_heading = 'weighted avg'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), digits)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)
    report += '\n'

    # accuracy
    row_fmt_accuracy = '{:>{width}s} ' + \
                        ' {:>9.{digits}}' * 2 + ' {:>9.{digits}f}' + \
                        ' {:>9}\n'
    report += row_fmt_accuracy.format("accuracy", '', '',
                                        report_dict["accuracy"],
                                        report_dict["macro avg"]["support"],
                                        width=width,
                                        digits=digits)

    # macro avg and weighted avg
    for avg in ("macro avg","weighted avg"):
        report += row_fmt.format(avg, *report_dict[avg].values(), width=width, digits=digits)

    return report

# Data manipulation functions
def dict_extract_tensor(d): # Not used
    # TODO document
    if not isinstance(d,dict):
        return d

    return list(map(dict_extract_tensor,d.values()))

def dict_op(op,*d):
    # TODO document
    d0 = d[0]
    if not isinstance(d0,dict):
        return op(*d)

    vals = []
    for k in d0.keys():
        d_new = map(operator.itemgetter(k),d)
        vals.append(dict_op(op,*d_new))

    return dict(zip(d0.keys(),vals))