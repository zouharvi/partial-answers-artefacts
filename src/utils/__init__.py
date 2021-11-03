import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.model_selection
import pickle
from utils.data import *
import itertools as it

def get_compute_device():
    """
    Return PyTorch GPU device if available, else CPU.
    GPU device is without an index (use CUDA_VISIBLE_DEVICES to specify a device).
    """
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE


def load_data_raw(path, check=False, singleton=False):
    """
    Loads raw JSON file and parses it into a list of articles.
    """
    data = load_data(path, format="json")

    if singleton:
        # in case the input is a single COP edition
        data = [
            {**article, "cop_edition": data["cop_edition"]}
            for article in data["articles"]
        ]
    else:
        # add cop_edition and flatten all lists from the newspaper
        data = [
            {**article, "cop_edition": newspaper["cop_edition"]}
            for newspaper in data
            for article in newspaper["articles"]
        ]

    # check that the data is what we expect
    if check:
        assert len(data) == 33660
        assert all([
            set(article.keys()) == {
                'path', 'raw_text', 'newspaper', 'date', 'headline', 'body', 'classification', 'cop_edition'
            }
            for article in data
        ])

    return data


def load_data(path, format="json"):
    """
    Saves any data in either JSON or pickle format.
    Format is NOT inferred automatically from the path.
    """
    if format == "json":
        with open(path, "r") as f:
            return json.load(f)
    elif format == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise Exception("Unknown load format specified")

def save_data(path, data, format="json"):
    """
    Saves any data in either JSON or pickle format.
    Format is NOT inferred automatically from the path.
    """
    if format == "json":
        with open(path, "w") as f:
            return json.dump(data, f)
    elif format == "pickle":
        with open(path, "wb") as f:
            return pickle.dump(data, f)
    else:
        raise Exception("Unknown save format specified")

def get_x(data, target):
    """
    @TODO missing comment
    """
    data_x, _ = zip(*data)
    return [x[target] for x in data_x]

def get_y(data, targets):
    """
    @TODO missing comment
    """
    _, data_y = zip(*data)

    # Get targets
    local_t = [x for x in targets if x in Y_KEYS_LOCAL]
    global_t = [x for x in targets if x in {"subject", "geographic"}]

    labels = list()

    # Extract local targets
    label_names_local = list()
    if local_t:
        y_local = ([y[t] for y in data_y] for t in local_t)
        y_local = ([[x] for x in t] for t in y_local)
        bin_local, y_local = zip(*map(binarize_data, y_local))
        y_local = np.array([np.argmax(y, axis=1) for y in y_local]).T

        labels.append(y_local)
        label_names_local = [b.classes_ for b in bin_local]

    # Extract non-local targets
    global_t_post = list()
    label_names_global = list()
    if global_t:
        y_global = ([y[t] for y in data_y] for t in global_t)
        bin_global, y_global = zip(*map(binarize_data, y_global))
        y_global = np.concatenate(list(y_global), axis=1)

        labels.append(y_global)
        label_names_global = [b.classes_ for b in bin_global]

        global_t_post = it.chain(
            *[[gt] * len(labels) for gt, labels in zip(global_t, label_names_global)])
        global_t_post = list(global_t_post)

        # TODO: [l+"_F", l+"_T"] please
        label_names_global = [
            "{l}_F¡{l}_T".format(l=l).split("¡")
            for l in it.chain(*label_names_global)
        ]

    # Put everything together
    targets_post = local_t + global_t_post
    label_names = label_names_local + label_names_global
    label_names = list(map(list, label_names))
    labels = np.concatenate(labels, axis=1)

    return targets_post, label_names, labels


def make_split(data_list, splits, random_state=0):
    """
    @TODO missing comment
    """
    data_list = list(zip(*data_list))
    data_splits = []
    for split in splits:
        data_list, data_split = sklearn.model_selection.train_test_split(
            data_list,
            test_size=split,
            random_state=random_state)

        data_splits.append(data_split)
    if len(data_list) > 0:
        data_splits.append(data_list)

    r = tuple(it.starmap(zip, data_splits))

    r_x, r_y = zip(*r)
    r_y = tuple(map(np.array, r_y))

    return tuple(zip(r_x, r_y))


def streamline_data(data, x_filter="headline", y_filter="newspaper", binarize="output"):
    """
    Automatically prepare and sanitize data to list of (text, class) where class has been binarized.
    Available y_filter are newspaper, ncountry, ncompas, subject, industry, geographic.
    If freq_cutoff is None, then defaults for subject, industry and geographic will be used.

    If binarize is "output" (default), then:
    Returns (binarizer, [(text, binarized_class_2)])

    If binarize is "input", then:
    Returns (binarizer, [(binarized_class_1, class)])

    If binarize is "none", then:
    Returns (binarizer, [(text, class)])

    If binarize is "all", then both the input and output is turned into a binarized array.
    The output is then
    Returns ((binarizer_1, binarizer_2), [(binarizerd_class_1, binarized_class_2)])
    """

    assert binarize in {"all", "input", "output", None, "none"}

    # TODO: this may potentially cause an issue in the future if "craft" is stored on y
    if x_filter in X_KEYS | {"craft"}:
        x_filter_name = str(x_filter)
        def x_filter(x, y): return [x[x_filter_name]]
    elif x_filter in Y_KEYS:
        x_filter_name = str(x_filter)

        # resolve encapsulating subject and geographic in a list twice
        if x_filter in Y_KEYS_LOCAL:
            def x_filter(x, y): return [y[x_filter_name]]
        else:
            def x_filter(x, y): return y[x_filter_name]
    elif callable(x_filter):
        pass
    else:
        raise Exception("Invalid x_filter parameter")

    if y_filter in Y_KEYS | {"craft"}:
        y_filter_name = str(y_filter)
        # resolve encapsulating subject and geographic in a list twice
        if y_filter in Y_KEYS_LOCAL:
            def y_filter(x, y): return [y[y_filter_name]]
        else:
            def y_filter(x, y): return y[y_filter_name]
    elif callable(y_filter):
        pass
    else:
        raise Exception("Invalid x_filter parameter")

    data_x = [x_filter(x, y) for x, y in data]
    data_y = [y_filter(x, y) for x, y in data]

    if binarize is None or binarize == "none":
        return list(zip(data_x, data_y))
    elif binarize == "output":
        binarizer_1, data_y = binarize_data(data_y)
        return binarizer_1, list(zip(data_x, data_y))
    elif binarize == "input":
        binarizer_2, data_x = binarize_data(data_x)
        return binarizer_2, list(zip(data_x, data_y))
    elif binarize == "all":
        binarizer_1, data_y = binarize_data(data_y)
        binarizer_2, data_x = binarize_data(data_x)
        return (binarizer_1, binarizer_2), list(zip(data_x, data_y))


def binarize_data(data_y):
    """
    Binarizes labels and returns them together with the binarizer object
    """
    binarizer = MultiLabelBinarizer()
    data_y = binarizer.fit_transform(data_y)
    return binarizer, data_y


def argmax_n(arr, n):
    """
    Extracts indicies of top `n` values from an array-like object.
    """
    n = min(len(arr), n)
    return np.argpartition(arr, -n)[-n:]


def rprec_local(gold_y, pred_y):
    """
    Computes a single instance of R-Precision (single sample).
    """
    ones_1 = {i for i, t in enumerate(gold_y) if t == 1}
    ones_2 = set(argmax_n(pred_y, len(ones_1)))
    return len(ones_1 & ones_2) / len(ones_1)


def rprec(gold_y, pred_y):
    """
    Computes average R-Precsion score from gold labels and scorings.
    The true labels must be binary-encoded.
    """
    return np.average([
        rprec_local(y, pred_y)
        for y, pred_y in zip(gold_y, pred_y)
    ])


def powerset(iterable, nonempty=False):
    """
    Generates all powersets of a list in order of increasing cardinality.
    """
    s = list(iterable)
    return it.chain.from_iterable(
        it.combinations(s, r)
        for r in range(1 if nonempty else 0, len(s) + 1)
    )


def pretty_json(data):
    """
    Returns prettified json of the given object.
    """
    return json.dumps(data, indent=4)