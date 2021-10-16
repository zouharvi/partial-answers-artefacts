import json
from sklearn.preprocessing import MultiLabelBinarizer
from utils_data import *
import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
DEVICE_CPU = torch.device("cpu")


def load_data_raw(path, check=False):
    """
    Loads raw JSON file and parses it into a list of articles
    """
    with open(path, "r") as f:
        data = json.load(f)

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


def load_data(path):
    with open(path, "r") as f:
        return json.load(f)


def save_data(path, data):
    with open(path, "w") as f:
        return json.dump(data, f)


X_KEYS = {"headline", "body"}
Y_KEYS = {"newspaper", "newspaper_country", "newspaper_compas",
          "month", "year", "subject", "geographic"}


def streamline_data(data, x_filter="headline", y_filter="newspaper"):
    """
    Automatically prepare and sanitize data to list of (text, class) where class has been binarized.
    Available y_filter are newspaper, newspaper_country, newspaper_compas, subject, industry, geographic.
    If freq_cutoff is None, then defaults for subject, industry and geographic will be used.

    Returns (Binarizer, [(text, binarized class)])
    """

    if x_filter in X_KEYS:
        x_filter_name = str(x_filter)
        def x_filter(x): return x[x_filter_name]
    elif callable(x_filter):
        pass
    else:
        raise Exception("Invalid x_filter parameter")

    if y_filter in Y_KEYS:
        y_filter_name = str(y_filter)
        def y_filter(y): return y[y_filter_name]
    elif callable(y_filter):
        pass
    else:
        raise Exception("Invalid x_filter parameter")

    data_x, data_y = zip(*data)
    data_x = [x_filter(x) for x in data_x]
    data_y = [y_filter(y) for y in data_y]

    return binarize_data(data_x, data_y)


def binarize_data(data_x, data_y):
    binarizer = MultiLabelBinarizer()
    data_y = binarizer.fit_transform(data_y)

    return binarizer, list(zip(data_x, data_y))


def streamline_data_craftRestv1(data, x_filter="headline", y_filter="newspaper"):
    """
    Prepends all answers to the input apart from the one specified by "y_filter_key"

    Returns (Binarizer, [(text, binarized class)])
    """
    assert y_filter in Y_KEYS
    assert x_filter in X_KEYS

    # TODO: for now drop these lists because they are long and there is no clean way to fuse them into the model
    Y_KEYS_LOCAL = Y_KEYS - {"subject", "geographic"} 
    def x_manipulator(x, y):
        y = {**y, y_filter: "None"}
        return ' | '.join([y[k] for k in Y_KEYS_LOCAL]) + " | " + x[x_filter]

    data_x = [
        x_manipulator(x, y)
        for x, y in data
    ]
    data_y = [
        [y[y_filter]]
        for x, y in data
    ]

    return binarize_data(data_x, data_y)
