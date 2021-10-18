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
Y_KEYS = {
    "newspaper", "ncountry", "ncompas",
    "month", "year", "subject", "geographic"
}
Y_KEYS_LOCAL = Y_KEYS - {"subject", "geographic"}
Y_KEYS_TO_CODE = {
    "newspaper": "n",
    "ncountry": "c",
    "ncompas": "o",
    "month": "m",
    "year": "y",
    "subject": "s",
    "geographic": "g"
}


def streamline_data(data, x_filter="headline", y_filter="newspaper", binarize_input=False):
    """
    Automatically prepare and sanitize data to list of (text, class) where class has been binarized.
    Available y_filter are newspaper, ncountry, ncompas, subject, industry, geographic.
    If freq_cutoff is None, then defaults for subject, industry and geographic will be used.

    Returns (binarizer, [(text, binarized_class_2)])

    If binarize_input is True, then the input is also turned into a class.
    The output is then
    Returns ((binarizer_1, binarizer_2), [(binarizerd_class_1, binarized_class_2)])
    """

    # TODO: this may potentially cause an issue in the future if "craft" is stored on y
    if x_filter in X_KEYS | {"craft"}:
        x_filter_name = str(x_filter)
        def x_filter(x, y): return [x[x_filter_name]]
    elif x_filter in Y_KEYS:
        x_filter_name = str(x_filter)
        def x_filter(x, y): return [y[x_filter_name]]
    elif callable(x_filter):
        pass
    else:
        raise Exception("Invalid x_filter parameter")

    if y_filter in Y_KEYS | {"craft"}:
        y_filter_name = str(y_filter)
        def y_filter(x, y): return [y[y_filter_name]]
    elif callable(y_filter):
        pass
    else:
        raise Exception("Invalid x_filter parameter")

    data_x = [x_filter(x,y) for x,y in data]
    data_y = [y_filter(x,y) for x,y in data]

    binarizer_1, data_y = binarize_data(data_y)

    if not binarize_input:
        return binarizer_1, list(zip(data_x, data_y))
    else:
        binarizer_2, data_x = binarize_data(data_x)
        return (binarizer_1, binarizer_2), list(zip(data_x, data_y))


def binarize_data(data_y):
    binarizer = MultiLabelBinarizer()
    data_y = binarizer.fit_transform(data_y)

    return binarizer, data_y

    # return binarizer, list(zip(data_x, data_y))
