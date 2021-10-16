import json
from sklearn.preprocessing import MultiLabelBinarizer
from utils_data import *
from collections import Counter
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

def filter_data(data, cutoff=False):
    data_x = [
        {
            "headline": article["headline"],
            "body": article["body"],
        }
        for article in data
    ]

    def y_filter(x, y_filter_key):
        if x["classification"][y_filter_key] is None:
            return []
        else:
            return [
                item["name"]
                for item in x["classification"][y_filter_key]
            ]

    data_y = [
        {
            "newspaper": article["newspaper"],
            "newspaper_country": NEWSPAPER_TO_COUNTRY[article["newspaper"]],
            "newspaper_compas": NEWSPAPER_TO_COMPAS[article["newspaper"]],
            "month": article["date"].split()[0],
            "year": article["date"].split()[-1],
            "subject": y_filter(article, "subject"),
            "geographic": y_filter(article, "geographic"),
        }
        for article in data
    ]

    counter_sub = Counter()
    counter_geo = Counter()
    for article_y in data_y:
        counter_sub.update(article_y["subject"])
        counter_geo.update(article_y["geographic"])
    allowed_sub = {x[0] for x in counter_sub.most_common() if x[1] >= 1000}
    allowed_geo = {x[0] for x in counter_geo.most_common() if x[1] >= 250}

    data_y = [
        {
            **article_y,
            "subject": [x for x in article_y["subject"] if x in allowed_sub],
            "geographic": [x for x in article_y["geographic"] if x in allowed_geo],
        }
        for article_y in data_y
    ]

    data = [
        (x, y)
        for x, y in zip(data_x, data_y)
        if (not cutoff) or (len(y["subject"]) != 0 and len(y["geographic"]) != 0)
    ]

    return data


def streamline_data(data, x_filter="headline", y_filter="newspaper"):
    """
    Automatically prepare and sanitize data to list of (text, class) where class has been binarized.
    Available y_filter are newspaper, newspaper_country, newspaper_compas, subject, industry, geographic.
    If freq_cutoff is None, then defaults for subject, industry and geographic will be used.

    Returns (Binarizer, [(text, binarized class)])
    """

    if x_filter in {"headline", "body"}:
        x_filter_name = str(x_filter)
        def x_filter(x): return x[x_filter_name]
    elif callable(x_filter):
        pass
    else:
        raise Exception("Invalid x_filter parameter")

    if y_filter in {"newspaper", "newspaper_country", "newspaper_compas", "month", "year", "subject", "geographic"}:
        y_filter_name = str(y_filter)
        def y_filter(y): return y[y_filter_name]
    elif callable(y_filter):
        pass
    else:
        raise Exception("Invalid x_filter parameter")

    data_x, data_y = zip(*filter_data(data, cutoff=True))

    data_x = [x_filter(x) for x in data_x]
    data_y = [y_filter(y) for y in data_y]

    binarizer = MultiLabelBinarizer()
    data_y = binarizer.fit_transform(data_y)

    return binarizer, list(zip(data_x, data_y))
