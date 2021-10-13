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

def load_data(path, check=False):
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

def streamline_data(data, x_filter="headline", y_filter="newspaper", freq_cutoff=None):
    """
    Automatically prepare and sanitize data to list of (text, class) where class has been binarized.
    Available y_filter are newspaper, newspaper_country, newspaper_compas, subject, industry, geographic.
    If freq_cutoff is None, then defaults for subject, industry and geographic will be used.

    Returns (Binarizer, [(text, binarized class)])
    """

    if x_filter == "headline":
        def x_filter(x): return x["headline"]
    elif x_filter == "body":
        def x_filter(x): return x["body"]

    if y_filter == "newspaper":
        def y_filter(x): return [x["newspaper"]]
    elif y_filter == "newspaper_country":
        def y_filter(x): return [NEWSPAPER_TO_COUNTRY[x["newspaper"]]]
    elif y_filter == "newspaper_compas":
        def y_filter(x): return [NEWSPAPER_TO_COMPAS[x["newspaper"]]]
    elif y_filter == "organization":
        raise DeprecationWarning("The class ORGANIZATION has been deprecated because of low diverse frequency representation")
    elif y_filter in {"subject", "industry", "geographic"}:
        y_filter_key = str(y_filter)

        def y_filter(x):
            if x["classification"][y_filter_key] is None:
                return set()
            else:
                return [
                    item["name"]
                    for item in x["classification"][y_filter_key]
                ]

        if freq_cutoff is None:
            freq_cutoff = {
                "subject": 1000,
                "industry": 1000,
                "geographic": 500,
            }[y_filter_key]

    data_x = [
        x_filter(article)
        for article in data
    ]
    data_y = [
        y_filter(article)
        for article in data
    ]

    if freq_cutoff is not None:
        counter = Counter()
        for article_y in data_y:
            counter.update(article_y)
        allowed = {x[0] for x in counter.most_common() if x[1] >= freq_cutoff}
        
        data_y = [
            [item for item in article_y if item in allowed]
            for article_y in data_y
        ]
    
    binarizer = MultiLabelBinarizer()
    data_y = binarizer.fit_transform(data_y)

    return binarizer, list(zip(data_x, data_y))
