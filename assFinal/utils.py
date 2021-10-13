import json
from sklearn.preprocessing import MultiLabelBinarizer


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
            set(article.keys()) == {'path', 'raw_text', 'newspaper', 'date', 'headline', 'body', 'classification', 'cop_edition'}
            for article in data
        ])

    return data


def streamline_data(data, x_filter="headline", y_filter="newspaper"):
    if x_filter == "headline":
        def x_filter(x): return x["headline"]
    elif x_filter == "body":
        def x_filter(x): return x["body"]

    if y_filter == "newspaper":
        def y_filter(x): return [x["newspaper"]]
    if y_filter in {"subject", "organization", "industry", "geographic"}:
        def y_filter(x):
            return {item["name"] for item in x["classification"]["subject"]}

    data_x = [
        x_filter(article)
        for article in data
    ]
    binarizer = MultiLabelBinarizer()
    data_y = binarizer.fit_transform([
        y_filter(article)
        for article in data
    ])
    return binarizer, list(zip(data_x, data_y))