import json

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    # add cop_edition and flatten all lists from the newspaper
    return [
        {**article, "cop_edition": newspaper["cop_edition"]}
        for newspaper in data
        for article in newspaper["articles"]
    ]