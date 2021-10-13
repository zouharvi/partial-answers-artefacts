import json

def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

# data = load_data("data/final/COP.all.json")
# print(len(data))