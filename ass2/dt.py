import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import re

accuracy_score = confusion_matrix

data = """
yellow small round yes
yellow small round no
green small irregular yes
green large irregular no
yellow large round yes
yellow small round yes
yellow small round yes
yellow small round yes
green small round no
yellow large round no
yellow large round yes
yellow large round no
yellow large round no
yellow large round no
yellow small irregular yes
yellow large irregular yes
"""

data = [line.split() for line in data.strip().split("\n")]
data_col = np.array([
    ["yellow", "green"].index(x[0]) for x in data
]).reshape(-1, 1)
data_siz = np.array([
    ["small", "large"].index(x[1]) for x in data
]).reshape(-1, 1)
data_shp = np.array([
    ["irregular", "round"].index(x[2]) for x in data
]).reshape(-1, 1)
data_edb = np.array([
    ["no", "yes"].index(x[3]) for x in data
])

data_X, data_Y = np.concatenate((data_col, data_siz, data_shp), axis=1), data_edb

## Decission tree
print(data_X,data_Y)

dtc = DecisionTreeClassifier(criterion="entropy").fit(data_X,data_Y)
dot_str = export_graphviz(
    dtc,
    feature_names=["Color","Size","Shape"],
    class_names=["Yes","No"],
    filled=True)

dot_str = dot_str.replace("Size <= 0.5","Size: small")
dot_str = dot_str.replace("Color <= 0.5","Color: yellow")
dot_str = dot_str.replace("Shape <= 0.5","Shape: round")
dot_str = re.sub(r"\\nvalue = \[\d+, \d+\]","",dot_str)
with open("dt.dot","w") as f:
    f.write(dot_str)



print("LEVEL 0")
print("ig", mutual_info_classif(
    np.concatenate((data_col, data_siz, data_shp), axis=1),
    data_edb,
    discrete_features=True
))
print("mcc", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_col, data_edb).predict(data_edb), data_edb)
)

data_col_gsmall = np.array([
    ["yellow", "green"].index(x[0]) for x in data if x[1] == "small"
]).reshape(-1, 1)
data_col_glarge = np.array([
    ["yellow", "green"].index(x[0])for x in data if x[1] == "large"
]).reshape(-1, 1)
data_shp_gsmall = np.array([
    ["irregular", "round"].index(x[2]) for x in data if x[1] == "small"
]).reshape(-1, 1)
data_shp_glarge = np.array([
    ["irregular", "round"].index(x[2]) for x in data if x[1] == "large"
]).reshape(-1, 1)
data_edb_gsmall = np.array([
    ["no", "yes"].index(x[3]) for x in data if x[1] == "small"
])
data_edb_glarge = np.array([
    ["no", "yes"].index(x[3]) for x in data if x[1] == "large"
])

print("LEVEL 1")

print("ig|small", mutual_info_classif(
    np.concatenate((data_col_gsmall, data_shp_gsmall), axis=1),
    data_edb_gsmall,
    discrete_features=True
))
print("mcc|small", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_col_gsmall, data_edb_gsmall).predict(data_edb_gsmall), data_edb_gsmall)
)

print("ig|large", mutual_info_classif(
    np.concatenate((data_col_glarge, data_shp_glarge), axis=1),
    data_edb_glarge,
    discrete_features=True
))
print("mcc|large", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_col_glarge, data_edb_glarge).predict(data_edb_glarge), data_edb_glarge)
)


data_col_gsmall_ground = np.array([
    ["yellow", "green"].index(x[0]) for x in data if x[1] == "small" and x[2] == "round"
]).reshape(-1, 1)
data_col_gsmall_girrgl = np.array([
    ["yellow", "green"].index(x[0]) for x in data if x[1] == "small" and x[2] == "irregular"
]).reshape(-1, 1)


data_shp_glarge_gyellw = np.array([
    ["irregular", "round"].index(x[2]) for x in data if x[1] == "large" and x[0] == "yellow"
]).reshape(-1, 1)
data_shp_glarge_ggreen = np.array([
    ["irregular", "round"].index(x[2]) for x in data if x[1] == "large" and x[0] == "green"
]).reshape(-1, 1)


data_edb_gsmall_ground = np.array([
    ["no", "yes"].index(x[3]) for x in data if x[1] == "small" and x[2] == "round"
])
data_edb_gsmall_girrgl = np.array([
    ["no", "yes"].index(x[3]) for x in data if x[1] == "small" and x[2] == "irregular"
])
data_edb_glarge_gyellw = np.array([
    ["no", "yes"].index(x[3]) for x in data if x[1] == "large" and x[0] == "yellow"
])
data_edb_glarge_ggreen = np.array([
    ["no", "yes"].index(x[3]) for x in data if x[1] == "large" and x[0] == "green"
])

print("LEVEL 2")
print("ig|small,round", mutual_info_classif(
    data_col_gsmall_ground,
    data_edb_gsmall_ground,
    discrete_features=True
))
print("mcc|small,round", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_col_gsmall_ground, data_edb_gsmall_ground).predict(data_edb_gsmall_ground), data_edb_gsmall_ground)
)
print("ig|small,irrgl", mutual_info_classif(
    data_col_gsmall_girrgl,
    data_edb_gsmall_girrgl,
    discrete_features=True
))
print("mcc|small,irrgl", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_col_gsmall_girrgl, data_edb_gsmall_girrgl).predict(data_edb_gsmall_girrgl), data_edb_gsmall_girrgl)
)

print("ig|large,yellw", mutual_info_classif(
    data_shp_glarge_gyellw,
    data_edb_glarge_gyellw,
    discrete_features=True
))
print("mcc|large,yellw", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_shp_glarge_gyellw, data_edb_glarge_gyellw).predict(data_edb_glarge_gyellw), data_edb_glarge_gyellw)
)
print("ig|large,green", mutual_info_classif(
    data_shp_glarge_ggreen,
    data_edb_glarge_ggreen,
    discrete_features=True
))
print("mcc|large,green", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_shp_glarge_ggreen, data_edb_glarge_ggreen).predict(data_edb_glarge_ggreen), data_edb_glarge_ggreen)
)

print("LEVEL 3")

data_edb_gsmall_ground_gyellw = np.array([
    ["no", "yes"].index(x[3]) for x in data if x[1] == "small" and x[2] == "round" and x[0] == "yellow"
]).reshape(-1, 1)
data_edb_gsmall_girrgl_ggreen = np.array([
    ["no", "yes"].index(x[3]) for x in data if x[1] == "small" and x[2] == "irregular" and x[0] == "green"
]).reshape(-1, 1)
print("mcc|small,round,yellw", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_edb_gsmall_ground_gyellw, data_edb_gsmall_ground_gyellw).predict(data_edb_gsmall_ground_gyellw), data_edb_gsmall_ground_gyellw)
)
print("mcc|small,round,green", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_edb_gsmall_girrgl_ggreen, data_edb_gsmall_girrgl_ggreen).predict(data_edb_gsmall_girrgl_ggreen), data_edb_gsmall_girrgl_ggreen)
)


data_edb_gsmall_ground_gyellw = np.array([
    ["no", "yes"].index(x[3]) for x in data if x[1] == "large" and x[2] == "round" and x[0] == "yellow"
]).reshape(-1, 1)
data_edb_gsmall_girrgl_gyellw = np.array([
    ["no", "yes"].index(x[3]) for x in data if x[1] == "large" and x[2] == "irregular" and x[0] == "yellow"
]).reshape(-1, 1)
print("mcc|large,round,yellw", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_edb_gsmall_ground_gyellw, data_edb_gsmall_ground_gyellw).predict(data_edb_gsmall_ground_gyellw), data_edb_gsmall_ground_gyellw)
)
print("mcc|large,irrgl,yellw", accuracy_score(
    DummyClassifier(strategy="most_frequent").fit(data_edb_gsmall_girrgl_gyellw, data_edb_gsmall_girrgl_gyellw).predict(data_edb_gsmall_girrgl_gyellw), data_edb_gsmall_girrgl_gyellw)
)
