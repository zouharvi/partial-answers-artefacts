import numpy as np

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, KFold

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
data_x = np.concatenate((data_col, data_siz, data_shp), axis=1)

logdata_train = []
logdata_train_f1 = []
logdata_loocv = []

print("train")
for weights in ["uniform", "distance"]:
    for n_neighbours in range(1, len(data)+1):
        model = KNeighborsClassifier(n_neighbors=n_neighbours, weights=weights)
        model.fit(data_x, data_edb)
        pred = model.predict(data_x)
        score = accuracy_score(pred, data_edb)
        score_f1 = f1_score(pred, data_edb)
        print(n_neighbours, score)
        logdata_train.append((n_neighbours, weights, score))
        logdata_train_f1.append((n_neighbours, weights, score_f1))

print("loocv")
for weights in ["uniform", "distance"]:
    for n_neighbours in range(1, len(data)):
        model = KNeighborsClassifier(n_neighbors=n_neighbours, weights=weights)
        score = cross_validate(model, data_x, data_edb, cv=KFold(n_splits=len(data_x)))
        score = np.average(score["test_score"])
        print(n_neighbours, score)
        logdata_loocv.append((n_neighbours, weights, score))


plt.plot(
    [x[0] for x in logdata_train if x[1] == "uniform"],
    [x[2] for x in logdata_train if x[1] == "uniform"],
    label="Uniform (ACC)", c="tab:blue", linestyle=":"
)
plt.plot(
    [x[0] for x in logdata_train if x[1] == "distance"],
    [x[2] for x in logdata_train if x[1] == "distance"],
    label="Distance (ACC)", c="tab:red", linestyle=":"
)
plt.plot(
    [x[0] for x in logdata_train_f1 if x[1] == "uniform"],
    [x[2] for x in logdata_train_f1 if x[1] == "uniform"],
    label="Uniform (F1)", c="tab:blue", linestyle="-"
)
plt.plot(
    [x[0] for x in logdata_train_f1 if x[1] == "distance"],
    [x[2] for x in logdata_train_f1 if x[1] == "distance"],
    label="Distance (F1)", c="tab:red", linestyle="-"
)
# plt.plot(
#     [x[0] for x in logdata_loocv if x[1] == "uniform"],
#     [x[2] for x in logdata_loocv if x[1] == "uniform"],
#     label="Uniform (LOOCV)", c="tab:blue", linestyle="-", alpha=0.7,
# )
# plt.plot(
#     [x[0] for x in logdata_loocv if x[1] == "distance"],
#     [x[2] for x in logdata_loocv if x[1] == "distance"],
#     label="Distance (LOOCV)", c="tab:red", linestyle="-", alpha=0.7,
# )
plt.title("KNN Mushroom Edibility Performance")
plt.xlabel("n_neighbours")
plt.ylabel("Train accuracy/F1 score")
plt.legend()
plt.show()