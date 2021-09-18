from ast import ExtSlice
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

logdata_train_acc = []
logdata_train_f1 = []
logdata_loocv_acc = []
logdata_loocv_f1 = []

print("train")
for weights in ["uniform", "distance"]:
    for n_neighbours in range(1, len(data)+1):
        model = KNeighborsClassifier(n_neighbors=n_neighbours, weights=weights)
        model.fit(data_x, data_edb)
        pred = model.predict(data_x)
        score_acc = accuracy_score(pred, data_edb)
        score_f1 = f1_score(pred, data_edb)
        print(n_neighbours, score_acc)
        logdata_train_acc.append((n_neighbours, weights, score_acc))
        logdata_train_f1.append((n_neighbours, weights, score_f1))

print("loocv")
for weights in ["uniform", "distance"]:
    for n_neighbours in range(1, len(data)):
        model = KNeighborsClassifier(n_neighbors=n_neighbours, weights=weights)

        intermediate_acc = []
        intermediate_pred = []
        intermediate_f1 = []
        for train_index, test_index in KFold(n_splits=len(data_x)).split(data_x):
            model.fit(data_x[train_index], data_edb[train_index])
            pred = model.predict(data_x[test_index])
            intermediate_acc.append(accuracy_score(y_true=data_edb[test_index], y_pred=pred))
            intermediate_pred.append(pred[0])
        score_f1 = f1_score(y_true=data_edb, y_pred=intermediate_pred)

        score_acc = np.average(intermediate_acc)
        # score_f1 = np.average(intermediate_f1)
        print(n_neighbours, score_acc)
        logdata_loocv_acc.append((n_neighbours, weights, score_acc))
        logdata_loocv_f1.append((n_neighbours, weights, score_f1))

plt.figure(figsize=(9,4))
fig1 = plt.subplot(121)

plt.plot(
    [x[0] for x in logdata_train_acc if x[1] == "uniform"],
    [x[2] for x in logdata_train_acc if x[1] == "uniform"],
    label="Train (ACC)", c="tab:blue", linestyle=":"
)
plt.plot(
    [x[0] for x in logdata_train_f1 if x[1] == "uniform"],
    [x[2] for x in logdata_train_f1 if x[1] == "uniform"],
    label="Train (F1)", c="tab:blue", linestyle="-", alpha=0.7,
)
plt.plot(
    [x[0] for x in logdata_loocv_acc if x[1] == "uniform"],
    [x[2] for x in logdata_loocv_acc if x[1] == "uniform"],
    label="LOOCV (ACC)", c="tab:red", linestyle=":"
)
plt.plot(
    [x[0] for x in logdata_loocv_f1 if x[1] == "uniform"],
    [x[2] for x in logdata_loocv_f1 if x[1] == "uniform"],
    label="LOOCV (F1)", c="tab:red", linestyle="-", alpha=0.7,
)

plt.title("KNN Mushroom Edibility Performance (Uniform)")
plt.ylabel("Train/LOOCV accuracy/F1 score")
plt.xlabel("n_neighbours")
plt.legend()

fig2 = plt.subplot(122)
plt.plot(
    [x[0] for x in logdata_train_acc if x[1] == "distance"],
    [x[2] for x in logdata_train_acc if x[1] == "distance"],
    label="Train (ACC)", c="tab:blue", linestyle=":"
)
plt.plot(
    [x[0] for x in logdata_train_f1 if x[1] == "distance"],
    [x[2] for x in logdata_train_f1 if x[1] == "distance"],
    label="Train (F1)", c="tab:blue", linestyle="-", alpha=0.7,
)
plt.plot(
    [x[0] for x in logdata_loocv_acc if x[1] == "distance"],
    [x[2] for x in logdata_loocv_acc if x[1] == "distance"],
    label="LOOCV (ACC)", c="tab:red", linestyle=":"
)
plt.plot(
    [x[0] for x in logdata_loocv_f1 if x[1] == "distance"],
    [x[2] for x in logdata_loocv_f1 if x[1] == "distance"],
    label="LOOCV (F1)", c="tab:red", linestyle="-", alpha=0.7,
)
plt.title("KNN Mushroom Edibility Performance (Distance)")
# plt.ylabel("LOOCV accuracy/F1 score")
plt.xlabel("n_neighbours")
plt.legend()

plt.tight_layout()
plt.show()