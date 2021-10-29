"""
Define the standard and joint meta-models for success prediction.
"""

import sys
sys.path.append("src")
import utils

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import TensorDataset
import torch.nn.functional
import numpy as np
from sklearn.metrics import precision_score

DEVICE = utils.get_compute_device()


class ModelStandard(torch.nn.Module):
    """
    Standard model
    - input: artefact signature, posterior, hidden representation
    - output: success probability
    Assume to be used for the month task.
    """

    def __init__(self):
        super().__init__()

        self.dropout_layer = nn.Dropout(p=0.75)
        self.model = nn.Sequential(
            # |signature| = 4
            # |posterior (month)| = 7
            # |hidden representation| = 768
            nn.Linear(4 + 7 + 768, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # two output classes (positive and negative) so that this model can use
            # softmax and cross entropy;
            # alternative is to use a sigmoid and binary cross entropy
            nn.Linear(64, 2),
        ).to(DEVICE)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )

    def train_epochs(self, data_train, data_dev, epochs=20000):
        """
        Train the model and report results on train and development data.
        """
        # store the original training data so that they can be used for evaluation
        data_train_raw = data_train

        # transform the training data and move them to the compute device
        data_train = TensorDataset(
            # take posterior and signature
            torch.Tensor([
                np.concatenate((x[1], x[2]))
                for x, y in data_train
            ]).to(DEVICE),
            # take the hidden representation
            torch.Tensor([
                x[0]
                for x, y in data_train
            ]).to(DEVICE),
            # output
            torch.LongTensor([y * 1 for x, y in data_train]).to(DEVICE),
        )
        data_train = DataLoader(data_train, batch_size=128, shuffle=True)

        for epoch in range(epochs):
            self.train()
            losses = []

            # iterate one training step for every batch
            for x, rep, y in data_train:
                out = self(x, rep)
                loss = self.loss_fn(out, y)
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # report results every 50 epochs
            if (epoch + 1) % 50 == 0:
                print("Epoch", epoch + 1, f"loss: {np.average(losses):.4f}")

                with torch.no_grad():
                    self.eval()
                    print(f"Train: {self.eval_data(data_train_raw)}")
                    print(f"Dev:   {self.eval_data(data_dev)}")

    def eval_data(self, data):
        """
        Report precision on the input data
        """

        y_pred = []
        y_true = []
        for (rep, pos, sig), y in data:
            out = self(
                torch.Tensor(
                    np.concatenate((pos, sig))
                ).to(DEVICE),
                torch.Tensor(rep).to(DEVICE),
                batched=False
            ).cpu()
            out = torch.nn.functional.softmax(out, dim=0)

            # output decision
            out = out[1] >= out[0] + 0.5
            y_pred.append(out)
            y_true.append(y)

        return f"{precision_score(y_true, y_pred):.2%}"

    def forward(self, x, rep, batched=True):
        # dropout the hidden representation input and join it with the rest
        rep = self.dropout_layer(rep)
        x = torch.cat((x, rep), dim=1 if batched else 0)
        return self.model(x)


class ModelJoint(torch.nn.Module):
    """
    Joint model
    - input: posterior, hidden representation
    - output: probability of success for fusion of each artefact (+ no artefact)
    Assume to be used for the month task.
    """

    def __init__(self):
        super().__init__()

        self.dropout_layer = nn.Dropout(p=0.75)
        self.model = nn.Sequential(
            # |posterior (month)| = 7
            # |hidden representation| = 768
            nn.Linear(7 + 768, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # two output classes (positive and negative) for every artefact
            # so that this model can use softmax and cross entropy;
            # alternative is to use a sigmoid and binary cross entropy
            nn.Linear(64, 5 * 2),
        ).to(DEVICE)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )

    def train_epochs(self, data_train, data_dev, epochs=20000):
        """
        Train the model and report results on train and development data.
        """
        # store the original training data so that they can be used for evaluation
        data_train_raw = data_train

        # transform the training data and move them to the compute device
        data_train = TensorDataset(
            # take posterior
            torch.Tensor([
                x[1]
                for x, y in data_train
            ]).to(DEVICE),
            # take the hidden representation
            torch.Tensor([
                x[0]
                for x, y in data_train
            ]).to(DEVICE),
            # output
            torch.LongTensor([y for x, y in data_train]).to(DEVICE),
        )
        data_train = DataLoader(data_train, batch_size=128, shuffle=True)

        for epoch in range(epochs):
            self.train()
            losses = []

            # iterate one training step for every batch
            for x, rep, y in data_train:
                out = self(x, rep)
                loss = sum([
                    self.loss_fn(out_i, y_i)
                    for out_i, y_i in zip(out, y)
                ])
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # report results every 50 epochs
            if (epoch + 1) % 50 == 0:
                print("Epoch", epoch + 1, f"loss: {np.average(losses):.4f}")

                with torch.no_grad():
                    self.eval()
                    print(f"Train: {self.eval_data(data_train_raw)}")
                    print(f"Dev:   {self.eval_data(data_dev)}")

    def eval_data(self, data):
        """
        Report average precision on the input data across all artefact dimensions
        """
        y_pred_indiv = []
        y_true_indiv = []
        y_pred_no = []
        y_true_no = []
        for (rep, pos, sig), y in data:
            outs = self(
                torch.Tensor(pos).to(DEVICE),
                torch.Tensor(rep).to(DEVICE),
                batched=False
            ).cpu()

            # process every artefact
            for out, y_i in zip(outs[:4], y):
                out = torch.nn.functional.softmax(out, dim=0)
                
                # output decision
                out = out[1] >= out[0] + 0.5
                y_pred_indiv.append(out)
                y_true_indiv.append(y_i)

            # process "no artefact"
            out = outs[4]
            out = torch.nn.functional.softmax(out, dim=0)
            # output decision
            out = out[1] >= out[0] + 0.5
            y_pred_no.append(out)
            y_true_no.append(y[4])

        return format(precision_score(y_true_no, y_pred_no), ".2%") + ", " + format(precision_score(y_true_indiv, y_pred_indiv), ".2%")

    def forward(self, x, rep, batched=True):
        # dropout the hidden representation input and join it with the rest
        rep = self.dropout_layer(rep)
        x = torch.cat((x, rep), dim=1 if batched else 0)

        # reshape flat output (5*2) to a rectangular matrix
        if batched:
            return self.model(x).reshape(-1, 5, 2)
        else:
            return self.model(x).reshape(5, 2)
