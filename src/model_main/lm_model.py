import sys
sys.path.append("src")
import utils

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel

import operator as op
import itertools as it
import tqdm


class LMModel(nn.Module):
    def __init__(
        self,
        cls_target_dimensions=list(),
        loss_weights=None,
        lm="bert-base-uncased",
        embed_strategy="cls",
        head_thickness="shallow",
        freeze_lm=False,
        max_length=256,
        epochs=3,
        batch_size=16,
        optimizer=torch.optim.Adam,
        optimizer_params=dict(lr=5e-5),
        device=utils.DEVICE
    ):

        super(LMModel, self).__init__()

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(lm)
        lm = AutoModel.from_pretrained(lm, output_hidden_states=True).to(device)

        # Build classification heads
        if head_thickness == "shallow":
            classification_heads = [
                nn.Linear(768, ct).to(device) for ct in cls_target_dimensions
            ]
        elif head_thickness == "mid":
            classification_heads = [
                nn.Sequential(
                    nn.Linear(768, 512),
                    nn.ReLU(),
                    nn.Linear(512, ct)
                ).to(device)
                for ct in cls_target_dimensions
            ]
        else:
            raise Exception("Unknown head_thickness")

        # Create loss and optimizer
        loss = nn.CrossEntropyLoss()

        if loss_weights is None:
            loss_weights = torch.ones(
                len(cls_target_dimensions), device=device)
        else:
            loss_weights = torch.tensor(loss_weights, device=device)
        weights_total = torch.sum(loss_weights)

        # Set attributes
        # Config
        self.embed_strategy = embed_strategy
        self.max_length = max_length
        self.device = device

        # Tokenizer
        self.tokenizer = tokenizer

        # Layers
        self.lm = lm
        self.classification_heads = classification_heads

        # Training
        self.freeze_lm = freeze_lm
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_weights = loss_weights
        self.weights_total = weights_total

        self.loss = loss

        # The optimizer has to be set last because self.parameters() scans class variables
        self.optimizer = optimizer(
            params=self.parameters(), **optimizer_params)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None
    ):
        # Embedd text
        x = self.lm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]

        # Reduce dimension using heuristic
        if self.embed_strategy == "cls":
            x = x[:, 0, :]
        elif self.embed_strategy == "avg":
            x = torch.mean(x, 1)
        elif self.embed_strategy != "all":
            raise Exception(
                f"Embed strategy {self.embed_strategy} is not valid."
            )

        # For each head make classification
        if self.classification_heads:
            y = [head(x) for head in self.classification_heads]
            return y

        return x

    def forward2(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None
    ):
        # Embedd text
        model_output = self.lm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Reduce dimension using heuristic
        if self.embed_strategy == "cls":
            x = model_output[0][:, 0, :]
        elif self.embed_strategy == "avg":
            x = torch.mean(model_output[0], 1)
        elif self.embed_strategy != "all":
            raise Exception(
                f"Embed strategy {self.embed_strategy} is not valid."
            )

        y = self.classification_heads[0](x)

        # take cls
        return [t[:,0,:] for t in model_output.hidden_states], y

    # Utility functions
    def fit(self,
            X_train, y_train,
            X_dev, y_dev):  # Only works with classification heads
        self._train(X_train, y_train, X_dev, y_dev)

    def predict(self, X):
        dl = self._convert2batched(X)
        dl = tqdm.tqdm(dl)  # Add progress bar

        self.lm.eval()
        with torch.no_grad():
            x = map(self._predict, dl)

            if self.classification_heads:
                x = zip(*x)
                x = map(torch.cat, x)

            x = [t.cpu().numpy() for t in x]

            if not self.classification_heads:
                x = np.concatenate(x)
        return x

    def predict2(self, X, top_cls_only=True):
        dl = self._convert2batched(X)
        dl = tqdm.tqdm(dl)  # Add progress bar

        assert len(self.classification_heads) == 1

        self.lm.eval()
        with torch.no_grad():
            # map is intentional here so that the data can be removed from the GPU device
            # once they are not needed
            x = map(lambda sample: self.forward2(*sample), dl)
            if top_cls_only:
                x = [
                    (sample[0][0].cpu().numpy(), sample[1].cpu().numpy())
                    for sample in x
                ]
            else:
                x = [
                    ([t.cpu().numpy() for t in sample[0]], sample[1].cpu().numpy())
                    for sample in x
                ]
        return x

    def save_to_file(self, filename):
        torch.save(self.state_dict(), filename)

    def load_from_file(self, filename):
        s_dict = torch.load(filename)
        self.load_state_dict(s_dict)

    # Private functions
    def _convert2batched(self, X, y=None, shuffle=False):

        def collate_fn(data):
            data = zip(*data)
            data = map(torch.stack, data)
            data = list(map(op.methodcaller("to", self.device), data))

            if len(data) > 3:
                return data[:3], data[3]
            else:
                return data

        # Convert to tensors
        X = list(X)
        X = self.tokenizer(
            X, padding=True, max_length=self.max_length,
            truncation=True, return_tensors="pt"
        ).data
        tensors = list(X.values())
        if y is not None:
            tensors.append(torch.tensor(y))

        tensors = TensorDataset(*tensors)
        dl = DataLoader(
            tensors,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            # pin_memory=True,
        )

        return dl

    def _train(self, X_train, y_train, X_dev, y_dev):
        dl_train = self._convert2batched(X_train, y_train, shuffle=True)
        dl_dev = self._convert2batched(X_dev, y_dev)

        for e in range(self.epochs):
            self._train_epoch(dl_train, dl_dev, epoch_i=e)

    def _train_epoch(
        self,
        dl_train: DataLoader,
        dl_dev: DataLoader = None,
        epoch_i=None
    ):

        self.train()

        # Train loop
        dl_train = tqdm.tqdm(dl_train, desc="Epoch {}".format(epoch_i + 1))
        losses = 0
        accs = 0
        for batch in dl_train:
            X, y = batch
            loss, acc = self._train_step(X, y)

            losses = losses * 0.9 + loss.item() * 0.1
            accs = accs * 0.9 + acc.item() * 0.1

            dl_train.set_postfix(dict(loss=losses, acc=accs))

        # Validation loop
        if dl_dev is None:
            return

        with torch.no_grad():
            dl_dev = tqdm.tqdm(dl_dev, desc="Validation")
            losses = []
            accs = []
            for batch in dl_dev:
                X, y = batch
                y_pred = self._predict(X)

                loss = self._compute_loss(y_pred, y)
                acc = self._compute_acc(y_pred, y)

                losses.append(loss.item())
                accs.append(acc.item())

            print("Dev evaluation", dict(loss=np.mean(losses), acc=np.mean(accs)))

    def _train_step(self, X, y):
        self.optimizer.zero_grad()

        y_pred = self._predict(X)
        loss = self._compute_loss(y_pred, y)

        loss.backward()
        self.optimizer.step()

        acc = self._compute_acc(y_pred, y)

        return loss, acc

    def _predict(self, X):

        return self(*X)

    def _compute_loss(self, y_pred, y):

        losses = self._compute_losses(y_pred, y)
        l = self._reduce_losses(losses)

        return l

    def _compute_losses(self, y_pred, y):
        y = y.T

        losses = it.starmap(self.loss, zip(y_pred, y))
        losses = list(losses)
        losses = torch.stack(losses)

        return losses

    def _reduce_losses(self, losses):
        losses = losses * self.loss_weights
        return torch.sum(losses) / self.weights_total

    def _compute_acc(self, y_pred, y):

        y_pred = [torch.argmax(yp, dim=1) for yp in y_pred]
        y_pred = torch.stack(y_pred, dim=1)

        return torch.mean((y == y_pred).float())
