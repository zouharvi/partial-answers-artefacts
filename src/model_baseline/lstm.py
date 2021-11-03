#!/usr/bin/env python3

"""
Baseline LSTM model with pre-trained word embeddings (GloVe).
"""

import sys
sys.path.append("src")
import utils
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import torch.nn
import numpy as np

DEVICE = utils.get_compute_device()


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data", default="data/final/clean.json",
        help="Location of joined data JSON",
    )
    args.add_argument(
        "-to", "--target-output", default="newspaper",
        help="Target variable",
    )
    args.add_argument(
        "-ti", "--target-input", default="both",
        help="Input variable",
    )
    args.add_argument(
        "--glove", default="data/glove.pkl",
        help="Path to GloVe dictionary",
    )
    return args.parse_args()


class Model(torch.nn.Module):
    def __init__(self, output_dim, single_class):
        super().__init__()
        self.tfidf_dim = 4096 * 8
        self.glove_dim = 200
        self.single_class = single_class

        self.lstm = torch.nn.LSTM(
            input_size=self.glove_dim,
            hidden_size=256,
            bidirectional=True
        )
        self.tfidf_dropout = torch.nn.Dropout(p=0.75)
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(256 * 2 + self.tfidf_dim, 512),
            torch.nn.Dropout(p=0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_dim),
            torch.nn.Softmax(dim=1) if single_class else torch.nn.Sigmoid()
        )

    def forward(self, x):
        words, tfidf = x
        output, (hn, cn) = self.lstm(words)
        # take the output for the first word as the sequence representation
        output = output[:, 0, :]
        # dropout tfidf
        tfidf = self.tfidf_dropout(tfidf)
        # add tfidf
        output = torch.cat((output, tfidf), dim=1)
        output = self.classification(output)
        return output

    def preprocess(self, data, glove):
        vectorizer = TfidfVectorizer(
            max_features=self.tfidf_dim, ngram_range=(1, 2))
        data_tfidf = vectorizer.fit_transform([x[0]["body"] for x in data])

        data_new = []
        for line, body_tfidf in zip(data, data_tfidf):
            # TODO: better tokenization?
            words_head = line[0]["headline"].lower().split()
            words_body = line[0]["body"].lower().split()[:20]
            words_glove = [
                glove[word] if word in glove else [0.0] * self.glove_dim
                for word in words_head + words_body
            ]

            if self.single_class:
                output = torch.LongTensor([
                    np.argmax(line[1])
                ])
            else:
                output = torch.FloatTensor([
                    line[1]
                ])

            data_new.append((
                (
                    torch.Tensor([words_glove]).to(DEVICE),
                    torch.Tensor(body_tfidf.toarray()).to(DEVICE),
                ),
                output.to(DEVICE),
            ))
        return data_new

    def train_epochs(self, data_train, data_dev, epochs, batch_size=128):
        if self.single_class:
            loss_function = torch.nn.CrossEntropyLoss()
        else:
            loss_function = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
        )

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            self.lstm.train(True)
            self.classification.train(True)
            for i in range(len(data_train) // batch_size):
                optimizer.zero_grad()
                batch = data_train[i * batch_size:(i + 1) * batch_size]
                loss = 0
                for x, y in batch:
                    output = self(x)
                    loss += loss_function(output, y)
                loss.backward()
                optimizer.step()

            self.eval()
            if self.single_class:
                print(f"Train ACC: {self.eval_data_acc(data_train):.2%}")
                print(f"Dev   ACC: {self.eval_data_acc(data_dev):.2%}")
            else:
                print(f"Train RPrec: {self.eval_data_rprec(data_train):.2%}")
                print(f"Dev   RPrec: {self.eval_data_rprec(data_dev):.2%}")

    def eval_data_acc(self, data):
        hits = []
        for x, y in data:
            with torch.no_grad():
                output = self(x)
                hits.append(
                    y[0].item() == torch.argmax(output[0], dim=0).item()
                )
        return np.average(hits)

    def eval_data_full(self, data):
        pred_y = []
        true_y = []
        for x, y in data:
            with torch.no_grad():
                output = self(x)
                true_y.append(y[0].item())
                pred_y.append(torch.argmax(output[0], dim=0).item())
        return classification_report(true_y, pred_y, zero_division=0)

    def eval_data_rprec(self, data):
        scores = []
        for x, y in data:
            with torch.no_grad():
                output = self(x)
                # print(output[0].shape)
                scores.append(output[0].tolist())
        return utils.rprec([x[1][0].tolist() for x in data], scores)


if __name__ == "__main__":
    args = parse_args()
    data = utils.load_data(args.data)
    _, data = utils.streamline_data(
        data,
        x_filter=lambda x, y: x,
        y_filter=args.target_output, binarize="output"
    )
    glove = utils.load_data(args.glove, format="pickle")

    if args.target_output in {"subject", "geographic"}:
        single_classs = False
    else:
        single_classs = True

    model = Model(output_dim=len(data[0][1]), single_class=single_classs)
    data = model.preprocess(data, glove)

    print("Total data:", len(data))
    print("Glove len:", len(data[0][0][0]))
    print("Glove dim:", len(data[0][0][0][0]))
    print("TFIDF dim:", data[0][0][1].shape)
    print("Output dim:", len(data[0][1]))
    print("Output:", data[0][1])

    data_dev, data_test, data_train = utils.make_split(
        (data,),
        (1000,1000,),
        random_state=0,
        simple=True,
    )

    model.to(DEVICE)
    model.train_epochs(data_train, data_test, 60)

    # test evaluation
    if single_classs:
        print(model.eval_data_full(data_test))
    else:
        print(f"Test RPrec: {model.eval_data_full(data_test):.2%}")
