#!/usr/bin/env python3

import argparse
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection
import torch.nn
import pickle
import numpy as np

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data", default="data/final/clean.json",
        help="Location of joined data JSON",
    )
    args.add_argument(
        "--target-output", "--to", default="newspaper", help="Target variable",
    )
    args.add_argument(
        "--target-input", "--ti", default="both", help="Input variable",
    )
    args.add_argument(
        "--glove", default="data/glove.pkl", help="Path to GloVe dictionary",
    )
    return args.parse_args()


class Model(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.tfidf_dim = 4096
        self.glove_dim = 100

        self.lstm = torch.nn.LSTM(
            input_size=self.glove_dim,
            hidden_size=128,
            bidirectional=True
        )
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(256+self.tfidf_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        words, tfidf = x
        # print(words.shape)
        output, (hn, cn) = self.lstm(words)
        # take the output for the first word as the sequence representation
        output = output[:, 0, :]
        # add tfidf
        output = torch.cat((output, tfidf), dim=1)
        # print(output.shape)
        output = self.classification(output)
        # print(output.shape)
        return output

    def preprocess(self, data, glove):
        vectorizer = TfidfVectorizer(max_features=self.tfidf_dim, ngram_range=(1, 2))
        data_tfidf = vectorizer.fit_transform([x[0]["body"] for x in data])

        data_new = []
        for line, body_tfidf in zip(data, data_tfidf):
            # TODO: better tokenization?
            words = line[0]["headline"].lower().split()
            words_glove = [
                glove[word] if word in glove else [0.0]*self.glove_dim
                for word in words
            ]
            data_new.append((
                (
                    torch.Tensor([words_glove]).to(DEVICE),
                    torch.Tensor(body_tfidf.toarray()).to(DEVICE),
                ),
                torch.LongTensor([
                    np.argmax(line[1])
                ]).to(DEVICE),
            ))
        return data_new

    def train_epochs(self, data_train, data_dev, epochs, batch_size=32):
        loss_function = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-4,
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
            print(f"Train ACC: {self.eval_data(data_train):.2%}")
            print(f"Dev   ACC: {self.eval_data(data_dev):.2%}")

    def eval_data(self, data):
        hits = []
        for x, y in data:
            with torch.no_grad():
                output = self(x)
                hits.append(y[0].item() == torch.argmax(output[0], dim=0).item())
        return np.average(hits)


if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.data)
    _, data = streamline_data(
        data,
        x_filter=lambda x, y: x,
        y_filter=args.target_output, binarize="output"
    )
    with open(args.glove, "rb") as f:
        glove = pickle.load(f)

    if args.target_output in {"subject", "geographic"}:
        raise NotImplementedError()

    model = Model(output_dim=len(data[0][1]))
    data = model.preprocess(data, glove)
    print("Total data:", len(data))
    print("Glove len:", len(data[0][0][0]))
    print("Glove dim:", len(data[0][0][0][0]))
    print("TFIDF dim:", data[0][0][1].shape)
    print("Output dim:", len(data[0][1]))
    print("Output:", data[0][1])
    data_train, data_test = sklearn.model_selection.train_test_split(
        data,
        test_size=100,
        random_state=0,
    )
    model.to(DEVICE)
    model.train_epochs(data_train, data_test, 10)
