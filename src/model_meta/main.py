#!/usr/bin/env python3

import sys
sys.path.append("src")
import argparse
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier

from model import ModelStandard, ModelJoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", default='data/misc/meta_month.pkl',
        help="Path of the data file."
    )
    parser.add_argument(
        "-cm", "--col-mode", default='no',
        help="What artefact mode to use (no, indiv, multi)"
    )
    parser.add_argument(
        "-pm", "--posterior-mode", default='true',
        help="What posterior mode to use (true, frozen)"
    )

    parser.add_argument(
        "--model", default='standard',
        help="What model to use (standard, joint)"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    def col_mode(signature):
        if args.col_mode == "no":
            return sum(signature) == 0
        elif args.col_mode == "indiv":
            return sum(signature) == 1
        elif args.col_mode == "multi":
            return sum(signature) != 0
        raise Exception("Unknown col-mode")

    # collate
    assert len(data) % 16 == 0
    bucket = []
    data_new = []
    for x in data:
        bucket.append(x)
        if len(bucket) == 16:
            data_new.append(bucket)
            bucket = []
    assert len(bucket) == 0

    if args.posterior_mode == "frozen":
        for bucket_i, bucket in enumerate(data_new):
            # zeroth sample should be without artefacts
            sample_zero = bucket[0]
            assert sum(sample_zero[0][2]) == 0
            # use input from zeroth sample
            for sample_i, sample in enumerate(bucket):
                data_new[bucket_i][sample_i] = ((
                    sample_zero[0][0],
                    sample_zero[0][1],
                    data_new[bucket_i][sample_i][0][2],
                ), data_new[bucket_i][sample_i][1])

    if args.model == "joint":
        data_new_joint = []
        for bucket_i, bucket in enumerate(data_new):
            # zeroth sample should be without artefacts
            sample_zero = bucket[0]
            assert sum(sample_zero[0][2]) == 0
            y_new = [None]*5
            # set last element to prediction without artefacts
            y_new[4] = sample_zero[1]
        
            for sample in bucket:
                # individual artefact
                if sum(sample[0][2]) == 1:
                    artefact_i = np.argmax(sample[0][2])
                    y_new[artefact_i] = sample[1]

            data_new_joint.append(
                [(sample_zero[0], y_new)]
            )            
            data_new = data_new_joint
    else:
        # take only specific col_mode
        data_new = [
            [(x, y) for x, y in bucket if col_mode(x[2])]
            for bucket in data_new
        ]

    # split at sample boundaries
    data_train, data_dev = train_test_split(
        data_new,
        test_size=1000 // 16,
        random_state=0,
    )

    # flatten again from buckets
    data_train = [x for bucket in data_train for x in bucket]
    data_dev = [x for bucket in data_dev for x in bucket]

    if args.model != "joint":
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(
            [x[1] for x, y in data_train],
            [y for x, y in data_train],
        )
        dummy_pred = dummy.predict(
            [x for x, y in data_dev],
        )
        dummy_val = precision_score(
            [y for x, y in data_dev],
            dummy_pred,
        )
        print(f"Dummy dev:   {dummy_val:.2%}")

        model = RandomForestClassifier(n_estimators=100)
        model.fit(
            [x[1] for x, y in data_train],
            [y for x, y in data_train],
        )
        dt_pred = model.predict(
            [x[1] for x, y in data_dev],
        )
        dt_val = precision_score(
            [y for x, y in data_dev],
            dt_pred,
        )
        print(f"DT dev:   {dt_val:.2%}")

    if args.model == "standard":
        model = ModelStandard()
    elif args.model == "joint":
        model = ModelJoint()
    else:
        raise Exception("Unknown model specified")

    model.train_epochs(data_train, data_dev)
