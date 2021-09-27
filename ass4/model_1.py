#!/usr/bin/env python3

import random as python_random
import json
import argparse
import numpy
from keras.models import Sequential
from tensorflow.keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Activation, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import RepeatedKFold, cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

# Make reproducible as much as possible
numpy.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='train_NE.txt', type=str,
                        help="Input file to learn from (default train_NE.txt)")
    parser.add_argument("-e", "--embeddings", default='glove_filtered.json', type=str,
                        help="Embedding file we are using (default glove_filtered.json)")
    parser.add_argument("-vp", "--val_percentage", default=0.10, type=float,
                        help="Percentage of the data that is used for the validation set")
    parser.add_argument("-ts", "--test-set", type=str,
                        help="Separate test set to read from, instead of data splitting (or both)")
    parser.add_argument("--experiment", default="pred",
                        help="{search|pred}")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Output file to which we write predictions for test set")
    args = parser.parse_args()
    if args.test_set and not args.output_file:
        raise ValueError(
            "Always specify an output file if you specify a separate test set!")
    if args.output_file and not args.test_set:
        raise ValueError(
            "Output file is specified but test set is not -- probably you made a mistake")
    return args


def write_to_file(lst, out_file):
    '''Write list to file'''
    with open(out_file, "w") as out_f:
        for line in lst:
            out_f.write(line.strip() + '\n')
    out_f.close()


def read_corpus(corpus_file):
    '''Read in the named entity data from a file'''
    names = []
    labels = []
    for line in open(corpus_file, 'r'):
        name, label = line.strip().split()
        names.append(name)
        labels.append(label)
    return names, labels


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: numpy.array(embeddings[word]) for word in embeddings}


def vectorizer(words, embeddings):
    '''Turn words into embeddings, i.e. replace words by their corresponding embeddings'''
    return numpy.array([embeddings[word] for word in words])


def create_model():
    '''Create the Keras model to use'''
    # Define settings, you might want to create cmd line args for them (or some other more reproducible method)
    # Start with MSE, but again, experiment!
    loss_function = CategoricalCrossentropy(label_smoothing=0)
    # loss_function = 'kl_divergence'

    # Optimizer
    # optimizer = Adam(learning_rate=0.0015)
    # activation = LeakyReLU(alpha=0.05)
    activation = "relu"
    dropout = 0.1
    width = 48
    optimizer = AdamW(learning_rate=0.0015, weight_decay=0.008)

    kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3)
    kernel_regularizer=None

    # Now build the model
    model = Sequential()
    # First dense layer has the number of features as input and the number of labels as total units
    model.add(Dense(input_dim=300, units=width, kernel_regularizer=kernel_regularizer))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(input_dim=width, units=width, kernel_regularizer=kernel_regularizer))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(input_dim=width, units=5, kernel_regularizer=kernel_regularizer))
    model.add(Activation("softmax"))

    # Compile model using our settings, check for accuracy
    model.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


def train_model(model, X_train, Y_train, validation_split, batch_size, epochs):
    '''Train the model here. Note the different settings you can experiment with!'''

    print("Training on", X_train.shape[0], f"samples with {validation_split:.0%}% validation split" )

    # Finally fit the model to our data
    model.fit(
        X_train, Y_train,
        verbose=1, epochs=epochs,
        batch_size=batch_size, validation_split=validation_split
    )
    return model


def separate_test_set_predict(test_set, embeddings, encoder, model, output_file):
    '''Do prediction on a separate test set for which we do not have a gold standard.
       Write predictions to a file'''
    # Read and vectorize data
    test_emb = vectorizer([x.strip() for x in open(test_set, 'r')], embeddings)
    # Make predictions
    pred = model.predict(test_emb)
    # Convert to numerical labels and back to string labels
    test_pred = numpy.argmax(pred, axis=1)
    labels = [encoder.classes_[idx] for idx in test_pred]
    # Finally write predictions to file
    write_to_file(labels, output_file)


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_full, Y_full = read_corpus(args.input_file)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to embeddings
    X_full = vectorizer(X_full, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    # Use encoder.classes_ to find mapping back
    Y_full = encoder.fit_transform(Y_full)

    if args.experiment == "search":
        estimator = KerasClassifier(
            build_fn=create_model, epochs=11, batch_size=64, verbose=0
        )
        kfold = RepeatedKFold(n_splits=10, n_repeats=5)
        results = cross_val_score(
            estimator, X_full, Y_full,
            cv=kfold, n_jobs=5
        )
        print(results.mean())

    elif args.experiment == "pred":
        model = create_model()
        model = train_model(
            model, X_full, Y_full,
            validation_split=args.val_percentage,
            batch_size=64, epochs=11
        )

        # If we specified a test set, there are no gold labels available
        # Do predictions and print them to a separate file
        if args.test_set:
            if args.val_percentage != 0.0:
                raise Exception("When outputing on blind test, set -vp 0.0")
            separate_test_set_predict(
                args.test_set, embeddings, encoder, model, args.output_file
            )


if __name__ == '__main__':
    main()
