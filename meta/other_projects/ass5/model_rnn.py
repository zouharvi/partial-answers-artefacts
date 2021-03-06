from keras.backend import dropout
import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Embedding, LSTM, GRU, SimpleRNN, Bidirectional
from keras.layers.preprocessing.text_vectorization import TextVectorization
from keras.initializers import Constant
from tensorflow_addons.optimizers import AdamW
from keras.losses import CategoricalCrossentropy
from utils import report_accuracy_score, get_emb_matrix


class ModelRNN():
    def __init__(
            self,
            embeddings,
            X_all=None,
            args=None):
        '''TODO comment'''

        self.embeddings = embeddings

        # VECTORIZE
        # Transform words to indices using a vectorizer
        self.vectorizer = TextVectorization(
            standardize=None, output_sequence_length=300
        )
        # Use train and dev to create vocab - could also do just train
        text_ds = tf.data.Dataset.from_tensor_slices(X_all)
        self.vectorizer.adapt(text_ds)
        # Dictionary mapping words to idx
        voc = self.vectorizer.get_vocabulary()
        embd_matrix = get_emb_matrix(voc, embeddings)

        # Take embedding dim and size from emb_matrix
        embedding_dim = len(embd_matrix[0])
        num_tokens = len(embd_matrix)

        # Now build the model
        self.model = Sequential()
        self.model.add(Embedding(
            num_tokens, embedding_dim,
            embeddings_initializer=None if args.embd_random else Constant(
                embd_matrix),
            trainable=not args.embd_not_trainable,
            embeddings_regularizer=regularizers.l1_l2(
                l1=1e-5, l2=1e-4) if args.embd_reg else None,
        ))

        if args.embd_dense:
            self.model.add(Dense(
                units=300, activation=None, use_bias=True
            ))

        UNIT = {
            "lstm": LSTM,
            "gru": GRU,
            "rnn": SimpleRNN,
        }[args.rnn_unit.lower()]

        rnn_params = {
            "units": 128,
            "dropout": args.rnn_dropout,
            "recurrent_dropout": args.rnn_dropout_rec,
            "go_backwards": args.rnn_backwards,
        }

        for _ in range(args.rnn_layers - 1):
            if args.rnn_not_bi:
                self.model.add(
                    UNIT(**rnn_params, return_sequences=True,)
                )
            else:
                self.model.add(Bidirectional(
                    UNIT(**rnn_params, return_sequences=True,),
                    merge_mode=args.rnn_bimerge
                ))

        if args.rnn_not_bi:
            self.model.add(UNIT(**rnn_params))
        else:
            self.model.add(Bidirectional(
                UNIT(**rnn_params), merge_mode=args.rnn_bimerge))

        self.model.add(Dense(
            units=128, activation="relu"
        ))
        self.model.add(Dropout(args.dense_dropout))
        self.model.add(Dense(
            units=64, activation="relu"
        ))

        # Ultimately, end with dense layer with softmax
        self.model.add(Dense(
            input_dim=embedding_dim,
            units=6, activation="softmax"
        ))

        optimizer = {
            "adamw": AdamW(learning_rate=args.learning_rate, weight_decay=1e-4),
            "adam": tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            "sgd": tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
            "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate),
        }[args.optimizer.lower()]

        # Compile model using our settings, check for accuracy
        self.model.compile(
            loss=CategoricalCrossentropy(label_smoothing=0.0),
            # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.batch_size = args.batch_size
        self.epochs = args.epochs

    def train(self, X_train, Y_train, X_dev, Y_dev):
        '''Train the model here. Note the different settings you can experiment with!'''

        X_train_vect = self.vectorizer(
            np.array([[s] for s in X_train])
        ).numpy()
        X_dev_vect = self.vectorizer(
            np.array([[s] for s in X_dev])
        ).numpy()

        # Early stopping: stop training when there are three consecutive epochs without improving
        # It's also possible to monitor the training loss with monitor="loss"
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,  # restore_best_weights=True
        )

        # Finally fit the model to our data
        self.model.fit(
            X_train_vect, Y_train, verbose=1, epochs=self.epochs,
            callbacks=[callback], batch_size=self.batch_size,
            validation_data=(X_dev_vect, Y_dev)
        )
        # Print final accuracy for the model (clearer overview)
        report_accuracy_score(self.model.predict(X_dev_vect), Y_dev)

    def predict(self, X_test):
        X_test_vect = self.vectorizer(np.array([[s] for s in X_test])).numpy()
        return self.model.predict(X_test_vect)
