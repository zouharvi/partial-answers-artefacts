import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Embedding, LSTM, GRU, Bidirectional
from keras.layers.preprocessing.text_vectorization import TextVectorization
from keras.initializers import Constant
from tensorflow_addons.optimizers import AdamW
from keras.losses import CategoricalCrossentropy
from utils import report_accuracy_score, get_emb_matrix
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


class ModelLSTM():
    def __init__(self, 
            embeddings,
            X_all=None,
            epochs=50,
            batch_size=16,
            learning_rate=1e-3):
        '''Create the Keras model to use'''
        self.embeddings = embeddings

        # VECTORIZE
        # Transform words to indices using a vectorizer
        self.vectorizer = TextVectorization(
            standardize=None, output_sequence_length=50
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
            embeddings_initializer=Constant(embd_matrix),
            trainable=True,
            embeddings_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5),
        ))

        # Here you should add LSTM layers (and potentially dropout)
        self.model.add(Bidirectional(
            LSTM(units=128, dropout=0.1, return_sequences=True,)
        ))
        self.model.add(Bidirectional(
            LSTM(units=128, dropout=0.1)
        ))

        self.model.add(Dense(
            units=128, activation="relu"
        ))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(
            units=64, activation="relu"
        ))

        # Ultimately, end with dense layer with softmax
        self.model.add(Dense(
            input_dim=embedding_dim,
            units=6, activation="softmax"
        ))

        # Compile model using our settings, check for accuracy
        self.model.compile(
            loss=CategoricalCrossentropy(label_smoothing=0.1),
            # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            optimizer=AdamW(learning_rate=learning_rate, weight_decay=0.0001),
            metrics=['accuracy']
        )
        
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, X_train, Y_train, X_dev, Y_dev):
        '''Train the model here. Note the different settings you can experiment with!'''

        X_train_vect = self.vectorizer(
            np.array([[s] for s in X_train])
        ).numpy()
        X_dev_vect = self.vectorizer(
            np.array([[s] for s in X_dev])
        ).numpy()

        # Potentially change these to cmd line args again
        # And yes, don't be afraid to experiment!
        verbose = 1
        
        # Early stopping: stop training when there are three consecutive epochs without improving
        # It's also possible to monitor the training loss with monitor="loss"
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5,# restore_best_weights=True
        )

        # Finally fit the model to our data
        self.model.fit(
            X_train_vect, Y_train, verbose=verbose, epochs=self.epochs,
            callbacks=[callback], batch_size=self.batch_size,
            validation_data=(X_dev_vect, Y_dev)
        )
        # Print final accuracy for the model (clearer overview)
        report_accuracy_score(self.model.predict(X_dev_vect), Y_dev)

    def predict(self, X_test):
        X_test_vect = self.vectorizer(np.array([[s] for s in X_test])).numpy()
        return self.model.predict(X_test_vect)

class ModelTransformer():
    def __init__(self,
        lm="bert-base-uncased",
        max_length=100,
        epochs=3,
        batch_size=8,
        learning_rate=5e-5):
        '''Create the Keras model to use'''
        
        # Create and compile model
        model = TFAutoModelForSequenceClassification.from_pretrained(
            lm, num_labels=6
        )
        loss_function = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            loss=loss_function, optimizer=optim,
            metrics=['accuracy']
        )
        
        # Assign class attributes
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.model = model
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train, Y_train, X_dev, Y_dev):
        '''Train the model here. Note the different settings you can experiment with!'''
        tokens_train = self.tokenizer(
            X_train, padding=True, max_length=self.max_length,
            truncation=True, return_tensors="np").data
        tokens_dev = self.tokenizer(
            X_dev, padding=True, max_length=self.max_length,
            truncation=True, return_tensors="np").data
        self.model.fit(
            tokens_train, Y_train, verbose=1, epochs=self.epochs,
            batch_size=self.batch_size, validation_data=(tokens_dev, Y_dev)
        )
        Y_pred = self.model.predict(tokens_dev)["logits"]
        report_accuracy_score(Y_dev, Y_pred)
        
    def predict(self, X_test):
        x_tokens = self.tokenizer(
            X_test, padding=True, max_length=self.max_length,
            truncation=True, return_tensors="np").data
        return self.model.predict(x_tokens)["logits"]
