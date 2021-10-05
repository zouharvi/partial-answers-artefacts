from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Embedding, LSTM
from keras.initializers import Constant
import numpy as np
import tensorflow as tf
from utils import report_accuracy_score
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


class ModelLSTM():
    def __init__(self, embd_matrix):
        '''Create the Keras model to use'''

        # VECTORIZE
        # Transform words to indices using a vectorizer
        vectorizer = TextVectorization(
            standardize=None, output_sequence_length=50)
        # Use train and dev to create vocab - could also do just train
        text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
        vectorizer.adapt(text_ds)
        # Dictionary mapping words to idx
        voc = vectorizer.get_vocabulary()
        emb_matrix = get_emb_matrix(voc, embeddings)
        X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
        X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

        # Define settings, you might want to create cmd line args for them
        learning_rate = 0.01
        loss_function = 'categorical_crossentropy'
        optim = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        # Take embedding dim and size from emb_matrix
        embedding_dim = len(embd_matrix[0])
        num_tokens = len(embd_matrix)
        num_labels = len(6)
        # Now build the model
        self.model = Sequential()
        self.model.add(Embedding(
            num_tokens, embedding_dim,
            embeddings_initializer=Constant(embd_matrix), trainable=False,
        ))
        # Here you should add LSTM layers (and potentially dropout)
        raise NotImplementedError("Add LSTM layer(s) here")
        # Ultimately, end with dense layer with softmax
        self.model.add(Dense(input_dim=embedding_dim,
                       units=num_labels, activation="softmax"))
        # Compile model using our settings, check for accuracy
        self.model.compile(loss=loss_function,
                           optimizer=optim, metrics=['accuracy'])

    def train(self, X_train, Y_train, X_dev, Y_dev):
        '''Train the model here. Note the different settings you can experiment with!'''
        # Potentially change these to cmd line args again
        # And yes, don't be afraid to experiment!
        verbose = 1
        batch_size = 16
        epochs = 50
        # Early stopping: stop training when there are three consecutive epochs without improving
        # It's also possible to monitor the training loss with monitor="loss"
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3
        )

        # Finally fit the model to our data
        self.model.fit(
            X_train_vect, Y_train, verbose=verbose, epochs=epochs,
            callbacks=[callback], batch_size=batch_size,
            validation_data=(X_dev_vect, Y_dev)
        )
        # Print final accuracy for the model (clearer overview)
        report_accuracy_score(self.model.predict(X_dev), Y_dev)


class ModelBERT():
    def __init__(self, embd_matrix):
        '''Create the Keras model to use'''
        lm = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            lm, num_labels=6
        )
        loss_function = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )
        optim = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.model.compile(
            loss=loss_function, optimizer=optim,
            metrics=['accuracy']
        )

    def train(self, X_train, Y_train, X_dev, Y_dev):
        X_train = X_train[:10]
        Y_train = Y_train[:10]
        '''Train the model here. Note the different settings you can experiment with!'''
        tokens_train = self.tokenizer(
            X_train, padding=True, max_length=100,
            truncation=True, return_tensors="np").data
        tokens_dev = self.tokenizer(
            X_dev, padding=True, max_length=100,
            truncation=True, return_tensors="np").data
        self.model.fit(
            tokens_train, Y_train, verbose=1, epochs=1,
            batch_size=8, validation_data=(tokens_dev, Y_dev)
        )
        Y_pred = self.model.predict(tokens_dev)["logits"]
        report_accuracy_score(Y_dev, Y_pred)
