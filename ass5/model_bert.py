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
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


class ModelTransformer():
    def __init__(self,
        lm="bert-base-uncased",
        max_length=100,
        epochs=3,
        batch_size=8,
        learning_rate=5e-5,
        polinomial_decay_args=dict(
            decay_steps=10000,
            )
        ):
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
