import numpy as np
import keras
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Embedding, LSTM, GRU, SimpleRNN, Bidirectional
from keras.layers.preprocessing.text_vectorization import TextVectorization
from keras.initializers import Constant
from keras.callbacks import TensorBoard
from tensorflow_addons.optimizers import AdamW
from keras.losses import CategoricalCrossentropy
from utils import report_accuracy_score, get_emb_matrix
from transformers import AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix

class ModelTransformer():
    '''Class that encapsulates transformer language models for 6-way topic classification.'''
    def __init__(self,
        lm="bert-base-uncased",
        embedd_strategy="cls",
        freeze_lm=False,
        max_length=256,
        epochs=3,
        batch_size=16,
        learning_rate=5e-5,
        weight_decay=None,
        ):
        '''Create the Keras model to use.
        
        Parameters:
        ===========
            - lm: name of lanugage model to use (from HuggingFace modelhub)
            
            - embedd_strategy: The strategy to use to get a sentence embedding for classification.
                             Options are "cls", "avg", "lstm" and "bilstm".
            
            - max_length: Maximum length of the input sequence, larger inputs get truncated.
            
            - epochs: Number of epochs to train.
            
            - batch_size: Size of batch to process per training step.
            
            - learning_rate: Learning rate coefficient, controls training speed and convergence properties.
            
            - weight_decay: Dictionary containing parameters for polynomial weight decay'''
        
        # Create and compile model
        ## Inputs
        input_ids = tf.keras.Input(shape=(max_length, ),dtype='int32', name="input_ids")
        attention_mask = tf.keras.Input(shape=(max_length, ), dtype='int32',name="attention_mask")
        
        ## Get transformer LM
        transformer_model = TFAutoModel.from_pretrained(lm)
        
        if freeze_lm: # Offlline training
            for param in transformer_model.bert.weights:
                 param._trainable = False
        
        ## Add additional layers
        x = transformer_model([input_ids,attention_mask])
        if embedd_strategy == "cls":
            x = Dropout(rate=0.1)(x[1])
            x = Dense(units=6,activation="linear")(x)
        elif embedd_strategy == "avg":
            x = tf.math.reduce_mean(x[0],1)
            x = Dropout(rate=0.1)(x)
            x = Dense(units=6,activation="linear")(x)
        elif embedd_strategy == "lstm":
            x = LSTM(units=x[0].shape[-1])(x[0])
            x = Dropout(rate=0.1)(x)
            x = Dense(units=6,activation="linear")(x)
        elif embedd_strategy == "bilstm":
            x = Bidirectional(LSTM(units=x[0].shape[-1]))(x[0])
            x = Dropout(rate=0.1)(x)
            x = Dense(units=6,activation="linear")(x)
        model = tf.keras.Model(inputs=[input_ids,attention_mask],outputs=x)
        
        ## Loss function
        loss_function = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )
        
        ## Optimizer
        if weight_decay is None:
            optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optim = tfa.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=PolynomialDecay(
                    initial_learning_rate=learning_rate,
                    **weight_decay))
        
        ## Compile the model
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
        # Obtain text representations to use with the model
        tokens_train = self.tokenizer(
            X_train, padding=True, max_length=self.max_length,
            truncation=True, return_tensors="np").data
        tokens_dev = self.tokenizer(
            X_dev, padding=True, max_length=self.max_length,
            truncation=True, return_tensors="np").data
        
        # Fit model
        self.model.fit(
            tokens_train, Y_train, verbose=1, epochs=self.epochs,
            batch_size=self.batch_size, validation_data=(tokens_dev, Y_dev),
        )
        
        # Test on development set
        Y_pred = self.model.predict(tokens_dev)
        self.report_result(Y_dev, Y_pred)
        
    def predict(self, X_test):
        x_tokens = self.tokenizer(
            X_test, padding=True, max_length=self.max_length,
            truncation=True, return_tensors="np").data
        return self.model.predict(x_tokens)
    
    def report_result(self,Y_dev, Y_pred):
        report_accuracy_score(Y_dev, Y_pred)
        Y_dev = np.argmax(Y_dev,axis=1)
        Y_pred = np.argmax(Y_pred,axis=1)
        print(Y_pred)
        print(Y_dev)
        print(confusion_matrix(Y_dev, Y_pred))
        