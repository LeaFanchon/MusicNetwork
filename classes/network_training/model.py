#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:00:41 2019

@author: lea
"""
import pickle
import numpy as np
import tensorflow as tf

from classes.dataset_creation.chord import Chord

# The following paragraph is needed in order to use the GPUs while training. 
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
sess = tf.Session(config=config)
set_session(sess)

from keras.layers import Embedding, Dense, Bidirectional, CuDNNLSTM 
from keras.models import Sequential, model_from_json

from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.utils import multi_gpu_model

from classes.config import VOCABULARY_SIZE, EMBEDDING_LENGTH, LSTM_UNITS
from classes.config import EPOCHS, MAX_QUEUE_SIZE, WORKERS, LEARNING_RATE
from classes.config import GPU_NUMBER, MIN_LEARNING_RATE, SEQUENCE_LENGTH
#from classes.config import BLUE_SCORE

class Model():
    """
    Model of neural net to train on the dataset.
    """
    def __init__(self, advancement=None, manual=None):
        """
        model:
            The recurrent neural network to be trained.
        parallel_model:
            Model copied on used GPUs. 
        acc:
            history['acc'] returned when fitting the model.
        loss:
            history['loss'] returned when fitting the model.
        val_acc:
            history['val_acc'] returned when fitting the model.
        val_loss:
            history['val_loss'] returned when fitting the model.
        compiled:
            Status of the current model.
        strategy:
            Used to distribute training on several GPUs.        
        advancement:
            Used to store the coefficients at the end of each epoch.
        manual:
            The learning rate can be tuned manually at the end of each epoch
            by writing it in a file. "manual" is the file's name.
        """       
        self.model = None
        self.parallel_model = None
        self.acc = []
        self.loss = []
        self.val_acc = []
        self.val_loss = []      
        self.compiled = False
        self.advancement = advancement
        self.manual = manual
                
    def design(self, parallel=True):
        """
        Create a model to train.
        """
        with tf.device('/cpu:0'):
            
            self.model = Sequential()
            
            self.model.add(Embedding(VOCABULARY_SIZE, 
                                     EMBEDDING_LENGTH,
                                     input_length=SEQUENCE_LENGTH))
            
            for i in range(len(LSTM_UNITS)):
                units = LSTM_UNITS[i]
                
                # All LSTM layers return sequences except the last one. 
                ret = (i != len(LSTM_UNITS)-1)
                lstm_layer = CuDNNLSTM(units=units, return_sequences=ret)
                lstm_layer = Bidirectional(lstm_layer)
                self.model.add(lstm_layer)
                                
            self.model.add(Dense(VOCABULARY_SIZE, activation='softmax'))    
        
            print(self.model.summary())     
            
        if parallel is True:
            self.parallel_model = multi_gpu_model(self.model, gpus=GPU_NUMBER)
    
    def build(self):
        """
        Compile model.
        """

        optimizer = Adam(lr=LEARNING_RATE, 
                         clipnorm=1., 
                         beta_1=0.9, 
                         beta_2=0.999)
        #optimizer = RMSprop(lr=LEARNING_RATE, rho=0.9, clipnorm=1.0)
 
        self.parallel_model.compile(loss='categorical_crossentropy', 
                                    optimizer=optimizer, 
                                    metrics=['accuracy'])
        self.compiled = True
                
        
    def find_learning_rate(self, epoch):
        """
        Callback allowing to change the learning rate and store current 
        coefficients in a file at the end of each epoch. 
        """            
        new_lr = LEARNING_RATE
        
        if self.advancement is not None:
            try:
                # Save the current coefficients to a file. 
                file = "{0}/coefficients_epoch_{1}.h5"
                folder = self.advancement
                self.model.save_weights(file.format(folder, epoch))
            except Exception as exc:
                m = "Exception {0} when saving coefficients."
                print(m.format(repr(exc)))                    
                    
        if self.manual is not None:
            try:
                with open(self.manual, 'r') as manual_lr:
                    new_lr = float(manual_lr.read())
            except Exception as exc:
                m = "Exception {0} when reading the learning rate."
                print(m.format(repr(exc)))
                print("Learning rate: {0}.".format(new_lr))
        else:
            # Divide the learning rate by 10 every 5 epochs 
            # until reaching a chosen threshold.
            updated = LEARNING_RATE / (10.0 ** (epoch // 5))
            new_lr = max(updated, MIN_LEARNING_RATE)
            
        return new_lr
        
    def run_generator(self, training, validation):
        """
        Fit the model on a dataset with batches of data.
        Parameters training and validation must inherit from ku.Sequence.
        """
        
        if not self.compiled:
            self.build()
            
        train_steps = training.batch_number
        val_steps = validation.batch_number
               
        lr_schedule = LearningRateScheduler(self.find_learning_rate)
        
        history = self.parallel_model.fit_generator(generator = training,
                                      steps_per_epoch = train_steps,
                                      epochs = EPOCHS,
                                      verbose = 1,
                                      callbacks=[lr_schedule],
                                      validation_data = validation,
                                      validation_steps = val_steps,
                                      max_queue_size=MAX_QUEUE_SIZE,
                                      workers=WORKERS,
                                      use_multiprocessing=True)
        
        self.acc += history.history['accuracy']
        self.loss += history.history['loss']
        self.val_acc += history.history['val_accuracy']
        self.val_loss += history.history['val_loss']
                        
    def save(self, architecture, coefficients, history):
        """
        Save to disk the architecture and the coefficients, and the
        training history of the model for later use.
        """
        
        # Save the architecture of the model in json format
        model_architecture = self.model.to_json()
        with open(architecture, "w") as json_file:
            json_file.write(model_architecture)
            
        # Save the values of the weights in HDF5 format
        self.model.save_weights(coefficients)
        
        # Dump the training history
        with open(history, "wb") as his:
            pickle.dump(self.acc, his)
            pickle.dump(self.loss, his)
            pickle.dump(self.val_acc, his)
            pickle.dump(self.val_loss, his)
    
    def load(self, architecture, coefficients, history):
        """
        Load a previously trained model.
        """        
        
        # Load the architecture of a preexisting model
        with tf.device('/cpu:0'):   
            try:
                json_file = open(architecture, 'r')
                model_architecture = json_file.read()
                json_file.close()
                self.model = model_from_json(model_architecture)
                    
            except Exception:
                print("A problem occurred while loading model.")
                self.design(parallel=False)
                
            # Load weights into new model
            self.model.load_weights(coefficients)
                
        self.parallel_model = multi_gpu_model(self.model, gpus=GPU_NUMBER)
        
        try:
            # Load training history into model
            with open(history, "rb") as his:
                self.acc = pickle.load(his)
                self.loss = pickle.load(his)
                self.val_acc = pickle.load(his)
                self.val_loss = pickle.load(his)
                
        except Exception as exc:
            m = "Error loading history: {0}"
            print(m.format(repr(exc)))
                
    def predict_next_chord(self, predictors=[0], randomness=5, soprano=None):
        """
        Given a sequence of chords, predict the next one.
        Pick at random among the n best options, with n = randomness.
        If not None, the soprano note is given as a constraint.        
        """
        if not self.compiled:
            self.build()
        
        sequence = np.vstack([predictors])
        options = list(enumerate(self.model.predict(sequence)[0]))
        options.sort(reverse=True, key=lambda x:x[1])
        
        
        def is_eligible(chord, soprano):
            """
            Defines if chord is a potential candidate.
            """
            if chord[0] == soprano and 0 not in chord[1:]:
                return True
            else:
                return False
        
        if soprano is not None:
            chords = [Chord(model=chord[0]) for chord in options]
            chords = [chord.satb for chord in chords]
            eligible = [is_eligible(chord, soprano) for chord in chords]
            new_options = []
            
            for i in range(len(eligible)):
                if eligible[i] is True:
                    new_options.append(options[i])
                    
            chosen = np.random.randint(randomness)
            next_chord = new_options[chosen][0]
            
        else:
            chosen = np.random.randint(randomness)
            next_chord = options[chosen][0]
        
        return next_chord
    
    def create_music(self, predictors=None, how_many=100, randomness=5):
        """
        Given a sequence of chords, predict the next n ones, with
        n = how_many.
        """
        if not self.compiled:
            self.build()
        
        if predictors is None:
            predictors = [0] * SEQUENCE_LENGTH
        
        predicted = []
        while len(predicted) < how_many:
            next_chord = self.predict_next_chord(predictors, randomness)
            predicted.append(next_chord)  
            predictors.append(next_chord)  
            predictors = predictors[-SEQUENCE_LENGTH:]
                    
        return predicted
    
    def harmonize(self, soprano, randomness=1, dense=True):
        """
        Harmonize a voice of soprano. 
        """
        if not self.compiled:
            self.build()
        
        predictors = [0] * SEQUENCE_LENGTH
        harmonized = []
        
        for note in soprano:
            next_chord = self.predict_next_chord(predictors, randomness, note)
            harmonized.append(next_chord)  
            predictors.append(next_chord)  
            predictors = predictors[-SEQUENCE_LENGTH:]
                    
        return harmonized