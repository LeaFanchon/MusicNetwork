#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:52:48 2019

@author: lea
"""
import keras.utils as ku 
import numpy as np
import sqlite3
import pickle

from classes.config import VOCABULARY_SIZE, BATCH_SIZE, SEQUENCE_LENGTH

class Batch():
    """
    One batch of data.
    """
    def __init__(self, folder, index):
        """
        sequence_length:
            Maximum length of a sequence in the loaded batch.
            All the smaller sequences will be padded to reach this length.
        predictors:
            Features of the loaded batch. 
            Sequences of chords - each chord being represented by an integer.
        labels:
            Targets.
            For a given predictor, its target is the chord that followed 
            the last predictor's chord in the original midi file used
            to create this observation.   
        index:
            Index of the file from which the current batch must be loaded. 
        folder:
            Where the sqlite3 database is stored.        
        database:
            Connection to the observations database.
        size:
            Size of the current batch.
        """
        self.sequence_length = 0
        self.folder = folder
        self.database = None
        self.index = index
        self.size = 0

        self.predictors = [None] * BATCH_SIZE
        self.labels = [None] * BATCH_SIZE
        
        # Generate the batch
        self.prepare_database()
        self.load()
        self.find_sequence_length()
        self.clean()
        self.close_database()
        
    def prepare_database(self):
        """
        Connect to the database. 
        """
        name = "{0}/database_{1}.db"
        self.database = sqlite3.connect(name.format(self.folder, BATCH_SIZE))
        
    def load(self):
        """
        Read the batch into memory.
        - The features are loaded in predictors
        - The targets are loaded in labels
        """
        
        select = "SELECT observation FROM observations WHERE batch = {0}"
        select = select.format(self.index)
        c = self.database.cursor()

        try:
            for obs in c.execute(select):      
                observation = pickle.loads(obs[0])
                self.predictors[self.size] = observation[0]                                
                self.labels[self.size] = observation[1]
                self.size += 1
                
        except Exception as exc:
            message = "Error in batch generation {0}: {1}"
            print(message.format(self.index, repr(exc)))
            raise

        self.predictors = self.predictors[:self.size]
        self.labels = self.labels[:self.size]
        
    def find_sequence_length(self):
        """
        Find the maximum length of a sequence in the loaded batch.
        """
        if SEQUENCE_LENGTH is None:
            lengths = [len(obs) for obs in self.predictors]
            self.sequence_length = max(lengths)
        else:
            self.sequence_length = SEQUENCE_LENGTH
            self.predictors = [p[-SEQUENCE_LENGTH:] for p in self.predictors]
        
    def clean(self):
        """
        Once the batch is loaded, finish preprocessing it :
            - Pad the predictors so they're all the same length
            - Turn the vector of labels into a set of one-hot vectors.
        """        
        # Pre-pad all the sequences so they are the same length
        for p in range(self.size):
            pad = self.sequence_length - len(self.predictors[p])
            padding =  [0] * pad
            self.predictors[p] = padding + self.predictors[p]
            
        # Convert to numpy array
        self.predictors = np.vstack(self.predictors)
                
        # Turn the labels to categoricals
        voc = VOCABULARY_SIZE
        self.labels = ku.to_categorical(self.labels, num_classes=voc)
        
    def close_database(self):
        """
        Close the connection.
        """
        self.database.close()