#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:31:28 2019

@author: lea
"""
 
import keras.utils as ku
from classes.network_training.batch import Batch
from classes.config import BATCH_SIZE

class DataGenerator(ku.Sequence):
    
    def __init__(self, folder, batch_range):
        """
        folder:
            Where the observations are stored, each in an individual file.        
        batch_range:
            Batch numbers belonging to this data generator.
        batch_number:
            Number of batches.
        batch_size:
            Size of each batch.
        """
        
        # Initialize general attributes
        self.folder = folder
        self.batch_range = batch_range
        self.batch_number = len(self.batch_range)
        self.batch_size = BATCH_SIZE

    def __len__(self):
        """
        Returns the number of batches.
        """
        return len(self.batch_range)
        
    def __getitem__(self, idx):
        """
        Returns one batch of data.
        """
        batch = Batch(self.folder, idx)      
        return batch.predictors, batch.labels
    