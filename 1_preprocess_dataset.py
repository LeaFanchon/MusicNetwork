#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objectives : - Store the dataset in a sqlite3 database, in a table called 
               observations with three columns : id, observation, batch. 
               The aim is to allow training the neural network using batches
               of data, in order to avoid loading all the data into memory
               at the same time.
"""

from classes.network_training.datasetpreprocessor import DatasetPreprocessor

# file: where all the observations have been stored.
file = "data/dataset.pickle"

# folder: where to create a database to store the observations.
folder = "data/database"

# total_size: set it to None if all the dataset is needed. 
total_size = None

dataset = DatasetPreprocessor(file=file, 
                              folder=folder,
                              total_size=total_size)

dataset.preprocess_dataset()

print("Dataset preprocessed.")