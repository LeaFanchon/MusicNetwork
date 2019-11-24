#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 18:18:57 2019

@author: lea
"""

import sqlite3
import pickle
import os

from sklearn.utils import shuffle
from classes.config import BATCH_SIZE

class DatasetPreprocessor():
    
    def __init__(self, file, folder, total_size=None):
        """
        file:
            Where the compressed dataset is stored.
        folder:
            Folder where to save the database.
        database:
            Connection to the database where the new batches of 
            observations are stored.
        total_batches:
            Total number of batches.  
        total_size:
            Total number of observations to put in the training 
            and validation datasets.  
        shuffled:
            Shuffled range [|0, total_size|[.            
        """
        
        self.file = file
        self.total_size = total_size
                
        self.folder = folder
        self.database = None
        
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        if self.total_size is None: 
            self.find_size()
        
        self.total_batches = (self.total_size // BATCH_SIZE) + 1
        self.shuffled = []
        
    def find_size(self):
        """
        If the size was not given to the constructor as a parameter, 
        browse once through the dataset to count the number of observations.
        """
        self.total_size = 0
        
        with open(self.file, 'rb') as dataset:
            while True:
                try:
                    pickle.load(dataset)
                    self.total_size += 1
                except EOFError:
                    break
        
    def preprocess_dataset(self):
        """
        Encapsulates all the dataset preprocesing steps.
        """
        self.prepare_database()
        self.one_observation_per_file()
        self.shuffle_dataset()
        self.one_batch_per_file()
        self.close_database()

    def prepare_database(self):
        """
        Create the database to store batches of observations in self.folder. 
        Create the table observations. 
        """
        # Create database
        name = "{0}/database_{1}.db"
        self.database = sqlite3.connect(name.format(self.folder, BATCH_SIZE))
        c = self.database.cursor()
        
        # Create table
        create = "CREATE TABLE observations "
        create += "(id INTEGER, observation BLOB, batch INTEGER)"
        c.execute(create)
        
        # Create a unique index on the id column
        c.execute("CREATE UNIQUE INDEX obs_index ON observations (id)")
        
        # Create an index on the batch column
        c.execute("CREATE INDEX batch_index ON observations (batch)")
        
        # Don’t sync to disk after every insert
        c.execute("PRAGMA synchronous = OFF")
        
        # Don't store rollback journal
        c.execute("PRAGMA journal_mode = OFF")
        
    def one_observation_per_file(self):
        """
        Read through input file containing the compressed dataset.
        For each observation, store it in the observations table in 
        self.database.
        """
        
        index = 0
        
        c = self.database.cursor()
        insert = "INSERT INTO observations ('id', 'observation', 'batch') "
        insert +=  "VALUES (?, ?, ?)"
        
        with open(self.file, 'rb') as dataset:
             
            while index < self.total_size:
                try:
                    
                    # Load the observation
                    observation = pickle.load(dataset)
                    
                    # Dump it in a variable
                    p = pickle.HIGHEST_PROTOCOL
                    pickled = pickle.dumps(observation, protocol=p)
                    
                    # Insert into table
                    c.execute(insert, (index, sqlite3.Binary(pickled), 0))
                    
                    index += 1  
                            
                except EOFError:
                    break
            
        # In case the dataset contained less than total_size observations.
        self.total_size = index
        self.total_batches = (self.total_size // BATCH_SIZE) + 1
        
    def shuffle_dataset(self):
        """
        Shuffle the dataset.
        
        As the observations are stored in self.database in individual files 
        named "n.pickle", with n in [|0, total_size|[, shuffling the dataset
        means here just shuffling the integers in [|0, total_size|[.
        """
        self.shuffled = shuffle(range(self.total_size))        
            
    def one_batch_per_file(self):
        """        
        Observations are shuffled and distributed in batches of size 
        BATCH_SIZE. In practice, the column batch of the observations table 
        is updated with a batch number.
        """
        
        index = 0
        batch_index = 0
        
        c = self.database.cursor()        
        update = "UPDATE observations SET batch = {0} WHERE id = {1}"
        
        for i, index in enumerate(self.shuffled):
            
            if i != 0 and i % BATCH_SIZE == 0:
                batch_index += 1
            
            c.execute(update.format(batch_index, index))            

    def close_database(self):
        """
        Commit changes and close the connection.
        """
        # Commit changes
        self.database.commit()
        
        # Close connection
        self.database.close()