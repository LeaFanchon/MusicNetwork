#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os

from classes.dataset_creation.statistics import Statistics
from classes.dataset_creation.observationgenerator import ObservationGenerator

"""
Transform midi files into a suitable and normalized dataset
that can be fed to a neural net. 
"""


class Dataset():
    """
    Collection of observations extracted from all the midi files that
    can be found in a given directory and its subdirectories.
    The observations are not kept in memory, but immediately written in 
    the file provided to the constructor.
    """
   
    def __init__(self, directory, file_name, listen_folder):
        """
        directory:
            Directory to browse to create the dataset. 
        file_name:
            File in which the new dataset should be saved.
        listen_folder: 
            Some midi files will be created in listen_folder in order to 
            be able to listen to a fraction of the new dataset.       
        all_files:
            List of all the midi files to be included in the dataset. 
        statistics:
            Numbers to keep track of the progress of the generation of 
            the dataset.
        """
        
        self.directory = directory
        self.file_name = file_name
        self.listen_folder = listen_folder
        self.all_files = []
        self.statistics = Statistics() 
        
        self.browse()
            
    def browse(self, directory=None):
        """
        Recursively browse a directory and its subdirectories.
        Find:
            - the total number of midi files in self.directory
            - the list of the midi files to be used in the dataset
        """
        
        if directory is None:
            directory = self.directory
            
        basepath = '{}/'.format(directory)
    
        for entry in os.listdir(basepath):
        
            # Count midi files and store their names in a list
            if re.search(r'\.mid', entry):
                self.statistics.total.value += 1
                full_name = os.path.join(basepath, entry)
                self.all_files.append(full_name)
                
             # If there are subdirectories in the directory, visit them too. 
            if not os.path.isfile(os.path.join(basepath, entry)):                    
                self.browse(basepath + entry)  
        
        
    def create_observations(self, thread_number=12):
        """
        Create threads to browse through the list of candidate midi files.
        For each midi file:
            - Create out of it as many observations as possible
            - Store them in self.target
        """
        
        # Open the target file.        
        with open(self.file_name,"wb") as target:
            
            threads = []
            files = self.all_files
            stats = self.statistics
            rep = self.listen_folder
            
            try:
                # Create the threads
                for t in range(thread_number):
                    thread = ObservationGenerator(files, target, stats, rep, t)
                    threads.append(thread)
                            
                # Start the threads
                for t in range(thread_number):
                    threads[t].start()
                
                # Join the threads
                for t in range(thread_number):
                    threads[t].join()
                    
                m = "Dataset successfully created and stored in {}"
                print(m.format(self.file_name))
                
            except KeyboardInterrupt:
                
                for t in range(thread_number):
                    threads[t].terminate()
                
                print("Termination of the threads; might take a few minutes.")
                
                for t in range(thread_number):
                    threads[t].join()
            
