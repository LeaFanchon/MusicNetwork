#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:11:57 2019

@author: lea
"""

from multiprocessing import Value

class Statistics:
    """
    Keep track of the progress of the generation of the dataset.
    """
    def __init__(self):
        """
        total:
            Total number of midi files to be processed. 
        done:
            Number of files that have already been processed.
        observations:
            Number of observations generated.
        current_index:
            Next midi file to visit. 
        """
        self.total = Value('i', 0)
        self.done = Value('i', 0)
        self.observations = Value('i', 0)
        self.current_index = Value('i', 0)