#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objective : Extract observations to be fed to a neural network from a corpus 
            of midi files.
            
            Each observation consists in : 
                - The features: a sequence of N-notes chords
                - The target: the following N-notes chord.
"""

import numpy as np
from classes.dataset_creation.dataset import Dataset
from classes.config import SEED

np.random.seed(SEED)

# directory: folder where the corpus of midi files is stored.
directory = "data/midi_raw"

# file: where the new observations will all be stored.
file = "data/dataset.pickle"

# listen_folder: some midi files will be created in listen_folder in order to
#                be able to listen to a fraction of the new dataset.
listen_folder = "data/midi_examples"

# Create the dataset.
ds = Dataset(directory, file_name=file, listen_folder=listen_folder)
ds.create_observations(thread_number=12)

print("End")
