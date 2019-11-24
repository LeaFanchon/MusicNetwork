#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objective : Harmonize a voice of soprano. 
"""

import numpy as np

from classes.network_training.model import Model
from classes.dataset_creation.observation import Observation
from classes.dataset_creation.chord import Chord
from classes.config import SEED

np.random.seed(SEED)

architecture = ""
coefficients = "models/advancement_6_LSTM_11_10_2019/coefficients_epoch_54.h5"
history = ""

model = Model()
model.load(architecture, coefficients, history)

soprano = [[1, 3, 5, 6, 8, 10, 12, 1, 1, 3, 5, 6, 8, 10, 12, 1], 
           [6, 8, 10, 11, 1, 3, 5, 1],
           [1, 3, 4, 6, 8, 9, 12, 1, 1, 11, 9, 8, 6, 4, 3, 1]]

for i in range(len(soprano)):

    harmonized = model.harmonize(soprano[i], randomness=1)
    
    harmonized = [Chord(model=chord) for chord in harmonized]
    observation = Observation(harmonized)
    
    # harmonized_folder: where to store the new generated midi file.
    harmonized_folder = 'data/midi_harmonized'
    
    # name_obs: name of the generated midi file.
    name_obs = '{0}/harmonized_{1}.midi'.format(harmonized_folder, i)
        
    observation.create_midi_file()
    observation.midi_file.save(name_obs)

print("Finished")