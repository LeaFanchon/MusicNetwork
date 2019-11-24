#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objective : Load a trained model into memory and use it to generate 
            some music.
"""
import random
import sqlite3
import numpy as np

from classes.dataset_creation.observation import Observation
from classes.dataset_creation.chord import Chord
from classes.network_training.model import Model
from classes.network_training.batch import Batch

from classes.config import TRAIN_PERCENTAGE, BATCH_NUMBER, BATCH_SIZE
from classes.config import SEED, SEQUENCE_LENGTH

np.random.seed(SEED)

# Final trained model's architecture coefficients and history.
#architecture = "models/architecture.json"
#coefficients = "models/coefficients.h5"
#history = "models/history.pickle"

architecture = ""
coefficients = "models/advancement_6_LSTM_11_10_2019/coefficients_epoch_54.h5"
history = ""


model = Model()
model.load(architecture, coefficients, history)

# folder: where the dataset is stored in a sqlite3 database.
folder = "data/database"

# If BATCH_NUMBER is None, all the dataset shall be used for training. 
# Find the total number of batches. 
if BATCH_NUMBER is None:
    name = "{0}/database_{1}.db"
    database = sqlite3.connect(name.format(folder, BATCH_SIZE))
    c = database.cursor()
    for res in c.execute("SELECT MAX(batch) FROM observations"):
        BATCH_NUMBER = res[0] + 1
    database.close()


# batch_index: index of random batch belonging to the validation set.
train_split = int(BATCH_NUMBER * TRAIN_PERCENTAGE)
batch_index = random.randint(train_split, BATCH_NUMBER)
batch = Batch(folder, batch_index)

for i in range(10):
    
    # observation_index: index of a random observation in the chosen batch.
    observation_index = random.randint(0, BATCH_SIZE-1)
    
    predictors = batch.predictors[observation_index]
    labels = batch.labels[observation_index]
    
    # Remove empty chords at the beginning of predictors.
    empty = 0
    while empty < len(predictors) and predictors[empty] == 0:
        empty+=1
    predictors = predictors[empty:]
    
    if len(predictors) > 0:
        
        # Pad / truncate the sequence of predictors to the expected length
        truncate = list(predictors[-SEQUENCE_LENGTH:])
        pad = [0]*(SEQUENCE_LENGTH-len(truncate))
        
        predictors = pad + truncate
            
        # Let the model add the next n chords.
        n = 100
        predicted = model.create_music(predictors=predictors, how_many=n)
        
        predicted = list(predicted)
        predicted = [Chord(model=p) for p in predicted]
        observation = Observation(predicted)
        
        # generated_folder: where to store the new generated midi file.
        generated_folder = 'data/midi_generated'
        
        # name_obs: name of the generated midi file.
        name_obs = '{0}/generated_{1}.midi'.format(generated_folder, i)
            
        observation.create_midi_file()
        observation.midi_file.save(name_obs)
        
        # Only the predictors:          
        predictors = [Chord(model=p) for p in predictors]
        predictors_observation = Observation(predictors)
        name_predictors = '{0}/predictors_{1}.midi'.format(generated_folder, i)
        predictors_observation.create_midi_file()
        predictors_observation.midi_file.save(name_predictors)
    
print("Finished")
