#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 23:11:19 2019
@author: lea
Global variables used to design the dataset and train the network.
"""

# For reproducibility.
SEED = 1

#---------------------------------------------------------------
# GENERATING THE DATASET
#---------------------------------------------------------------

# If there are less than FEATURES_MIN_NUMBER features, the features will 
# be padded with empty chords
FEATURES_MIN_NUMBER = 1

# When creating a new Observation, the parameter 'chords' passed to the
# constructor has to have a rate of non-empty chords at least equal to
# DENSITY.
DENSITY = .6

# Number of tracks, meaning number of notes in each chord.
TRACK_NUMBER = 4

# Minimum admissible track density for each track of an Observation.
# When creating a new Observation, each track of the parameter 'chords' 
# passed to the constructor has to have a rate of non-empty notes at least 
# equal to TRACK_DENSITY.
TRACK_DENSITY = .4

# Minimum admissible rate of chords that have at least (TRACK_NUMBER/2) + 1 
# distinct notes. 
HARMONY_MIN_GRADE = .5

# When trying to listen to an Observation, define, for each track,
# the range to which it will be projected when a midi file is created.
OCTAVES = [6, 6, 5, 4]

# When trying to listen to an Observation, define a slowing factor that will
# add space between chords and thus make the piece less dense.
SLOWING_FACTOR = 300

# When converting an observation from relative to absolute time, 
# maximum acceptable number of time ticks of the converted file.
MAX_DURATION = 10000000

#---------------------------------------------------------------
# TRAINING THE NETWORK 
#---------------------------------------------------------------

# Number of GPUs to be used for training. 
GPU_NUMBER = 3

# While the model is training, size of the queue constituted by the CPU
MAX_QUEUE_SIZE = 10

# CPU threads working to prepare the batches to be fed to the GPUs.
WORKERS = 12

# Size of the batches fed to the neural network.
BATCH_SIZE = 1024

# Total number of batches to be used, including both training and 
# validation sets. Set to None if all batches must be used.
BATCH_NUMBER = None

# Observations will be padded or truncated until they contain 
# SEQUENCE_LENGTH chords. Set to None if sequence length is not fixed.
SEQUENCE_LENGTH = 100

# Percentage of observations that will be attributed to the training set. 
# The rest will belong to the validation set. 
TRAIN_PERCENTAGE = 0.9

# Total number of possible chords.
VOCABULARY_SIZE = 13 ** 4

# Size of the embedding layer's output.
EMBEDDING_LENGTH = 13 * 4

# Number of units of the LSTM layers
LSTM_UNITS = [64, 64, 64, 64, 64, 64]

# On how many epochs the training should occur
EPOCHS = 300

# The initial learning rate while training the model
LEARNING_RATE = 1e-3

# The initial learning rate changes while training
# Minimum value it should not get less than:
MIN_LEARNING_RATE = 1e-3

# Recurrent dropout.
DROPOUT = 0.1

# Blue score parameter for generation
BLUE_SCORE = 3