#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objective : Train a model to generate music and store it to disk.
"""
import sqlite3

from classes.network_training.datagenerator import DataGenerator
from classes.network_training.model import Model

from classes.config import TRAIN_PERCENTAGE
from classes.config import BATCH_NUMBER, BATCH_SIZE

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

# train_split: number of batches belonging to the training set.
train_split = int(BATCH_NUMBER * TRAIN_PERCENTAGE)

# train_range: range of batch numbers belonging to the training set.
train_range = range(train_split)

# validation_range: range of batch numbers belonging to the validation set.
validation_range = range(train_split, BATCH_NUMBER)

# generators: to be passed to the model to train it with batches of data.
training_generator = DataGenerator(folder, train_range)
validation_generator = DataGenerator(folder, validation_range)

# Save the coefficients in execution_folder at the end of each epoch. 
advancement = "models/advancement"

# Tune the learning rate manually by writing it in the file "manual"
manual = "models/learning_rate.txt"

model = Model(advancement, manual)

# Initial trained model's architecture coefficients and history.
# We want to pick up training where we left it off. 
init_architecture = ""
init_coefficients = "models/advancement_6_LSTM_11_10_2019/coefficients_epoch_54.h5"
init_history = ""

model.load(init_architecture, init_coefficients, init_history)
#model.design()
model.build()

model.run_generator(training_generator, validation_generator)

# Final trained model's architecture coefficients and history.
architecture = "models/architecture.json"
coefficients = "models/coefficients.h5"
history = "models/history.pickle"

model.save(architecture, coefficients, history)

print("Training completed.")