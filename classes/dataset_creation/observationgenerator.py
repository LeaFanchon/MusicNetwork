#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:31:37 2019

@author: lea
"""
import pickle

from mido import MidiFile
from multiprocessing import Lock
from multiprocessing import Process
from classes.dataset_creation.midiprocessor import MidiProcessor

lock_target = Lock()
lock_stats = Lock()
lock_console = Lock()

class ObservationGenerator(Process):
    """
    Generate observations from a list of raw midi files. 
    """
    def __init__(self, midi_files, target, stats, listen_folder, identifier):
        """
        midi_files:
            List of midi files from which observations can be extracted.
        target_file:
            File where the created observations must be written. 
        stats:
            Simple object to keep track of the dataset's generation 
            progression. 
        listen_folder: 
            Some midi files will be created in listen_folder in order to 
            be able to listen to a fraction of the new dataset.       
        identifier:
            Thread identifier. 
        stop:
            Used to terminate the thread.
        """
        Process.__init__(self)
        self.midi_files = midi_files
        self.target_file = target
        self.stats = stats
        self.listen_folder = listen_folder
        self.identifier = identifier
        self.stop = False
                    
    def run(self):
        """
        Browse through the list of candidate midi files.
        For each of them:
            - Create out of it as many observations as possible
            - Store them in self.file
        """
        
        total = len(self.midi_files)
        
        while self.stats.current_index.value < total and not self.stop:
            
            # Find the next midi file to process.
            lock_stats.acquire()
            index = self.stats.current_index.value
            self.stats.current_index.value += 1
            lock_stats.release()
                
            # Extract observations from the midi file.
            if index < total: 
                
                name = self.midi_files[index]
                midi = None
                try:
                    midi = MidiFile(name)
                except Exception as exc:         
                    m = '{0} {1} {2}'
                    lock_console.acquire()
                    print(m.format(self.identifier, name, repr(exc)))
                    lock_console.release()
                                        
                if midi is not None and MidiProcessor.is_valid(midi):
                    try:
                        processed = MidiProcessor(midi)   
                        midi_observations = processed.create_observations()
                        self.save_to_file(midi_observations)
                                                
                    except Exception as exc:
                        m = '{0} {1} {2}'
                        lock_console.acquire()
                        print(m.format(self.identifier, name, repr(exc)))
                        lock_console.release()
                        
                lock_stats.acquire()
                self.stats.done.value += 1
                lock_stats.release()
                
                # Print progress 
                if index % 100 == 0:
                    done = 100 * self.stats.done.value
                    done /= self.stats.total.value
                    m = '{0} - {1} - {2} - {3:.3f}%'
                    lock_console.acquire()
                    print(m.format(self.identifier, name, index, done))
                    lock_console.release()
                        
    def save_to_file(self, midi_observations):
        """
        Compress and save the newly created observations to a physical file.
        """
        for observation in midi_observations:

            # Compress each chord in the observation features and target.            
            model_features = [f.model for f in observation.features]
            model_target = observation.target.model  
            new_observation = [model_features, model_target]
            
            # Save to file
            lock_target.acquire()
            pickle.dump(new_observation, self.target_file)
            lock_target.release()
            
            # Create midi files of the some of the new observations. 
            lock_stats.acquire()
            if self.stats.observations.value % 10000 == 0:
                listen = self.listen_folder
                name_obs = '{0}/file_{1}.midi'
                incr = self.stats.observations.value
                observation.create_midi_file()
                observation.midi_file.save(name_obs.format(listen, incr))
            
            self.stats.observations.value += 1
            lock_stats.release()
    
    def terminate(self):
        """
        Stop the execution of the thread.
        """
        self.stop = True