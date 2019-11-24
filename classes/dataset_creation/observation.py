#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 23:11:01 2019
@author: lea
"""

import numpy as np

from mido import Message
from mido import MidiFile
from mido import MidiTrack

from classes.dataset_creation.chord import Chord

from classes.config import TRACK_NUMBER
from classes.config import FEATURES_MIN_NUMBER
from classes.config import OCTAVES
from classes.config import SLOWING_FACTOR
from classes.config import TRACK_DENSITY
from classes.config import DENSITY
from classes.config import HARMONY_MIN_GRADE

class Observation():
    """
    One observation consists in:
        - n features : n consecutive chords, 
        - 1 target : the following chord.
    """
     
    def __init__(self, chords):
        """
        Attributes:
        - features: 
            List of a fixed number of consecutive n-notes chords,
            with n = TRACK_NUMBER.
        - targets: 
            The target is the next chord after the features in the 
            original piece.
        - midi_file: 
            Midi file created from the features and targets in order
            to listen to the observation.
        - Slowing_factor: 
            To make the observation easier to listen to.
        """ 
        self.features =  chords[:-1]
        
        # If there are too few features, add empty chords at the beginning
        pad_length = max(0, FEATURES_MIN_NUMBER - len(self.features))
        pad = [Chord() for t in range(pad_length)]
        self.features = pad + self.features
        
        self.target = chords[-1]
        self.midi_file = MidiFile()
        self.slowing_factor = SLOWING_FACTOR

    def is_valid(chords):
        """
        Check that chords is a valid list to create an observation.
        """           
        # Does the list contain at least one non empty chord? 
        empty = [c for c in chords if c.is_empty()]
        if len(empty) == len(chords):
            return False
        
        # Is the list of chords dense enough to become an Observation?
        density = Observation.get_density(chords)
        if density < DENSITY:
            return False
        
        # Do all the tracks have a sufficiently high rate of non-empty notes?
        satbs = [chord.satb for chord in chords]
        tracks = np.array(satbs).T
        for track in tracks:
            significant = [note for note in track if note != 0]
            
            if len(significant) / len(track) < TRACK_DENSITY:
                return False                

        # Are the chords interesting enough?
        # Rate of chords that have at least (TRACK_NUMBER/2) + 1 
        # distinct notes. 
        chords = [c for c in satbs if c != [0] * TRACK_NUMBER]
        treshold = (TRACK_NUMBER/2.0) + 1 
        complexity = [len(set(c)) >= treshold for c in chords]
        
        if (sum(complexity) / len(complexity)) < HARMONY_MIN_GRADE:
           return False
        
        return True
    
    def get_density(chords):
        """
        Get the rate of non-empty chords in chords.
        """
        total = len(chords)
        empty_chords = [chord for chord in chords if chord.is_empty()]
        nb_empty = len(empty_chords)
        
        return 1 - (nb_empty / total)
    
    def create_midi_file(self):
        """
        Create a valid midi file to listen to the observation.
        """
        # We add the target twice so it is discernible in the audio file
        chords = self.features + [self.target] * 2
        chords = Observation.widen_note_range(chords)      
                
        midi_tracks = []
        last_tick = []
        
        for track in range(TRACK_NUMBER):
            midi_tracks.append(MidiTrack())
            last_tick.append(0)
            
        slow = self.slowing_factor
        
        for tick, chord in enumerate(chords):
            for t, track in enumerate(midi_tracks):
                if chord.midi[t] != 0:                    
                    note = chord.midi[t]
                    
                    # Convert absolute time to relative time
                    time = (tick - last_tick[t]) * slow                    
                    msg = Message(type='note_on', note=note, time=time)
                    
                    track.append(msg)    
                    last_tick[t] = tick
                   
        for track in midi_tracks:
            self.midi_file.tracks.append(track)
            
    def widen_note_range(chords):
        """
        Chords is a list of chord objects which notes lie in the range [0:12]. 
        
        This method adds movement within each track so that it is not confined 
        to the same octave all along:
            - The first note of each track is transposed in a dedicated octave
            - Each following note is transposed to the octave that makes it 
              closest to the previous note, provided it doesn't get out of the
              available midi range.
        """
        
        chords = np.array([chord.satb for chord in chords])
        tracks = chords.T         
        first_note_index = {}
        
        # For each track, identify the index of the first significant note.
        for t, track in enumerate(tracks):            
            first_note_index[t] = 0            
            for n, note in enumerate(track):
                if note != 0:
                    first_note_index[t] = n     
                    
                    # Map the first significant note to an audible octave.
                    tracks[t][n] += 12 * OCTAVES[t]
            
        for (t, track) in enumerate(tracks):
            
            previous_note = track[first_note_index[t]]
            
            # Range to which the track will be projected.
            interval = range(OCTAVES[t], OCTAVES[t] + 3)
            
            # Map each note to the octave where it is the closest to the 
            # previous note of the track, provided it stays in an admissible 
            # range.
            for n, note in enumerate(track[first_note_index[t] + 1 :]):
                if 0 not in [note, previous_note]:
                    
                    note = Observation.closest(note, previous_note, interval)
                    previous_note = note
                    track[n + first_note_index[t] + 1] = note
                    
        return [Chord(midi=chord) for chord in tracks.T]
            
    def closest(note, previous_note, admissible_range):
        """
        With note being an integer between 1 and 12, return its projection
        into the octave where it is the closest to previous_note, provided
        that it remains in the provided admissible range.
        """  
        def transposition(i):
            """
            Transpose 'note' i octaves higher.
            """
            return i * 12 + note
        
        def distance(i):
            """
            Return the interval between the transposed note and previous note.
            """
            return np.abs(transposition(i) - previous_note)
        
        candidates = np.array([distance(i) for i in admissible_range])
        
        return transposition(np.argmin(candidates) + admissible_range[0])
                