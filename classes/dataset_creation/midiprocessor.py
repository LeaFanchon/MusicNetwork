#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 23:10:34 2019
@author: lea
"""

import re
import numpy as np

from classes.dataset_creation.observation import Observation
from classes.dataset_creation.chord import Chord

from classes.config import TRACK_NUMBER
from classes.config import MAX_DURATION

class MidiProcessor():
    """
    Take a midi file as input and extract relevant information from it.
    The aim of the processing of the file is to get from it the highest
    possible number of normalized observations to add to the dataset. 
    """
   
    def __init__(self, midi):
        """
        Attributes:
            - midi: 
                Midi file
            - significant_tracks: 
                The tracks capturing the most movement in the raw midi file.      
            - absolute_times: 
                For each significant track, stores the absolute time of each 
                'note_on' event, with the corresponding note in 'C' key.
            - chords:
                Final formatting of the piece. For each time unit, stores a 
                chord of TRACK_NUMBER notes. 
        """
            
        self.midi = midi
        self.significant_tracks = []
        self.absolute_times = []
        self.chords = []
        
        # Feed attributes
        self.choose_significant_tracks()
        self.convert_to_absolute_time()
        self.find_chords()        
        
    def is_valid(midi):
        """
        Check that the midi file can be used in the dataset.
        Has the midi at least TRACK_NUMBER tracks containing note_on events? 
        """
        tracks = midi.tracks
                
        valid_tracks = 0
        for track in tracks:
            msgs = [msg for msg in track if msg.type == 'note_on']
            if len(msgs) > 0:
                valid_tracks += 1
            if valid_tracks >= TRACK_NUMBER:
                break
            
        # Has the midi at least TRACK_NUMBER tracks containing a reasonable
        # number of note_on events? 
        if valid_tracks < TRACK_NUMBER:
            return False
                
        return True
                                
    def choose_significant_tracks(self):
        """
        Find the tracks that capture most of the movement of the piece. 
        If more than TRACK_NUMBER tracks seem to be string instruments, 
        search only among them.         
        """             
        tracks = self.midi.tracks
                
        # Count the number of string instruments are present 
        track_names = ''.join([t.name.lower() for t in tracks])
        regex = r"viol|contrab"
        string_number = len(re.findall(regex, track_names))
        
        # If more than TRACK_NUMBER tracks seem to be string instruments, 
        # search only among them.         
        if string_number >= TRACK_NUMBER:
            tracks = [t for t in tracks if re.search(regex, t.name.lower())]
            
        # For each eligible track, get a measure of its importance
        # and the mean of its notes.
        tracks = [MidiProcessor.track_importance(t) for t in tracks]
        
        # Sort by decreasing importance
        tracks.sort(reverse=True, key=lambda x:x[1])
                
        # Select the TRACK_NUMBER candidates with highest importance
        tracks = tracks[:TRACK_NUMBER]
        
        # Sort them by decreasing mean pitch.
        tracks.sort(reverse=True, key=lambda x:x[2])
        
        # Select the tracks and store them in the instance attribute
        self.significant_tracks = [t[0] for t in tracks]
    
    def track_importance(track):
        """
        Return a 3 elements list with:
            - the track given as parameter
            - a measure of importance of the track. 
            - the mean of the notes present in the track.
        """
        notes = [m.note for m in track if m.type == 'note_on']
        if notes:
            importance = np.std(notes) * len(notes)
            mean = np.mean(notes)
        else:
            importance = 0
            mean = -1           
        
        return [track, importance, mean]
    
    def convert_to_absolute_time(self):
        """
        For each significant track, store the absolute time of each note_on
        event. 
        """
        self.absolute_times = [MidiProcessor.track_to_absolute_time(t) 
                               for t in self.significant_tracks]
     
    def track_to_absolute_time(track):
        """
        For the given track, calculate the absolute time of each note_on
        event. If two notes are played simultaneously in the same track, 
        take only one of them.
        """
        absolute_events = []
        absolute_time = 0
        previous_event = None
        
        for msg in track:
            
            # If msg.time > 1, it means that the previous event lasted
            # at least one unit of time.
            if msg.time > 1 and previous_event is not None:
                absolute_events.append(previous_event)
                previous_event = None
            
            absolute_time += msg.time
            
            # Is msg a candidate to be added to the list of events?
            if msg.type == 'note_on':
                previous_event = [absolute_time, msg.note]
                
        # Add the last event of the track
        if previous_event is not None:
            absolute_events.append(previous_event)
            
        return absolute_events
                               
    def find_chords(self):
        """
        Having found the absolute time on which each event of the
        significant tracks is played, reformat the piece to have, for 
        each tick (meaning each indivisible time unit), a chord of 
        TRACK_NUMBER notes containing the note played by each track  
        on that tick.
        - The notes are integers between 1 and 128. 
        - The event 'no note is played on that tick' is encoded by 0. 
        """
        times = self.absolute_times
        
        #  all_timestamps : all the ticks on which a note is played
        all_timestamps = []
        
        # timestamps_by_track : same but separated by track
        timestamps_by_track = {}
        
        for t, track in enumerate(times):
            all_timestamps += [event[0] for event in track]
            timestamps_by_track[t] = [event[0] for event in track]
        
        if max(all_timestamps) - min(all_timestamps) > MAX_DURATION:
            raise ValueError("Exceeded maximum duration.")
        
        for tick in range(min(all_timestamps), max(all_timestamps) + 1):
            
            # Create a chord object for each time tick.
            midi_notes = [0] * TRACK_NUMBER
            for t, track in enumerate(times):
                if tick in timestamps_by_track[t]:
                    index = timestamps_by_track[t].index(tick)
                    midi_notes[t] = track[index][1] + 1
            
            chord = Chord(midi=midi_notes)
            self.chords.append(chord)
                   
    def create_observations(self, chords=None):
        """
        From a list of chords such as the one returned by find_chords, 
        extract recursively and return a collection of not-too-sparse 
        Observations.
        """
        
        if chords is None:
            chords = self.chords
        
        # Return if chords is empty
        if not chords:
            return []
        
        # Return if chords contains only empty chords
        empty = [chord for chord in chords if chord.is_empty()]
        if len(chords) == len(empty):
            return []
        
        observations = []
        
        chords = MidiProcessor.densify_chords(chords)

        # Is the list of chords dense enough to become an Observation?
        if Observation.is_valid(chords):
            
            # First transpose it to the 12 possible tonalities.
            transpositions = []
            for step in range(12):
                t = [chord.transpose(step) for chord in chords]
                transpositions.append(t)
                
            # Then, for each one of the chords, create 12 observations
            # in which it is the target, one for each possible tonality. 
            for (c, chord) in enumerate(chords):
                for step in range(12):
                    transposed = transpositions[step][:c+1]
                    if Observation.is_valid(transposed):
                        observations.append(Observation(transposed))
                        
            return observations
            
        elif len(chords) > 1:
            # We split it in two equal halves and repeat the densification
            # process.            
            half = int(len(chords)/2) 
            first_half = self.create_observations(chords[:half])
            second_half = self.create_observations(chords[half:])
            return first_half + second_half
        
        else:
            return []
        
    def densify_chords(chords):
        """
        Compress a list of chords.
        Return a list of chords with at least one pair of consecutive
        non-empty chords.
        
        For instance, if N represents a non-empty chord and E an empty chord: 
        N-E-E-E-N-E-N-E-E-E-E-N  will be densified as  N-E-E-N-N-E-E-E-N.
        """        

        # empty_counter: 
        # While browsing the list, empty_counter is the number of consecutive
        # empty chords found since visiting the last non-empty chord.          
        empty_counter = 0
        
        # minimum_empty:
        # In between two non-empty chords, one can found for sure at least 
        # minimum_empty empty chords.
        minimum_empty = 10000000
              
        for chord in chords:            
            if chord.is_empty():
                empty_counter += 1
            elif empty_counter:  
                minimum_empty = min(empty_counter, minimum_empty)
                empty_counter = 0  
        
        # densified:
        # Densified version of chords, where minimum_empty empty chords have 
        # been removed between each pair of non-empty chords.
        densified = []
        empty_counter = 0        
        for chord in chords:
            if chord.is_empty():
                if empty_counter < minimum_empty:
                    empty_counter += 1
                else:
                    densified.append(chord)
            else:
                densified.append(chord)
                empty_counter = 0 
                            
        return densified
    