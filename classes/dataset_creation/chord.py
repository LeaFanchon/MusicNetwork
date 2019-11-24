#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:31:28 2019

@author: lea
"""
from classes.config import TRACK_NUMBER

class Chord:
        
    def __init__(self, satb=None, midi=None, model=None):
        """
        This class defines a chord. 
        satb: 
            List of TRACK_NUMBER integers between 0 and 12. A 0 means that
            no note is played by the track. The tracks are sorted by 
            decreasing mean pitch, meaning for example:
            Soprano - Alto - Tenor - Bass.
            Integer-note mapping:
            0:  Silence
            1:  C
            2:  C#
            3:  D
            4:  D#
            5:  E
            6:  F
            7:  F#
            8:  G
            9:  G#
            10: A
            11: A#
            12: B
        midi:
            Same as satb except for the fact that each note is an integer 
            between 0 and 128.
            This format is used to generate a midi file.
        model:
            Format of a chord when inputted to the neural network. 
            A single integer representing the chord. 
        
            As a chord consists of TRACK_NUMBER notes, each of which can take 
            13 values, the total possible number of chords is 13 to the power 
            of TRACK_NUMBER. 
            
            If TRACK_NUMBER=4, each chord can thus be mapped to an integer in
            the range [|0; 13⁴|[. This integer is chosen to be :
            soprano*1  + alto*13 + tenor*13² + bass*13³
        """
                
        self.satb = satb
        self.midi = midi
        self.model = model
        
        if satb is not None:            
            self.set_from_satb()
            return
            
        if midi is not None:            
            self.set_from_midi()
            return
            
        if model is not None:            
            self.set_from_model()
            return
        
        # Empty chord
        self.set_empty()
     
    def set_empty(self):
        """
        Set all formats to empty. 
        """
        self.satb = [0] * TRACK_NUMBER
        self.midi = [0] * TRACK_NUMBER
        self.model = 0  
        
    def set_from_satb(self):
        """
        Given the chord in satb format, compute the other formats. 
        """
        if min(self.satb) < 0 or max(self.satb) > 12:
            raise ValueError("Wrong range of notes for satb format")
            
        self.satb_to_midi()
        self.satb_to_model()
                   
    def set_from_midi(self):
        """
        Given the chord in midi format, compute the other formats. 
        """
        self.midi_to_satb()
        self.satb_to_model()
    
    def set_from_model(self):
        """
        Given the chord in model format, compute the other formats. 
        """
        self.model_to_satb()
        self.satb_to_midi()
       
    def satb_to_midi(self):
        """
        Convert satb format to midi format.
        """
        self.midi = [c for c in self.satb]
        
    def midi_to_satb(self):
        """
        Convert midi format to satb format.
        """
        self.satb =  [Chord.transpose_and_map(note) for note in self.midi]
        
    def satb_to_model(self):
        """
        Convert satb format to model format.
        """
        self.model = 0
        for i in range(TRACK_NUMBER):
            self.model += self.satb[i] * (13 ** i)
            
    def model_to_satb(self):
        """
        Convert model format to satb format.
        """
        self.satb = [0] * TRACK_NUMBER
        rest = self.model
        
        for i in reversed(range(TRACK_NUMBER)):            
            self.satb[i] = rest // (13 ** i)
            rest %= (13 ** i)
        
    def transpose(self, step):
        """
        Transpose a chord by adding a given interval to it (step). 
        Return the transposed chord object. 
        """                   
        transposed = [Chord.transpose_and_map(n, step) for n in self.satb]   
        return Chord(satb=transposed)
        
    def transpose_and_map(note, step=0):
        """
        Transpose 'note' an interval 'step' higher.
        Preserve the 'no note' event (when note == 0).
        Map the transposed note back into the interval [|0, 12|]
        """
        if note == 0:
            return note
        else:
            return ((note + step) % 12) + 1    
             
    def is_empty(self):
        """ 
        Returns True if all tracks are silent on this chord.
        """
        if self.satb == [0] * TRACK_NUMBER \
            and self.midi == [0] * TRACK_NUMBER \
            and self.model == 0:
            return True
        else:
            return False