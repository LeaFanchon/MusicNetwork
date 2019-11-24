# Music Network

The goal of this personal project is to build a recurrent neural network to create four-voice polyphonic music. 

This project can be broken down into four steps:

1. Find data and generate a dataset
2. Preprocess the dataset so it is ready to be fed to a neural network
3. Design and train a model on this dataset
4. Use the trained model to create music

## I. Dataset creation

### A. Find data
Raw data to train the network was found [here](http://www.kunstderfuge.com).
For 40 euros, I could download around 18000 midi files in one zip. 
Midi format seemed to me as a very good option to extract scores in a readable way.

### B. Process raw data
To create a dataset, I browsed through the midi files. For each visited file, five operations were performed : meaningful tracks extraction, conversion to absolute time with one chord per tick, densification, reduction and transposition to all 12 possible tonalities. 

#### Meaningful tracks extraction
The goal being to create four-voice polyphonic music, the observations must consist of n chords of 4 notes, the n-1 first being the predictors, and the n<sup>th</sup> the target to be predicted. For each visited midi file, the four most significant tracks must thus be selected. 
To evaluate the meaningfulness of a track, two elements are taken into account: the number of notes it plays throughout the midi file, and the standard deviation of these notes. 

#### Conversion to absolute time with one chord per tick
The midi format is initially encoded in relative time, meaning that the moment in which each event is supposed to happen is deduced by the time elapsed between this event and the previous one. As each observation must consist in one chord for each tick of time, the tracks must first be converted to absolute time. After conversion, the moment in which each event is supposed to happen is deduced by the elapsed time since the beginning of the piece instead. Each track is then a list of tuples containing a note and the moment it is played expressed in number of ticks elapsed since the piece started. From there, a simple algorithm converts the midi file into the desired format: a list of four-notes chords, one for each tick of time. 

#### Densification
The *one chord per tick* format still needs some transformations. As it is, it might be a very sparse data structure. This is why it needs to be densified. For instance, if we note C a chord with at least one track playing a note, and E an empty chord where all the tracks are silent, we need to get from the following state (*) to a state where there is at least one consecutive pair of non-silent chords (**). 

[C, E, E, E, C, E, E, E, E, C, E, E, C, E, E, E, C] (*)

[C, E, C, E, E, **C, C,** E, C] (**)

This densification process is applied recursively to each midi file, through the following algorithm, presented here in a simplified form:

'''

    # Returns a list of candidates. 
    # One candidate is a dense enough list of chords. 
    def select_observation_candidates(chords):

        # First densify the list of chords
        chords = densify(chords)
        
        # If the required density is not reached, apply the function recursively to each half of the list. 
        if density(chords) < density_treshold:
            half = int(len(chords)/2)
            first_half = chords[:half]
            second_half = chords[half:]
            return first_half + second_half
        else:
            return [chords]    
'''

Each midi file can thus contain any number of distinct parts considered good candidates to create observations. 

#### Reduction
In midi format, the pitch of a note is encoded by an integer ranging from 0 to 127. To this, we must also add the silent event, which takes us to 129 possibilities. As we handle 4-notes chords, if we keep this encoding, our vocabulary size is 129<sup>4</sup> = 276 922 881. Thinking that 276 millions is a bit high for a vocabulary size, I chose to lose the information of the octave of the handled notes. With this thinking, each note can now have only 13 different values: the 12 notes and the silent event. The vocabulary size is now 13<sup>4</sup> = 28561, which is more acceptable. 

#### Transposition
Once a chord list of length n is selected through the densification process and reduced, it is transposed to the 12 possible tonalities. I briefly thought of transposing all the pieces to C, but after giving it some thought I decided otherwise. If all the observations are in C, when the neural network creates a modulation to another tone, for example G major, it won't be able to properly generate the rest of the piece once the modulation is done. 

For each of the 12 tonalities, n observations are extracted, each chord of the list being the target of one of them. 

For instance, if the chord list is [C, E, C, E, E, C, C, E, C], of length 9: 

1. The list is first turned into a set of 12 lists {[C<sup>i</sup>, E, C<sup>i</sup>, E, E, C<sup>i</sup>, C<sup>i</sup>, E, C<sup>i</sup>] / i in [|0, 11|]}. 
2. For each of those 12 lists, 9 observations are extracted: <br>


| Predictors                                                                             |    Target            |
|----------------------------------------------------------------------------------------|----------------------|
| []                                                                                     |    C<sup>i</sup>     |
| [C<sup>i</sup>]                                                                        |    E                 |
| [C<sup>i</sup>, E]                                                                     |    C<sup>i</sup>     |
| [C<sup>i</sup>, E, C<sup>i</sup>]                                                      |    E                 |
| [C<sup>i</sup>, E, C<sup>i</sup>, E]                                                   |    E                 |
| [C<sup>i</sup>, E, C<sup>i</sup>, E, E]                                                |    C<sup>i</sup>     |
| [C<sup>i</sup>, E, C<sup>i</sup>, E, E, C<sup>i</sup>]                                 |    C<sup>i</sup>     |
| [C<sup>i</sup>, E, C<sup>i</sup>, E, E, C<sup>i</sup>, C<sup>i</sup>]                  |    E                 |
| [C<sup>i</sup>, E, C<sup>i</sup>, E, E, C<sup>i</sup>, C<sup>i</sup>, E]               |    C<sup>i</sup>     |


### C. Dataset
In the end, I obtained a pickle file of 1,1G, containing approximatively 18 million observations. 

## II. Preprocess the dataset

I stored the newly created dataset in a sqlite database, in a table containing three columns: 

- id (integer) : identifies the observation
- observation (blob) : pickled observation 
- batch number (integer) : identifies the batch to which the observation belongs.

The observations were shuffled before being assigned to a batch. 

I chose to distribute the data among batches for two reasons. The first reason is that the length of the observations' sequences is very variable. Some of them might have 0 predictors, and some several thousands. I wanted to avoid padding all the observations with a very long sequence of empty chords. Training the network one batch at a time makes it unnecessary. The second reason is that I wanted to avoid loading all the data in RAM at the same time. 

During training, the batches will be fed to the model by a DataGenerator inheriting from the keras Sequence object.

## III. Design and train the model.

### A. Model design

The first layer of the neural network is an embedding layer. It takes as input sequences of chords, each chord being encoded as an integer ranging from 0 to 13<sup>4</sup>-1, the vocabulary size. The layer maps these chords to their projection into a 13*4 = 52 dimensional space. 

The embedded chords then go through three bidirectional LSTM layers, each of 64 units.

The last layer is a dense layer mapping the output of the third LSTM layer to a softmax classifier. The final output of the RNN is one of the 13<sup>4</sup> vocabulary entries.

In total, the model has 5 428 613 trainable parameters, almost 6 million. This is a lot, but as I used CuDNNLSTM layers to optimize GPU training, I could not add recurrent dropout since it has not been implemented for them yet. So some overfitting might happen. 

Here is the summary of the model:<br>

| Layer (type)                     | Output Shape        |     Param      |
|----------------------------------|---------------------|----------------|
| embedding_1 (Embedding)          | (None, 100, 52)     |     1485172    |
| bidirectional_1 (Bidirectional)  | (None, 100, 128)    |     60416      | 
| bidirectional_2 (Bidirectional)  | (None, 100, 128)    |     99328      |
| bidirectional_3 (Bidirectional)  | (None, 128)         |     99328      |
| dense_1 (Dense)                  | (None, 28561)       |     3684369    |

### B. Model training

The model is trained on my personal computer. I use tensorflow-gpu running on cuda backend, and tried to get my 3 Nvidia GPUs involved in the training (*GeForce RTX 2060 SUPER*). The GPUs seemed to be unequally used during training:

- One of the 2060 is used at 60% on average
- The two other ones at 20%.





