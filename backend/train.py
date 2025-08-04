# import pickle
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, Activation
# from keras.utils import to_categorical
# from keras.callbacks import ModelCheckpoint
# import os

# def prepare_sequences(notes, n_vocab):
#     """ Prepare the sequences used by the Neural Network """
#     sequence_length = 100  # How many notes to look at before predicting the next one

#     # Get all unique pitch names
#     pitchnames = sorted(list(set(notes)))

#     # Create a dictionary to map pitches to integers
#     note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

#     network_input = []
#     network_output = []

#     # Create input sequences and the corresponding outputs
#     for i in range(0, len(notes) - sequence_length, 1):
#         sequence_in = notes[i:i + sequence_length]
#         sequence_out = notes[i + sequence_length]
#         network_input.append([note_to_int[char] for char in sequence_in])
#         network_output.append(note_to_int[sequence_out])

#     n_patterns = len(network_input)

#     # Reshape the input into a format compatible with LSTM layers
#     network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
#     # Normalize input between 0 and 1
#     network_input = network_input / float(n_vocab)

#     # One-hot encode the output
#     network_output = to_categorical(network_output, num_classes=n_vocab)

#     return (network_input, network_output)

# def create_network(network_in, n_vocab):
#     """ Create the structure of the neural network """
#     model = Sequential()
#     model.add(LSTM(
#         256, # Reduced from 512 for faster training
#         input_shape=(network_in.shape[1], network_in.shape[2]),
#         recurrent_dropout=0.2, # Reduced dropout
#         return_sequences=True
#     ))
#     model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.2))
#     model.add(LSTM(256))
#     model.add(Dense(128)) # Reduced from 256
#     model.add(Dropout(0.2))
#     model.add(Dense(n_vocab))
#     model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#     return model

# def train():
#     """ Train a Neural Network to generate music """
#     # Load the notes file
#     notes_path = os.path.join('model', 'notes')
#     with open(notes_path, 'rb') as filepath:
#         notes = pickle.load(filepath)

#     # --- CHANGE FOR FASTER TRAINING ---
#     # Use only a subset of the data to speed up training
#     notes = notes[:20000] # Using first 20,000 notes. Remove this line for full training.
    
#     # Get the number of unique pitches
#     n_vocab = len(set(notes))

#     # Prepare the sequences for the model
#     network_in, network_out = prepare_sequences(notes, n_vocab)

#     # Create the model
#     model = create_network(network_in, n_vocab)

#     print("Starting model training... (Fast Mode)")

#     # Define the checkpoint to save the best model weights
#     weights_path = os.path.join('model', 'weights-best-fast.keras') # Use a different file for the fast model
#     checkpoint = ModelCheckpoint(
#         weights_path,
#         monitor='loss',
#         verbose=0,
#         save_best_only=True,
#         mode='min'
#     )
#     callbacks_list = [checkpoint]

#     # Train the model
#     model.fit(
#         network_in, 
#         network_out, 
#         # --- CHANGE FOR FASTER TRAINING ---
#         epochs=20, # Reduced from 100
#         batch_size=128, # Increased batch size for potential speed up
#         callbacks=callbacks_list
#     )
    
#     print("Model training finished.")

# if __name__ == '__main__':
#     train()

import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
import argparse

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    pitchnames = sorted(list(set(notes)))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)

    return (network_input, network_output)

def create_network(network_in, n_vocab):
    """ Create the structure of the neural network """
    # --- CHANGE: Simplified the model for much faster training ---
    model = Sequential([
        LSTM(128, input_shape=(network_in.shape[1], network_in.shape[2]), recurrent_dropout=0.2, return_sequences=True),
        LSTM(128), # Removed one LSTM layer
        Dense(128),
        Dropout(0.2),
        Dense(n_vocab),
        Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def train(genre):
    """ Train a Neural Network for a specific genre """
    print(f"\n--- Training model for genre: {genre} (Fast Mode) ---")
    
    notes_path = os.path.join('model', f'notes_{genre}')
    try:
        with open(notes_path, 'rb') as filepath:
            notes = pickle.load(filepath)
    except FileNotFoundError:
        print(f"Error: Notes file not found at {notes_path}. Please run preprocess.py for this genre first.")
        return

    # --- CHANGE: Use a smaller subset of the data to drastically speed up training ---
    print(f"Original number of notes: {len(notes)}")
    notes = notes[:20000] 
    print(f"Using a subset of {len(notes)} notes for faster training.")
    
    n_vocab = len(set(notes))
    network_in, network_out = prepare_sequences(notes, n_vocab)
    model = create_network(network_in, n_vocab)

    print(f"Starting model training for '{genre}'...")

    weights_filename = f'weights-best-{genre}.keras'
    weights_path = os.path.join('model', weights_filename)
    
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(
        network_in, 
        network_out, 
        # --- CHANGE: Reduced epochs and increased batch size for speed ---
        epochs=15, 
        batch_size=256,
        callbacks=callbacks_list
    )
    
    print(f"--- Model training for '{genre}' finished. Weights saved to {weights_path} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model for a specific music genre.")
    parser.add_argument("genre", type=str, help="The name of the genre to train (e.g., 'jazz', 'rock').")
    
    args = parser.parse_args()
    
    train(args.genre)
