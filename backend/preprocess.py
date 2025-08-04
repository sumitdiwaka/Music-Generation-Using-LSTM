# import glob
# import pickle
# import numpy
# from music21 import converter, instrument, note, chord
# import os

# def get_notes_from_midi():
#     """
#     Parses all MIDI files in the data/midi_files directory, extracts the notes and chords,
#     and saves them to a file.
#     """
#     notes = []
    
#     # Define the path to the MIDI files
#     midi_path = os.path.join('data', 'midi_files', '**', '*.mid')
    
#     # Use glob to find all MIDI files recursively
#     for file in glob.glob(midi_path, recursive=True):
#         try:
#             print(f"Parsing {file}...")
#             midi = converter.parse(file)
            
#             notes_to_parse = None
            
#             # A MIDI file can have multiple parts (instruments)
#             # We'll try to get the notes from the first instrument part
#             parts = instrument.partitionByInstrument(midi)
            
#             if parts:  # file has instrument parts
#                 notes_to_parse = parts.parts[0].recurse()
#             else:  # file has notes in a flat structure
#                 notes_to_parse = midi.flat.notes
            
#             for element in notes_to_parse:
#                 if isinstance(element, note.Note):
#                     # If it's a note, store its pitch
#                     notes.append(str(element.pitch))
#                 elif isinstance(element, chord.Chord):
#                     # If it's a chord, store the normal order of the notes in the chord
#                     # (e.g., [C4, E4, G4] becomes 'C4.E4.G4')
#                     notes.append('.'.join(str(n) for n in element.normalOrder))
#         except Exception as e:
#             print(f"Could not parse {file}. Reason: {e}")

#     # Create the model directory if it doesn't exist
#     model_dir = 'model'
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     # Save the extracted notes to a file
#     notes_path = os.path.join(model_dir, 'notes')
#     with open(notes_path, 'wb') as filepath:
#         pickle.dump(notes, filepath)
        
#     print(f"\nSuccessfully parsed all files. Notes saved to {notes_path}")
#     return notes

# if __name__ == '__main__':
#     get_notes_from_midi()


import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
import os
import argparse # Import the argparse library to handle command-line arguments

def get_notes_from_midi(genre):
    """
    Parses MIDI files for a specific genre, extracts the notes and chords,
    and saves them to a genre-specific file.
    """
    notes = []
    
    # --- CHANGE: The path now points to the specific genre folder ---
    print(f"\n--- Processing genre: {genre} ---")
    midi_path = os.path.join('data', 'midi_files', genre, '*.mid')
    
    # Use glob to find all MIDI files in the specified genre folder
    files = glob.glob(midi_path)
    if not files:
        print(f"Error: No MIDI files found for genre '{genre}'. Check the folder name and that it contains .mid files.")
        return

    for file in files:
        try:
            print(f"Parsing {file}...")
            midi = converter.parse(file)
            
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Could not parse {file}. Reason: {e}")

    # Create the model directory if it doesn't exist
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- CHANGE: Save the notes to a file named after the genre ---
    notes_filename = f'notes_{genre}'
    notes_path = os.path.join(model_dir, notes_filename)
    with open(notes_path, 'wb') as filepath:
        pickle.dump(notes, filepath)
        
    print(f"\nSuccessfully parsed {len(files)} files for genre '{genre}'. Notes saved to {notes_path}")
    return notes

if __name__ == '__main__':
    # --- CHANGE: Set up the script to accept a genre argument from the command line ---
    parser = argparse.ArgumentParser(description="Preprocess MIDI files for a specific genre.")
    parser.add_argument("genre", type=str, help="The name of the genre folder to process (e.g., 'jazz', 'rock').")
    
    args = parser.parse_args()
    
    get_notes_from_midi(args.genre)

