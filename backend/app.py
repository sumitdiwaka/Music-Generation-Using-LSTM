# from flask import Flask, jsonify
# from flask_cors import CORS
# import pickle
# import numpy as np
# from music21 import instrument, note, chord, stream
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, Activation
# import os
# import base64

# # --- Initialize the Flask App ---
# # --- FIX: Point Flask to the frontend folder to serve the index.html file ---
# app = Flask(__name__, static_folder='../frontend', static_url_path='/')
# CORS(app) # Allow requests from the front-end

# # --- Global variables to hold the model and data ---
# MODEL = None
# NOTES = None
# PITCHNAMES = None
# N_VOCAB = 0

# def create_network(n_vocab):
#     """ Re-create the structure of the neural network """
#     # This structure MUST be identical to the one in train.py
#     model = Sequential()
#     model.add(LSTM(
#         256,
#         input_shape=(100, 1), # (sequence_length, n_features)
#         recurrent_dropout=0.2,
#         return_sequences=True
#     ))
#     model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.2))
#     model.add(LSTM(256))
#     model.add(Dense(128))
#     model.add(Dropout(0.2))
#     model.add(Dense(n_vocab))
#     model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#     return model

# def load_music_generation_model():
#     """Load the trained model and notes data."""
#     global MODEL, NOTES, PITCHNAMES, N_VOCAB

#     print("--- Loading model and notes data ---")
    
#     # Load the notes used to train the model
#     notes_path = os.path.join('model', 'notes')
#     with open(notes_path, 'rb') as filepath:
#         NOTES = pickle.load(filepath)

#     # Use the same slice of data as in the training script
#     NOTES = NOTES[:20000]

#     PITCHNAMES = sorted(list(set(NOTES)))
#     N_VOCAB = len(set(NOTES))
    
#     # Re-create the model structure
#     MODEL = create_network(N_VOCAB)
    
#     # Load the trained weights
#     weights_path = os.path.join('model', 'weights-best-fast.keras')
#     MODEL.load_weights(weights_path)
    
#     print("--- Model and data loaded successfully ---")


# def generate_notes(model, network_input, pitchnames, n_vocab):
#     """ Generate notes from the neural network based on a sequence of notes """
#     start = np.random.randint(0, len(network_input)-1)
#     int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
#     pattern = network_input[start]
#     prediction_output = []

#     # generate 200 notes (reduced for faster API response)
#     for note_index in range(200):
#         prediction_input = np.reshape(pattern, (1, len(pattern), 1))
#         prediction_input = prediction_input / float(n_vocab)
#         prediction = model.predict(prediction_input, verbose=0)
#         index = np.argmax(prediction)
#         result = int_to_note[index]
#         prediction_output.append(result)
#         pattern.append(index)
#         pattern = pattern[1:len(pattern)]

#     return prediction_output

# def create_midi_from_notes(prediction_output, filename="output.mid"):
#     """ convert the output from the prediction to notes and create a midi file """
#     offset = 0
#     output_notes = []

#     for pattern in prediction_output:
#         if ('.' in pattern) or pattern.isdigit():
#             notes_in_chord = pattern.split('.')
#             notes = []
#             for current_note in notes_in_chord:
#                 new_note = note.Note(int(current_note))
#                 new_note.instrument = instrument.Piano()
#                 notes.append(new_note)
#             new_chord = chord.Chord(notes)
#             new_chord.offset = offset
#             output_notes.append(new_chord)
#         else:
#             new_note = note.Note(pattern)
#             new_note.offset = offset
#             new_note.instrument = instrument.Piano()
#             output_notes.append(new_note)
#         offset += 0.5

#     midi_stream = stream.Stream(output_notes)
#     output_dir = 'generated_music'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     output_path = os.path.join(output_dir, filename)
#     midi_stream.write('midi', fp=output_path)
#     return output_path

# # --- FIX: Add a route to serve the main index.html file ---
# @app.route('/')
# def serve_index():
#     return app.send_static_file('index.html')

# # --- API Endpoint ---
# @app.route('/generate', methods=['GET'])
# def generate_music_endpoint():
#     """ The API endpoint that the front-end will call """
#     global MODEL, NOTES, PITCHNAMES, N_VOCAB

#     if MODEL is None:
#         return jsonify({'error': 'Model is not loaded yet!'}), 500

#     # Prepare sequences for generation
#     sequence_length = 100
#     note_to_int = dict((note, number) for number, note in enumerate(PITCHNAMES))
#     network_input = []
#     for i in range(0, len(NOTES) - sequence_length, 1):
#         sequence_in = NOTES[i:i + sequence_length]
#         network_input.append([note_to_int[char] for char in sequence_in])

#     # Generate notes
#     prediction_output = generate_notes(MODEL, network_input, PITCHNAMES, N_VOCAB)
    
#     # Create a MIDI file
#     midi_path = create_midi_from_notes(prediction_output)

#     # Read the MIDI file and encode it in Base64
#     with open(midi_path, 'rb') as midi_file:
#         midi_base64 = base64.b64encode(midi_file.read()).decode('utf-8')

#     # Return the Base64 encoded MIDI file as a JSON response
#     return jsonify({'midi': midi_base64})

# if __name__ == '__main__':
#     # Load the model when the server starts
#     load_music_generation_model()
#     # Run the Flask app
#     app.run(debug=True, port=5000)


# import os
# import base64
# import pickle
# import numpy as np
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# from music21 import converter, instrument, note, chord, stream
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, Activation
# import tempfile

# # --- 1. APP SETUP ---
# app = Flask(__name__, static_folder='../frontend', static_url_path='/')
# CORS(app)

# # --- 2. GLOBAL VARIABLES FOR AI MODELS ---
# GENERAL_MODEL = None
# GENERAL_NOTES = None
# GENERAL_PITCHNAMES = None
# N_VOCAB_GENERAL = 0

# GENRE_MODELS = {}
# GENRE_NOTES = {}
# GENRE_PITCHNAMES = {}
# N_VOCAB_GENRE = {}
# SUPPORTED_GENRES = ['jazz', 'rock', 'classical']

# # --- 3. AI MODEL LOADING LOGIC ---

# def create_network(n_vocab, simplified=False):
#     """ Creates the network architecture. """
#     if simplified:
#         model = Sequential([
#             LSTM(128, input_shape=(100, 1), recurrent_dropout=0.2, return_sequences=True),
#             LSTM(128),
#             Dense(128),
#             Dropout(0.2),
#             Dense(n_vocab),
#             Activation('softmax')
#         ])
#     else:
#         model = Sequential([
#             LSTM(256, input_shape=(100, 1), recurrent_dropout=0.2, return_sequences=True),
#             LSTM(256, return_sequences=True, recurrent_dropout=0.2),
#             LSTM(256),
#             Dense(128),
#             Dropout(0.2),
#             Dense(n_vocab),
#             Activation('softmax')
#         ])
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#     return model

# def load_all_models():
#     """ Loads the main model and all genre-specific models at startup. """
#     global GENERAL_MODEL, GENERAL_NOTES, GENERAL_PITCHNAMES, N_VOCAB_GENERAL
    
#     print("--- Loading General AI model ---")
#     with open(os.path.join('model', 'notes'), 'rb') as f: GENERAL_NOTES = pickle.load(f)[:20000]
#     GENERAL_PITCHNAMES = sorted(list(set(GENERAL_NOTES)))
#     N_VOCAB_GENERAL = len(GENERAL_PITCHNAMES)
#     GENERAL_MODEL = create_network(N_VOCAB_GENERAL, simplified=False)
#     GENERAL_MODEL.load_weights(os.path.join('model', 'weights-best-fast.keras'))
#     print("--- General AI Model loaded successfully ---")

#     for genre in SUPPORTED_GENRES:
#         print(f"--- Loading {genre.capitalize()} AI model ---")
#         try:
#             notes_path = os.path.join('model', f'notes_{genre}')
#             weights_path = os.path.join('model', f'weights-best-{genre}.keras')
            
#             with open(notes_path, 'rb') as f: notes = pickle.load(f)[:20000]
            
#             GENRE_NOTES[genre] = notes
#             pitchnames = sorted(list(set(notes)))
#             GENRE_PITCHNAMES[genre] = pitchnames
#             n_vocab = len(pitchnames)
#             N_VOCAB_GENRE[genre] = n_vocab
            
#             model = create_network(n_vocab, simplified=True)
#             model.load_weights(weights_path)
#             GENRE_MODELS[genre] = model
#             print(f"--- {genre.capitalize()} AI Model loaded successfully ---")
#         except Exception as e:
#             print(f"!!! Could not load model for genre '{genre}'. Error: {e} !!!")

# # --- 4. MUSIC GENERATION & CONVERSION LOGIC ---

# # --- FIX: Refactored generate_notes to take a starting pattern directly ---
# def generate_notes_from_pattern(model, initial_pattern, pitchnames, n_vocab):
#     """ Generates a sequence of notes starting from a given pattern. """
#     int_to_note = {num: note for num, note in enumerate(pitchnames)}
#     pattern = initial_pattern
#     prediction_output = []
#     for _ in range(200):
#         prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
#         prediction = model.predict(prediction_input, verbose=0)
#         index = np.argmax(prediction)
#         result = int_to_note[index]
#         prediction_output.append(result)
#         pattern.append(index)
#         pattern = pattern[1:]
#     return prediction_output

# def create_midi_from_notes(prediction_output):
#     """ Converts a list of notes into a Base64 MIDI string. """
#     offset = 0
#     output_notes = []
#     for pattern in prediction_output:
#         if ('.' in pattern) or pattern.isdigit():
#             notes_in_chord = pattern.split('.')
#             notes = [note.Note(int(n)) for n in notes_in_chord]
#             for n in notes: n.instrument = instrument.Piano()
#             new_chord = chord.Chord(notes)
#             new_chord.offset = offset
#             output_notes.append(new_chord)
#         else:
#             new_note = note.Note(pattern)
#             new_note.offset = offset
#             new_note.instrument = instrument.Piano()
#             output_notes.append(new_note)
#         offset += 0.5
    
#     midi_stream = stream.Stream(output_notes)
    
#     with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi_file:
#         midi_stream.write('midi', fp=temp_midi_file.name)
#         temp_midi_file.seek(0)
#         midi_base64 = base64.b64encode(temp_midi_file.read()).decode('utf-8')
    
#     os.remove(temp_midi_file.name)
#     return midi_base64

# # --- 5. PAGE & API ROUTES ---

# @app.route('/')
# def serve_index():
#     return app.send_static_file('index.html')

# @app.route('/generate', methods=['GET'])
# def generate_music_endpoint():
#     """ API for the 'Generate Music' button. """
#     if GENERAL_MODEL is None: return jsonify({'error': 'Model not loaded!'}), 500
    
#     note_to_int = {note: num for num, note in enumerate(GENERAL_PITCHNAMES)}
#     network_input = [[note_to_int[char] for char in GENERAL_NOTES[i:i + 100]] for i in range(len(GENERAL_NOTES) - 100)]
    
#     # FIX: Pick a random starting pattern here
#     start = np.random.randint(0, len(network_input) - 1)
#     initial_pattern = network_input[start]
    
#     prediction_output = generate_notes_from_pattern(GENERAL_MODEL, initial_pattern, GENERAL_PITCHNAMES, N_VOCAB_GENERAL)
#     midi_base64 = create_midi_from_notes(prediction_output)
#     return jsonify({'midi': midi_base64})

# @app.route('/api/convert', methods=['POST'])
# def convert_style_endpoint():
#     """ API for the new 'Convert Style' button. """
#     data = request.get_json()
#     target_genre = data.get('genre')
#     midi_base64_in = data.get('midi_data')

#     if not target_genre or target_genre not in SUPPORTED_GENRES:
#         return jsonify({'error': 'Invalid or unsupported genre.'}), 400
#     if not midi_base64_in:
#         return jsonify({'error': 'No MIDI data provided.'}), 400

#     model = GENRE_MODELS[target_genre]
#     pitchnames = GENRE_PITCHNAMES[target_genre]
#     n_vocab = N_VOCAB_GENRE[target_genre]
    
#     try:
#         midi_data = base64.b64decode(midi_base64_in)
#         with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi_file:
#             temp_midi_file.write(midi_data)
#             temp_midi_file.flush()
            
#             # Use .flatten() which is the modern equivalent of .flat
#             midi_stream = converter.parse(temp_midi_file.name).flatten()
            
#             seed_notes = []
#             for element in midi_stream.notes:
#                 if isinstance(element, note.Note):
#                     seed_notes.append(str(element.pitch))
#                 elif isinstance(element, chord.Chord):
#                     seed_notes.append('.'.join(str(n) for n in element.normalOrder))
#                 if len(seed_notes) >= 100:
#                     break
#         os.remove(temp_midi_file.name)

#         if len(seed_notes) < 100:
#             return jsonify({'error': 'Uploaded MIDI is too short (must contain at least 100 notes/chords).'}), 400

#     except Exception as e:
#         return jsonify({'error': f'Could not parse uploaded MIDI file. Reason: {e}'}), 500

#     note_to_int = {note: num for num, note in enumerate(pitchnames)}
#     network_input_seed = []
#     for n in seed_notes:
#         network_input_seed.append(note_to_int.get(n, np.random.randint(0, n_vocab - 1)))
    
#     # FIX: Pass the single seed pattern directly to the generation function
#     prediction_output = generate_notes_from_pattern(model, network_input_seed, pitchnames, n_vocab)
#     midi_base64_out = create_midi_from_notes(prediction_output)
    
#     return jsonify({'midi': midi_base64_out})

# # --- 6. RUN THE APP ---
# if __name__ == '__main__':
#     load_all_models()
#     app.run(debug=True, port=5000)
import os
import base64
import pickle
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
import tempfile
from datetime import datetime

# --- 1. APP & DATABASE SETUP ---
app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = 'a-super-secret-key-that-you-should-change'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'serve_login'

# --- 2. DATABASE MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    password = db.Column(db.String(200), nullable=False)
    profile_photo = db.Column(db.Text, nullable=True)
    songs = db.relationship('Song', backref='author', lazy=True, cascade="all, delete-orphan")

class Song(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    date_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    midi_data = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- 3. AUTHENTICATION API ENDPOINTS ---
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Missing required fields'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'Email address already exists'}), 409
    
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    new_user = User(
        fullname=data['fullname'], 
        email=data['email'], 
        phone=data.get('phone'), 
        password=hashed_password,
        profile_photo=data.get('profile_photo')
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'Account created successfully! Please log in.'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data['email']).first()
    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({'message': 'Invalid email or password.'}), 401
    login_user(user, remember=data.get('remember', False))
    return jsonify({'message': 'Login successful!'}), 200

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logout successful'}), 200

# --- 4. PROTECTED DATA & PROFILE API ENDPOINTS ---
@app.route('/api/user', methods=['GET'])
@login_required
def get_user_data():
    return jsonify({
        'fullname': current_user.fullname,
        'email': current_user.email,
        'phone': current_user.phone,
        'profile_photo': current_user.profile_photo
    })

@app.route('/api/user/update', methods=['POST'])
@login_required
def update_user_profile():
    data = request.get_json()
    user = User.query.get(current_user.id)
    
    if 'fullname' in data: user.fullname = data['fullname']
    if 'phone' in data: user.phone = data['phone']
    if 'profile_photo' in data: user.profile_photo = data['profile_photo']
    
    if 'new_password' in data and data['new_password']:
        if check_password_hash(user.password, data.get('current_password', '')):
            user.password = generate_password_hash(data['new_password'], method='pbkdf2:sha256')
        else:
            return jsonify({'message': 'Current password is incorrect.'}), 400
            
    db.session.commit()
    return jsonify({'message': 'Profile updated successfully.'}), 200

@app.route('/api/songs', methods=['GET'])
@login_required
def get_user_songs():
    songs = Song.query.filter_by(user_id=current_user.id).order_by(Song.date_created.desc()).all()
    song_list = [{'id': s.id, 'title': s.title, 'date_created': s.date_created.strftime('%b %d, %Y'), 'midi_data': s.midi_data} for s in songs]
    return jsonify(song_list)

@app.route('/api/songs/delete/<int:song_id>', methods=['DELETE'])
@login_required
def delete_song(song_id):
    song = Song.query.get(song_id)
    if song and song.user_id == current_user.id:
        db.session.delete(song)
        db.session.commit()
        return jsonify({'message': 'Song deleted successfully.'}), 200
    return jsonify({'message': 'Song not found or you do not have permission to delete it.'}), 404

# --- 5. PAGE SERVING ROUTES ---
@app.route('/')
def serve_login(): return app.send_static_file('login.html')

@app.route('/signup')
def serve_signup(): return app.send_static_file('signup.html')

@app.route('/dashboard')
@login_required
def serve_dashboard(): return app.send_static_file('user_dashboard.html')

@app.route('/generator')
@login_required
def serve_generator(): return app.send_static_file('index.html')

# --- 6. AI MUSIC GENERATION ---
GENERAL_MODEL = None
GENERAL_NOTES = None
GENERAL_PITCHNAMES = None
N_VOCAB_GENERAL = 0

GENRE_MODELS = {}
GENRE_NOTES = {}
GENRE_PITCHNAMES = {}
N_VOCAB_GENRE = {}
SUPPORTED_GENRES = ['jazz', 'rock', 'classical']

def create_network(n_vocab, simplified=False):
    """ Creates the network architecture. """
    if simplified:
        model = Sequential([
            LSTM(128, input_shape=(100, 1), recurrent_dropout=0.2, return_sequences=True),
            LSTM(128),
            Dense(128),
            Dropout(0.2),
            Dense(n_vocab),
            Activation('softmax')
        ])
    else:
        model = Sequential([
            LSTM(256, input_shape=(100, 1), recurrent_dropout=0.2, return_sequences=True),
            LSTM(256, return_sequences=True, recurrent_dropout=0.2),
            LSTM(256),
            Dense(128),
            Dropout(0.2),
            Dense(n_vocab),
            Activation('softmax')
        ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def load_all_models():
    """ Loads the main model and all genre-specific models at startup. """
    global GENERAL_MODEL, GENERAL_NOTES, GENERAL_PITCHNAMES, N_VOCAB_GENERAL
    
    print("--- Loading General AI model ---")
    with open(os.path.join('model', 'notes'), 'rb') as f: GENERAL_NOTES = pickle.load(f)[:20000]
    GENERAL_PITCHNAMES = sorted(list(set(GENERAL_NOTES)))
    N_VOCAB_GENERAL = len(GENERAL_PITCHNAMES)
    GENERAL_MODEL = create_network(N_VOCAB_GENERAL, simplified=False)
    GENERAL_MODEL.load_weights(os.path.join('model', 'weights-best-fast.keras'))
    print("--- General AI Model loaded successfully ---")

    for genre in SUPPORTED_GENRES:
        print(f"--- Loading {genre.capitalize()} AI model ---")
        try:
            notes_path = os.path.join('model', f'notes_{genre}')
            weights_path = os.path.join('model', f'weights-best-{genre}.keras')
            
            with open(notes_path, 'rb') as f: notes = pickle.load(f)[:20000]
            
            GENRE_NOTES[genre] = notes
            pitchnames = sorted(list(set(notes)))
            GENRE_PITCHNAMES[genre] = pitchnames
            n_vocab = len(pitchnames)
            N_VOCAB_GENRE[genre] = n_vocab
            
            model = create_network(n_vocab, simplified=True)
            model.load_weights(weights_path)
            GENRE_MODELS[genre] = model
            print(f"--- {genre.capitalize()} AI Model loaded successfully ---")
        except Exception as e:
            print(f"!!! Could not load model for genre '{genre}'. Error: {e} !!!")

def generate_notes_from_pattern(model, initial_pattern, pitchnames, n_vocab):
    """ Generates a sequence of notes starting from a given pattern. """
    int_to_note = {num: note for num, note in enumerate(pitchnames)}
    pattern = initial_pattern
    prediction_output = []
    for _ in range(200):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]
    return prediction_output

def create_midi_from_notes(prediction_output):
    """ Converts a list of notes into a Base64 MIDI string. """
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in notes: n.instrument = instrument.Piano()
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.instrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    
    midi_stream = stream.Stream(output_notes)
    
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi_file:
        midi_stream.write('midi', fp=temp_midi_file.name)
        temp_midi_file.seek(0)
        midi_base64 = base64.b64encode(temp_midi_file.read()).decode('utf-8')
    
    os.remove(temp_midi_file.name)
    return midi_base64

# --- 7. MUSIC GENERATION API ENDPOINTS ---
@app.route('/generate', methods=['POST'])
@login_required
def generate_music_endpoint():
    """ API for the 'Generate Music' button. """
    if GENERAL_MODEL is None: 
        return jsonify({'error': 'Model not loaded!'}), 500
    
    note_to_int = {note: num for num, note in enumerate(GENERAL_PITCHNAMES)}
    network_input = [[note_to_int[char] for char in GENERAL_NOTES[i:i + 100]] 
                    for i in range(len(GENERAL_NOTES) - 100)]
    
    start = np.random.randint(0, len(network_input) - 1)
    initial_pattern = network_input[start]
    
    prediction_output = generate_notes_from_pattern(GENERAL_MODEL, initial_pattern, 
                                                  GENERAL_PITCHNAMES, N_VOCAB_GENERAL)
    midi_base64 = create_midi_from_notes(prediction_output)
    
    song_title = f"Melody-{current_user.id}-{datetime.now().strftime('%f')}"
    new_song = Song(title=song_title, midi_data=midi_base64, author=current_user)
    db.session.add(new_song)
    db.session.commit()
    
    return jsonify({
        'midi': midi_base64, 
        'title': song_title
    })

@app.route('/api/convert', methods=['POST'])
@login_required
def convert_style_endpoint():
    """ API for the 'Convert Style' button. """
    data = request.get_json()
    target_genre = data.get('genre')
    midi_base64_in = data.get('midi_data')

    if not target_genre or target_genre not in SUPPORTED_GENRES:
        return jsonify({'error': 'Invalid or unsupported genre.'}), 400
    if not midi_base64_in:
        return jsonify({'error': 'No MIDI data provided.'}), 400

    model = GENRE_MODELS.get(target_genre)
    pitchnames = GENRE_PITCHNAMES.get(target_genre)
    n_vocab = N_VOCAB_GENRE.get(target_genre)
    
    if not all([model, pitchnames, n_vocab]):
        return jsonify({'error': f'Model for {target_genre} not loaded.'}), 500

    try:
        midi_data = base64.b64decode(midi_base64_in)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi_file:
            temp_midi_file.write(midi_data)
            temp_midi_file.flush()
            
            midi_stream = converter.parse(temp_midi_file.name).flatten()
            
            seed_notes = []
            for element in midi_stream.notes:
                if isinstance(element, note.Note):
                    seed_notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    seed_notes.append('.'.join(str(n) for n in element.normalOrder))
                if len(seed_notes) >= 100:
                    break
        os.remove(temp_midi_file.name)

        if len(seed_notes) < 100:
            return jsonify({'error': 'Uploaded MIDI is too short (must contain at least 100 notes/chords).'}), 400

    except Exception as e:
        return jsonify({'error': f'Could not parse uploaded MIDI file. Reason: {e}'}), 500

    note_to_int = {note: num for num, note in enumerate(pitchnames)}
    network_input_seed = []
    for n in seed_notes:
        network_input_seed.append(note_to_int.get(n, np.random.randint(0, n_vocab - 1)))
    
    prediction_output = generate_notes_from_pattern(model, network_input_seed, pitchnames, n_vocab)
    midi_base64_out = create_midi_from_notes(prediction_output)
    
    song_title = f"Converted-{target_genre}-{datetime.now().strftime('%f')}"
    new_song = Song(title=song_title, midi_data=midi_base64_out, author=current_user)
    db.session.add(new_song)
    db.session.commit()
    
    return jsonify({'midi': midi_base64_out, 'title': song_title})

# --- 8. INITIALIZATION & RUN ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    load_all_models()
    app.run(debug=True, port=5000)

# if __name__ == '__main__':
#     with app.app_context():
#         db.drop_all()
#         db.create_all()
#     app.run(debug=True)
