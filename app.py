import os
import requests
import zipfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import traceback
from datetime import datetime # Import datetime for timestamps

app = Flask(__name__)
CORS(app)

# --- Helper Functions for Google Drive Download and Extraction ---
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    print(f"[{datetime.now()}] Starting download from Google Drive file id: {file_id} ...")

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    # Check for and handle Google Drive download warning
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        # If a warning token is found, confirm the download
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Save the file content
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"[{datetime.now()}] Download completed: {destination}")

def extract_zip(zip_path, extract_to):
    print(f"[{datetime.now()}] Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[{datetime.now()}] Extraction completed.")

# --- Configuration and Model Loading ---
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure models directory exists

# CORRECTED GOOGLE DRIVE FILE IDs
YAMNET_MODEL_FILE_ID = '18FX4FDwmSUHZhbuQ_-GBDgtBwwmzyFiL'  # Corrected ID
CLASS_MAP_FILE_ID = '1dBjlS4aVXdKqSYhelLb6_W9OKMUFnAVY'    # Original correct ID

YAMNET_MODEL_ZIP = os.path.join(BASE_DIR, 'yamnet_model.zip')
CLASS_MAP_ZIP = os.path.join(BASE_DIR, 'yamnet_class_map.zip')

print(f"[{datetime.now()}] Application startup: Starting model and class map checks.")

# Download and extract YAMNet model if not already present
yamnet_model_extracted_path = os.path.join(MODEL_DIR, 'yamnet-tensorflow2-yamnet-v1')
if not os.path.isdir(yamnet_model_extracted_path):
    print(f"[{datetime.now()}] YAMNet model directory not found. Initiating download and extraction.")
    download_file_from_google_drive(YAMNET_MODEL_FILE_ID, YAMNET_MODEL_ZIP)
    extract_zip(YAMNET_MODEL_ZIP, MODEL_DIR)
    os.remove(YAMNET_MODEL_ZIP) # Clean up the zip file
    print(f"[{datetime.now()}] YAMNet model download and extraction complete.")
else:
    print(f"[{datetime.now()}] YAMNet model already present at {yamnet_model_extracted_path}.")

# Download and extract class map if not already present
class_map_file_path = os.path.join(MODEL_DIR, 'yamnet_class_map.csv')
if not os.path.isfile(class_map_file_path):
    print(f"[{datetime.now()}] Class map CSV not found. Initiating download and extraction.")
    download_file_from_google_drive(CLASS_MAP_FILE_ID, CLASS_MAP_ZIP)
    extract_zip(CLASS_MAP_ZIP, MODEL_DIR)
    os.remove(CLASS_MAP_ZIP) # Clean up the zip file
    print(f"[{datetime.now()}] Class map download and extraction complete.")
else:
    print(f"[{datetime.now()}] Class map already present at {class_map_file_path}.")


print(f"[{datetime.now()}] Application startup: Loading YAMNet TensorFlow model...")
yamnet_model = tf.saved_model.load(yamnet_model_extracted_path)
print(f"[{datetime.now()}] Application startup: YAMNet TensorFlow model loaded.")

print(f"[{datetime.now()}] Application startup: Loading class names from CSV...")
class_names = pd.read_csv(class_map_file_path)['display_name'].tolist()
print(f"[{datetime.now()}] Application startup: Class names loaded.")

# Define hazardous and semi-immediate classes based on their indices
hazardous_classes = {11, 102, 181, 280, 281, 307, 316, 317, 318, 319,
                     390, 393, 394, 420, 421, 422, 423, 424, 428, 429}
semi_immediate_classes = {302, 312}

# Audio processing constants
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# --- Audio Loading and Classification Functions ---
def load_audio(file_path):
    # librosa.load will resample the audio to SAMPLE_RATE if needed
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return waveform

def classify_single_chunk(waveform):
    # Ensure waveform is CHUNK_SAMPLES long for YAMNet input
    if len(waveform) < CHUNK_SAMPLES:
        # Pad with zeros if shorter
        waveform = np.pad(waveform, (0, CHUNK_SAMPLES - len(waveform)), mode='constant')
    else:
        # Truncate if longer
        waveform = waveform[:CHUNK_SAMPLES]

    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform_tensor)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy() # Get mean scores across frames

    top_class_index = int(np.argmax(mean_scores))
    prediction = {
        'class_name': class_names[top_class_index],
        'score': round(float(mean_scores[top_class_index]), 4),
        'hazardous': top_class_index in hazardous_classes,
        'semi_immediate': top_class_index in semi_immediate_classes
    }
    return prediction

# --- Flask Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(f"[{datetime.now()}] API Request: Received request.")
        if 'audio' not in request.files:
            print(f"[{datetime.now()}] API Request Error: No audio file provided.")
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        print(f"[{datetime.now()}] API Request: File received: {file.filename}")

        # Save to a temporary file for librosa to load
        temp_path = 'temp_audio_input.wav' # Use a more descriptive temp name
        file.save(temp_path)
        print(f"[{datetime.now()}] API Request: Audio saved to temp file.")

        waveform = load_audio(temp_path)
        print(f"[{datetime.now()}] API Request: Audio loaded into waveform.")

        prediction = classify_single_chunk(waveform)
        print(f"[{datetime.now()}] API Request: Classification complete.")

        os.remove(temp_path) # Clean up temp file
        print(f"[{datetime.now()}] API Request: Temp file removed. Prediction successful.")
        return jsonify({'predictions': [prediction]})

    except Exception as e:
        # Log the full traceback for debugging
        print(f"[{datetime.now()}] API Request Error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Prediction error. See server logs for details.'}), 500

# --- Entry point for local development (ignored by Gunicorn) ---
if __name__ == '__main__':
    # When running locally, Flask's development server will run on this port
    port = int(os.environ.get("PORT", 5001))
    print(f"[{datetime.now()}] Running Flask development server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)