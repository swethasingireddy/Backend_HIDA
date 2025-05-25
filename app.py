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
from datetime import datetime #

app = Flask(__name__)
CORS(app)


# --- Configuration and Model Loading ---
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, 'models')


os.makedirs(MODEL_DIR, exist_ok=True)


yamnet_model_path = os.path.join(MODEL_DIR, 'yamnet-tensorflow2-yamnet-v1')
class_map_path = os.path.join(MODEL_DIR, 'yamnet_class_map.csv')

print(f"[{datetime.now()}] Application startup: Starting model and class map loading.")



print(f"[{datetime.now()}] Application startup: Loading YAMNet TensorFlow model from {yamnet_model_path}...")
try:
    yamnet_model = tf.saved_model.load(yamnet_model_path)
    print(f"[{datetime.now()}] Application startup: YAMNet TensorFlow model loaded successfully.")
except Exception as e:
    print(f"[{datetime.now()}] ERROR: Failed to load YAMNet model from {yamnet_model_path}. "
          f"Ensure the 'yamnet-tensorflow2-yamnet-v1' directory is present in the 'models' folder in your repository.")
    print(f"[{datetime.now()}] Error details: {e}")
    raise

print(f"[{datetime.now()}] Application startup: Loading class names from {class_map_path}...")
try:
    class_names = pd.read_csv(class_map_path)['display_name'].tolist()
    print(f"[{datetime.now()}] Application startup: Class names loaded successfully.")
except Exception as e:
    print(f"[{datetime.now()}] ERROR: Failed to load class map from {class_map_path}. "
          f"Ensure 'yamnet_class_map.csv' is present in the 'models' folder in your repository.")
    print(f"[{datetime.now()}] Error details: {e}")
    # Re-raise or exit if class map loading is critical
    raise

# Defining hazardous and semi-immediate classes 
hazardous_classes = {11, 102, 181, 280, 281, 307, 316, 317, 318, 319,
                     390, 393, 394, 420, 421, 422, 423, 424, 428, 429}
semi_immediate_classes = {302, 312}

# Audio processing constants
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


def load_audio(file_path):
    #
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return waveform

def classify_single_chunk(waveform):
    
    if len(waveform) < CHUNK_SAMPLES:
   
        waveform = np.pad(waveform, (0, CHUNK_SAMPLES - len(waveform)), mode='constant')
    else:
     
        waveform = waveform[:CHUNK_SAMPLES]

    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform_tensor)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy() 

    top_class_index = int(np.argmax(mean_scores))
    prediction = {
        'class_name': class_names[top_class_index],
        'score': round(float(mean_scores[top_class_index]), 4),
        'hazardous': top_class_index in hazardous_classes,
        'semi_immediate': top_class_index in semi_immediate_classes
    }
    return prediction


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
        
        print(f"[{datetime.now()}] API Request Error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Prediction error. See server logs for details.'}), 500


if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5001))
    print(f"[{datetime.now()}] Running Flask development server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)