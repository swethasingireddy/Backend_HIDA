import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)
CORS(app)


def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    print(f"Starting download from Google Drive file id: {file_id} ...")

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"Download completed: {destination}")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")


BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

YAMNET_MODEL_FILE_ID = '18FX4FDwmSUHZhbuQ_-GBDgtBwwmzyFiL'  
CLASS_MAP_FILE_ID = '1dBjlS4aVXdKqSYhelLb6_W9OKMUFnAVY'    

YAMNET_MODEL_ZIP = os.path.join(BASE_DIR, 'yamnet_model.zip')
CLASS_MAP_ZIP = os.path.join(BASE_DIR, 'yamnet_class_map.zip')


if not os.path.isdir(os.path.join(MODEL_DIR, 'yamnet-tensorflow2-yamnet-v1')):
    download_file_from_google_drive(YAMNET_MODEL_FILE_ID, YAMNET_MODEL_ZIP)
    extract_zip(YAMNET_MODEL_ZIP, MODEL_DIR)
    os.remove(YAMNET_MODEL_ZIP)


if not os.path.isfile(os.path.join(MODEL_DIR, 'yamnet_class_map.csv')):
    download_file_from_google_drive(CLASS_MAP_FILE_ID, CLASS_MAP_ZIP)
    extract_zip(CLASS_MAP_ZIP, MODEL_DIR)
    os.remove(CLASS_MAP_ZIP)


yamnet_model_path = os.path.join(MODEL_DIR, 'yamnet-tensorflow2-yamnet-v1')
yamnet_model = tf.saved_model.load(yamnet_model_path)

class_map_path = os.path.join(MODEL_DIR, 'yamnet_class_map.csv')
class_names = pd.read_csv(class_map_path)['display_name'].tolist()



hazardous_classes = {11, 102, 181, 280, 281, 307, 316, 317, 318, 319,
                     390, 393, 394, 420, 421, 422, 423, 424, 428, 429}
semi_immediate_classes = {302, 312}

SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


def load_audio(file_path):
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

    top_class = int(np.argmax(mean_scores))
    prediction = {
        'class_name': class_names[top_class],
        'score': round(float(mean_scores[top_class]), 4),
        'hazardous': top_class in hazardous_classes,
        'semi_immediate': top_class in semi_immediate_classes
    }
    return prediction


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request")
        if 'audio' not in request.files:
            print("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        print(f"File received: {file.filename}")

        temp_path = 'temp_chunk.wav'
        file.save(temp_path)

        waveform = load_audio(temp_path)
        prediction = classify_single_chunk(waveform)

        os.remove(temp_path)
        return jsonify({'predictions': [prediction]})

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Prediction error.'}), 500

if __name__ == '__main__':
   
    pass
