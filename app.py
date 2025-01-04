from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json, Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, BatchNormalization # type: ignore

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

custom_objects = {
    "Sequential": Sequential,
    "Conv1D": Conv1D,
    "BatchNormalization": BatchNormalization,
}

with open('model/CNN_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
loaded_model.load_weights('model/best_model1_weights.h5')

with open('model/scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('model/encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    result = np.hstack([np.squeeze(zcr), np.squeeze(rmse), np.ravel(mfcc.T)])
    target_size = 2376
    if len(result) < target_size:
        result = np.pad(result, (0, target_size - len(result)), mode='constant')
    else:
        result = result[:target_size]
    return np.expand_dims(scaler2.transform([result]), axis=2)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        data, sr = librosa.load(file_path, sr=22050, duration=2.5, offset=0.6)
        features = extract_features(data)
        prediction = loaded_model.predict(features)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_label = encoder2.categories_[0][predicted_index]
        return jsonify({"emotion": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True) 