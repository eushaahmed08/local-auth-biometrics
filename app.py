from flask import Flask, render_template, request, jsonify
import csv
import os
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
DATA_FILE = 'touch_data.csv'

# 1. LOAD THE BRAIN
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("AI Brain Loaded Successfully!")
except:
    print("AI Brain not found. Please train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/log_touch', methods=['POST'])
def log_touch():
    data = request.json
    
    # 2. PREPARE THE DATA FOR PREDICTION
    # We need to calculate the same features we used in training
    # For a live test, we'll use a simplified version of speed/dist
    x = data.get('x')
    y = data.get('y')
    pressure = data.get('pressure')
    
    # Placeholder for live feature engineering (simplified)
    dist = 0.5 
    speed = 0.5
    is_pattern = 1 if data.get('auth_type') == 'pattern' else 0
    is_pin = 1 if data.get('auth_type') == 'pin' else 0

    # 3. ASK THE AI
    input_data = np.array([[x, y, pressure, dist, speed, is_pattern, is_pin]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0] # Will return 'genuine' or 'imposter'
    
    # Log the data as usual
    file_exists = os.path.isfile(DATA_FILE)
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['user_id', 'session_id', 'auth_type', 'key', 'x', 'y', 'pressure', 'timestamp'])
        writer.writerow(['genuine', data.get('session_id'), data.get('auth_type'), data.get('key'), x, y, pressure, data.get('duration')])

    return jsonify({
        "status": "success", 
        "prediction": prediction,
        "trust_score": "High" if prediction == 'genuine' else "Low"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)