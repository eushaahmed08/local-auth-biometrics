import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
app = Flask(__name__)

# 1. CONNECT TO SUPABASE
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# 2. LOAD THE XGBOOST BRAIN
def load_brain():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'model.pkl')
    
    try:
        if os.path.exists(model_path):
            # Use pickle for XGBoost
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✅ XGBoost Hybrid Brain Loaded Successfully!")
            return model
        else:
            print(f"⚠️ model.pkl not found at: {model_path}")
            return None
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return None

model = load_brain()

# --- HELPER: UPDATED FEATURE ENGINEERING ---
def extract_features(points, auth_type_str):
    """
    Must match the Colab logic EXACTLY.
    Order: auth_kind, avg_accel, max_accel, avg_velocity, total_time, width, height, point_count
    """
    if len(points) < 2:
        return None
    
    df_p = pd.DataFrame(points)
    
    # 1. auth_kind (1 for pattern, 0 for pin)
    auth_kind = 1 if auth_type_str == "pattern" else 0
    
    # 2 & 3. Acceleration stats
    # Using .get in case the key name varies
    accel_col = 'acceleration' if 'acceleration' in df_p.columns else 'accel'
    avg_accel = df_p[accel_col].mean()
    max_accel = df_p[accel_col].max()
    
    # 4 & 5. Velocity and Time
    # Note: Use 'duration' or 'time' consistently
    time_col = 'duration' if 'duration' in df_p.columns else 'time'
    dist = np.sqrt(df_p['x'].diff()**2 + df_p['y'].diff()**2).sum()
    total_time = df_p[time_col].max() - df_p[time_col].min()
    avg_velocity = dist / (total_time + 1) # +1 to prevent div by zero
    
    # 6 & 7. Geometry
    width = df_p['x'].max() - df_p['x'].min()
    height = df_p['y'].max() - df_p['y'].min()
    
    # 8. Point Count
    point_count = len(df_p)
    
    features = [
        auth_kind, avg_accel, max_accel, avg_velocity, 
        total_time, width, height, point_count
    ]
    
    print(f"📊 Features for {auth_type_str}: Vel={avg_velocity:.2f}, Time={total_time}ms")
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/collect', methods=['POST'])
def collect():
    global model
    data = request.json
    points = data.get('points', [])
    user_id = data.get('user_id', 'unknown')
    auth_type = data.get('auth_type', 'unknown')
    session_id = data.get('session_id', 'unknown')

    if not points:
        return jsonify({"status": "error", "message": "No points received"}), 400

    # Save raw points to Supabase for future training
    db_rows = [{"user_id": user_id, "session_id": session_id, "auth_type": auth_type, 
                "x": p['x'], "y": p['y'], "pressure": p.get('pressure', 1),
                "touch_size": p.get('size', 1), "acceleration": p.get('accel', 0),
                "duration": p['time']} for p in points]
    
    try:
        supabase.table("biometric_logs").insert(db_rows).execute()
    except Exception as e:
        print(f"Supabase Error: {e}")

    # AI Prediction Logic
    prediction_result = "Learning..."
    trust_score = "N/A"

    if model:
        # Pass both points and auth_type
        features_list = extract_features(points, auth_type)
        
        if features_list:
            # XGBoost expects a 2D array (DataFrame or NumPy)
            # We convert to DataFrame to keep feature names if possible, or just a 2D array
            features_array = np.array([features_list])
            
            # 1 = Genuine, 0 = Imposter (based on our Colab labeling)
            pred_code = model.predict(features_array)[0]
            
            # Get probability (confidence)
            prob = model.predict_proba(features_array)[0][1]
            
            prediction_result = "genuine" if pred_code == 1 else "imposter"
            trust_score = f"{prob*100:.1f}% Match"

    return jsonify({
        "status": "success", 
        "prediction": prediction_result, 
        "trust_score": trust_score
    })

# Note: We removed the /train route here. 
# It is better to train in Colab where you have better visualization and power.

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)