import os
import traceback
from flask import Flask, send_from_directory, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder='')

# Path to your trained ML model
MODEL_PATH = os.path.join('Models', 'random_forest_cache.pkl')
model = None

# --- Load ML Model ---
try:
    if os.path.exists(MODEL_PATH): 
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}") 
    else:
        print(f"No model found at {MODEL_PATH}. Server ML will fallback to heuristic.")
except Exception as e:
    print("Could not load model or joblib not installed:", e)
    model = None


# --- Serve Frontend ---
@app.route('/')
def index():
    return send_from_directory('.', 'cache.html')


# --- ML Eviction Endpoint ---
@app.route('/predict-evict', methods=['POST'])
def predict_evict():
        
    try:
        data = request.get_json(force=True)
        pages = data.get('pages', [])
        page_ids = data.get('pageIds', [])

        if not model:
            return jsonify({'error': 'ML model not loaded! Cannot predict eviction.'}), 500

        if not pages or not page_ids:
            return jsonify({'error': 'Invalid input data for prediction.'}), 400

        # --- Convert page stats into features DataFrame ---
        feature_vectors = []
        for p in pages:
            vec = [
                p.get('last_access_time', 0),
                p.get('access_count', 0),
                p.get('recency_rank', 0),
                p.get('access_type', 0),
                p.get('cache_item', 0)
            ]
            feature_vectors.append(vec)

        feature_names = [
            'last_access_time',
            'access_count',
            'recency_rank',
            'access_type',
            'cache_item'
        ]
        feature_df = pd.DataFrame(feature_vectors, columns=feature_names)

        # --- Predict eviction probability ---
        probs = model.predict_proba(feature_df)

        # Each row: [prob_keep, prob_evict]
        evict_scores = [p[1] for p in probs]  # Use probability for class=1 (evict)
        evict_index = int(np.argmax(evict_scores))
        evict_page = page_ids[evict_index]

        print(f"🔮 ML Model selected page '{evict_page}' for eviction.")
        return jsonify({'evict': evict_page})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# --- Run Flask Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
