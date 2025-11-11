import os
import traceback
from flask import Flask, send_from_directory, request, jsonify

# Simple Flask backend to support server-side ML eviction decisions.
# Place your serialized model at: Models/model.pkl (use joblib for sklearn models)
# See the comments in predict_evict() for where to adapt feature engineering.

app = Flask(__name__, static_folder='')  # serve files from project root
MODEL_PATH = os.path.join('Models', 'Models/random_forest_cache.pkl')
model = None

# Try to load model if available. We use joblib for sklearn artifacts.
try:
    import joblib
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"No model found at {MODEL_PATH}. Server ML will fallback to heuristic.")
except Exception as e:
    print('Could not load model or joblib not installed:', e)
    model = None


@app.route('/')
def index():
    return send_from_directory('.', 'cache.html')


@app.route('/predict-evict', methods=['POST'])
def predict_evict():
    """
    Expect JSON: { cache: [...], cacheHistory: { pageId: { frequency: <int>, lastUsed: <ms> } } }
    Return JSON: { evict: <pageId> }

    How to adapt for your model:
    - Replace the feature vector construction below with the exact features your model
      expects. For example, normalize recency: current_time_ms - lastUsed, scale values,
      add one-hot / embedding for pageId, sliding-window stats, etc.
    - Prefer saving a Pipeline (preprocessor + estimator) using joblib so server code
      can call model.predict directly without extra preprocessing.
    """
    try:
        data = request.get_json(force=True)
        cache = data.get('cache', [])
        cacheHistory = data.get('cacheHistory', {})

        # If no model available, fallback to heuristic: least frequent then least recent
        if model is None:
            evict = None
            lowestFreq = float('inf')
            oldestTime = float('inf')
            for pageId in cache:
                stats = cacheHistory.get(pageId, {'frequency': 0, 'lastUsed': 0})
                freq = stats.get('frequency', 0)
                last = stats.get('lastUsed', 0)
                if freq < lowestFreq:
                    lowestFreq = freq
                    oldestTime = last
                    evict = pageId
                elif freq == lowestFreq and last < oldestTime:
                    oldestTime = last
                    evict = pageId
            return jsonify({'evict': evict})

        # Example feature construction (VERY simple): [frequency, lastUsed]
        # Replace with your real features!
        feature_vectors = []
        page_ids = []
        for pageId in cache:
            stats = cacheHistory.get(pageId, {'frequency': 0, 'lastUsed': 0})
            freq = stats.get('frequency', 0)
            last = stats.get('lastUsed', 0)
            # Optionally convert last to recency: current_ms - last
            # recency = int(time.time() * 1000) - last
            feature_vectors.append([freq, last])
            page_ids.append(pageId)

        # Model invocation - adapt to your model's predict API
        preds = model.predict(feature_vectors)
        # Interpret preds: model may return a score for each candidate or a label.
        # This example assumes preds is a 1D array where higher value suggests eviction.
        try:
            import numpy as _np
            evict_index = int(_np.argmax(preds))
        except Exception:
            # If preds is a single label index, try to interpret that
            evict_index = int(preds[0]) if isinstance(preds, (list, tuple)) else 0

        evict = page_ids[evict_index]
        return jsonify({'evict': evict})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Run dev server
    app.run(host='0.0.0.0', port=5000, debug=True)
