import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify


def create_app(model_path=None):
    app = Flask(__name__)

    if model_path is None:
        model_path = os.path.join(os.getcwd(), 'models', 'rf_model.pkl')

    if not os.path.exists(model_path):
        raise SystemExit(f"Model file not found: {model_path}. Run training first.")

    m = joblib.load(model_path)
    model = m.get('model')
    features = m.get('features')

    if model is None or features is None:
        raise SystemExit('Saved model invalid: missing model or features list')

    app.config['model'] = model
    app.config['features'] = features

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'}), 200

    @app.route('/predict', methods=['POST'])
    def predict():
        payload = request.get_json(force=True)
        if payload is None:
            return jsonify({'error': 'Invalid or missing JSON payload'}), 400

        # Accept either a features dict or a list of values
        features = app.config['features']

        # 1) If client provides full dict of feature_name->value
        if 'features' in payload and isinstance(payload['features'], dict):
            feat_dict = payload['features']
            try:
                row = [feat_dict[f] for f in features]
            except KeyError as e:
                return jsonify({'error': f'Missing feature in payload: {e}'}), 400
            X = pd.DataFrame([row], columns=features)

        # 2) If client provides values as list in the correct order
        elif 'values' in payload and isinstance(payload['values'], (list, tuple)):
            vals = payload['values']
            if len(vals) != len(features):
                return jsonify({'error': f'Expected {len(features)} values, got {len(vals)}'}), 400
            X = pd.DataFrame([list(vals)], columns=features)

        # 3) If client asks to predict by row index from dataset (for convenience)
        elif 'row_index' in payload:
            row_index = int(payload['row_index'])
            csv_path = payload.get('csv') or os.path.join(os.getcwd(), 'dataset_tomat.csv')
            if not os.path.exists(csv_path):
                return jsonify({'error': f'CSV not found: {csv_path}'}), 400
            df = pd.read_csv(csv_path)
            try:
                X = pd.DataFrame([df.loc[row_index, features].values], columns=features)
            except Exception as e:
                return jsonify({'error': f'Failed to read row_index: {e}'}), 400

        else:
            return jsonify({'error': 'Payload must contain one of: "features" (dict), "values" (list), or "row_index" (int)'}), 400

        # Ensure numeric types
        try:
            X = X.astype(float)
        except Exception:
            pass

        model = app.config['model']
        try:
            pred = model.predict(X)
        except Exception as e:
            return jsonify({'error': f'Model prediction failed: {e}'}), 500

        # Return numpy types as native
        pred_val = float(np.asarray(pred).ravel()[0])
        return jsonify({'prediction': pred_val}), 200

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
