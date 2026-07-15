from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the saved model and feature names
model = joblib.load('model/model.pkl')
feature_names = joblib.load('model/feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Build DataFrame with correct column order
        input_dict = {}
        for col in feature_names:
            if col not in data:
                return jsonify({'error': f'Missing field: {col}'}), 400
            input_dict[col] = data[col]

        input_df = pd.DataFrame([input_dict])[feature_names]
        prediction = model.predict(input_df)[0]
        return jsonify({'predicted_price': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)