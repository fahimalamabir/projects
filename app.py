from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the model, scaler, and imputer
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['features']])
    
    # Preprocess the features
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)
    
    # Make a prediction
    prediction = model.predict(features_scaled)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
