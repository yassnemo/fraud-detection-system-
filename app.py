from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/random_forest_model.joblib')

@app.route('/')
def home():
    return "Welcome to the Fraud Detection API! Use the /predict endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])
    
    # any necessary preprocessing steps to be added here if needed.

    prediction = model.predict(input_df)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
