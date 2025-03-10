import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scale = pickle.load(open('scale.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure model.pkl and scale.pkl exist in the correct path.")
    exit(1)

@app.route('/')  # Home Page
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])  # Prediction Route
def predict():
    try:
        # Reading inputs given by the user
        input_feature = [float(x) for x in request.form.values()]
        
        # Ensure inputs match the expected feature count
        if len(input_feature) != len(scale.feature_names_in_):
            return render_template('result.html', prediction_text="Error: Incorrect number of inputs.")

        # Convert input to DataFrame with correct feature names
        data = pd.DataFrame([input_feature], columns=scale.feature_names_in_)

        # Transform data using the preloaded scaler
        data = scale.transform(data)

        # Make predictions using the loaded model
        prediction = model.predict(data)[0]

        # Render the result page with the prediction
        return render_template('result.html', prediction_text=f"Predicted Traffic Volume: {prediction}")

    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
