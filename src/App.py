from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os

# Dynamically construct the absolute paths to the models
model1_path = os.path.join(os.path.dirname(__file__), "model", "model1.keras")
model2_path = os.path.join(os.path.dirname(__file__), "model", "model2.keras")

# Load models
if os.path.exists(model1_path):
    model1 = tf.keras.models.load_model(model1_path)
    print("Model1 (next day) loaded successfully.")
else:
    raise FileNotFoundError(f"Model file not found at {model1_path}")

if os.path.exists(model2_path):
    model2 = tf.keras.models.load_model(model2_path)
    print("Model2 (next 3 days) loaded successfully.")
else:
    raise FileNotFoundError(f"Model file not found at {model2_path}")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        features = data.get('features')
        model_type = data.get('predictionType')

        # Ensure features array is correctly formatted
        input_data = np.array(features).reshape(1, 30, 6)

        # Select the model based on prediction type
        if model_type == "next_day":
            prediction = model1.predict(input_data)
        elif model_type == "next_3_days":
            prediction = model2.predict(input_data)
        else:
            raise ValueError("Invalid model_type. Choose 'next_day' or 'next_3_days'.")

        # Send response
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
