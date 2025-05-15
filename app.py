import os
from flask import Flask, render_template, request
import pickle
import numpy as np

# Print some debug info
print("Current working directory:", os.getcwd())
print("Templates folder exists:", os.path.exists("templates/index.html"))

# Load your trained model
try:
    model = pickle.load(open("classifier1.pkl", "rb"))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        print("Received form data:", request.form)

        # Extract input features from form
        features = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        print("Parsed features:", features)

        # Convert to array and predict
        data = np.array([features])
        prediction = model.predict(data)

        print("Raw model prediction:", prediction)  # Debug print

        # Use prediction directly (e.g., 'Maize')
        predicted_label = prediction[0]
        print("Predicted Label:", predicted_label)

        return render_template("index.html", result=predicted_label)

    except Exception as e:
        print("Error during prediction:", e)
        return f"<h2>Error occurred: {e}</h2>"

if __name__ == "__main__":
    app.run(debug=True)
