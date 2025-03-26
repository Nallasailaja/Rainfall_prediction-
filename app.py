from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model, scaler, and expected features
model_path = 'model/rainfall_prediction_model.pkl'
scaler_path = 'model/scaler.pkl'
features_path = 'model/expected_features.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    expected_features = joblib.load(features_path)  # Load expected feature names
else:
    model = scaler = expected_features = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler or not expected_features:
        return "❌ Model, Scaler, or expected features not loaded. Please check the 'model/' directory."

    try:
        # Get input data from form
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        precipitation = float(request.form['precipitation'])
        cloud_cover = float(request.form['cloud_cover'])
        pressure = float(request.form['pressure'])
        location = request.form['location'].strip()  # User-typed location

        # Create input data dictionary
        user_data = {
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Wind Speed': [wind_speed],
            'Precipitation': [precipitation],
            'Cloud Cover': [cloud_cover],
            'Pressure': [pressure]
        }

        # Set all expected location features to 0 initially
        for feature in expected_features:
            if feature.startswith('Location_'):
                user_data[feature] = [0]

        # Set the location feature to 1 if it exists in expected features
        location_column = f"Location_{location}"
        if location_column in expected_features:
            user_data[location_column] = [1]

        # Create a DataFrame with expected features
        input_df = pd.DataFrame(user_data)
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        result = "🌧️ Likely to rain tomorrow!" if prediction == 1 else "☀️ No rain expected tomorrow!"
        
        return render_template('index.html', result=result)

    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
