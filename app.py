from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 🔹 Load trained model
model = joblib.load("ev_soc_model.pkl")  # Make sure the file is in the same folder

# 🔹 Weather mapping dictionary
weather_mapping = {
    # ☀️ 1 = Clear weather
    "clear": 1,
    "sunny": 1,

    # ☁️ 2 = Cloudy-related weather
    "clouds": 2,
    "few clouds": 2,
    "scattered clouds": 2,
    "broken clouds": 2,
    "overcast clouds": 2,
    "slightly cloudy": 2,

    # 🌧️ 3 = Rainy or wet weather
    "rain": 3,
    "drizzle": 3,
    "thunderstorm": 3,
    "shower rain": 3,
    "light rain": 3,
    "moderate rain": 3,
    "heavy rain": 3,
    "ragged shower rain": 3,

    # 🌫️❄️ 4 = Difficult driving/weather conditions
    "mist": 4,
    "fog": 4,
    "haze": 4,
    "smoke": 4,
    "dust": 4,
    "sand": 4,
    "ash": 4,
    "squall": 4,
    "tornado": 4,
    "snow": 4,
    "sleet": 4,
    "freezing rain": 4,
    "dark": 4
}


@app.route('/predict_soc', methods=['POST'])
def predict_soc():
    try:
        data = request.get_json()

        distance = float(data['distance'])         # km
        velocity = float(data['velocity'])         # km/h
        temperature = float(data['temperature'])   # °C
        weather_str = data['weather'].strip().lower()

        # Convert string weather to numeric
        weather = weather_mapping.get(weather_str, 1)  # Default to 1 (sunny) if not found

        # 🔹 Prepare input and predict
        input_data = np.array([[velocity, temperature, weather]])
        predicted_soc_per_km = model.predict(input_data)[0]

        total_soc = predicted_soc_per_km * distance * 0.55 * 1000

        return jsonify({'soc': round(total_soc, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
