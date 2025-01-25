from flask import Flask, request, render_template, jsonify
import os
import requests
API_KEY = "Your API key"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        city = request.form.get("city")
        if city:
            weather_data = get_weather(city)
            if weather_data:
                return render_template("index.html", weather=weather_data)
            else:
                return render_template("index.html", error="City not found!")
    return render_template("index.html")

def get_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

@app.route("/api/weather", methods=["GET"])
def api_weather():
    city = request.args.get("city")
    if city:
        weather_data = get_weather(city)
        if weather_data:
            return jsonify(weather_data)
        return jsonify({"error": "City not found"}), 404
    return jsonify({"error": "City parameter is required"}), 400

if __name__ == "__main__":
    app.run(debug=True)
