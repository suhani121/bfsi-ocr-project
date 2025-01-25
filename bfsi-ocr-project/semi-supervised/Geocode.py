from flask import Flask, request, jsonify
import os
import requests

# Load environment variables (update API_KEY securely in a .env file)
API_KEY = "Your API key"
BASE_URL = "http://api.openweathermap.org/geo/1.0/direct"

# Initialize Flask app
app = Flask(__name__)

@app.route("/geocode", methods=["GET"])
def geocode():
    # Retrieve parameters from request
    city = request.args.get("city")
    state = request.args.get("state", "")
    country = request.args.get("country", "")
    limit = request.args.get("limit", 1)

# Construct query and parameters
    query = f"{city},{state},{country}".strip(',')
    params = {
        "q": query,
        "limit": limit,
        "appid": API_KEY
    }

        # Make a request to the GeoCoding API
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return jsonify(data), 200

@app.route("/weather", methods=["GET"])
def weather():
    
    # Retrieve parameters from request
    city = request.args.get("city")
    state = request.args.get("state", "")
    country = request.args.get("country", "")
    limit = request.args.get("limit", 1)


if __name__ == "__main__":
    app.run(debug=True)
