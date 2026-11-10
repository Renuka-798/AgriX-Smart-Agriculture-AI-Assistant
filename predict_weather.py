
import os
import requests
import pandas as pd
import numpy as np
import joblib

from datetime import datetime
from zoneinfo import ZoneInfo
from tensorflow.keras.models import load_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_DIR = "trained_models"

MODEL_PATH = os.path.join(
    MODEL_DIR, "weather_model.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR, "weather_scaler.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR, "weather_features.pkl"
)

METADATA_PATH = os.path.join(
    MODEL_DIR, "weather_metadata.pkl"
)

SEQUENCE_LENGTH = 30

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "rain_sum"
]

GEOCODING_URL = (
    "https://geocoding-api.open-meteo.com/v1/search"
)

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

print("Loading weather prediction model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Run train_weather_model.py first."
    )

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

# Load feature order saved during training.
if os.path.exists(FEATURES_PATH):
    FEATURES = joblib.load(FEATURES_PATH)

# Load metadata if available.
if os.path.exists(METADATA_PATH):

    metadata = joblib.load(METADATA_PATH)

    SEQUENCE_LENGTH = metadata.get(
        "sequence_length", 30
    )

print("Model loaded successfully!")
print("Sequence length:", SEQUENCE_LENGTH)
print("Features:", FEATURES)


# ==========================================
# 3. DYNAMIC CITY SEARCH
# ==========================================

def search_city(city_name):

    """
    Search for cities using Open-Meteo Geocoding API.

    Returns matching cities with:
    name, country, latitude, longitude, timezone.
    """

    if not city_name or not city_name.strip():
        raise ValueError("Please enter a city name.")

    params = {
        "name": city_name.strip(),
        "count": 10,
        "language": "en",
        "format": "json"
    }

    try:

        response = requests.get(
            GEOCODING_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        if not results:
            return []

        cities = []

        for item in results:

            cities.append({

                "name": item.get("name"),

                "country": item.get("country"),

                "admin1": item.get("admin1", ""),

                "latitude": item.get("latitude"),

                "longitude": item.get("longitude"),

                "timezone": item.get("timezone", "UTC")

            })

        return cities

    except requests.exceptions.RequestException as e:

        raise ConnectionError(
            f"City search failed: {e}"
        )


# ==========================================
# 4. GET CITY DETAILS
# ==========================================

def get_city_details(city_name, country=None, location_index=0):

    """
    Select a city from geocoding results.

    country is optional.
    location_index selects a matching result.
    """

    cities = search_city(city_name)

    if not cities:
        raise ValueError(
            f"No city found with name: {city_name}"
        )

    # Filter by country when provided.
    if country:

        cities = [
            city for city in cities
            if city["country"]
            and city["country"].lower() == country.lower()
        ]

    if not cities:
        raise ValueError(
            f"No matching city found in country: {country}"
        )

    if location_index < 0 or location_index >= len(cities):
        raise ValueError(
            "Invalid city selection."
        )

    return cities[location_index]


# ==========================================
# 5. FETCH RECENT HISTORICAL WEATHER
# ==========================================

def get_historical_weather(latitude, longitude, days=30):

    """
    Fetch recent completed daily weather records.

    These records are used as input to the LSTM model.
    """

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "rain_sum"
        ],

        "timezone": "auto",

        "past_days": days + 1,

        "forecast_days": 1

    }

    try:

        response = requests.get(
            WEATHER_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        if "daily" not in data:
            raise ValueError(
                "Historical weather data unavailable."
            )

        historical = pd.DataFrame(
            data["daily"]
        )

        historical["time"] = pd.to_datetime(
            historical["time"]
        )

        timezone = data.get("timezone", "UTC")

        today = datetime.now(
            ZoneInfo(timezone)
        ).date()

        # Exclude today's incomplete observation.
        historical = historical[
            historical["time"].dt.date < today
        ]

        historical = historical.sort_values(
            "time"
        ).tail(days).reset_index(drop=True)

        # Keep only model features.
        historical = historical[
            ["time"] + FEATURES
        ]

        historical = historical.dropna(
            subset=FEATURES
        )

        if len(historical) < days:

            raise ValueError(
                f"Only {len(historical)} completed "
                f"weather records available. "
                f"Need {days}."
            )

        # Ensure the last 30 observations are consecutive.
        differences = historical["time"].diff().dropna()

        if not (
            differences == pd.Timedelta(days=1)
        ).all():

            raise ValueError(
                "Historical weather contains missing dates. "
                "Cannot safely create LSTM sequences."
            )

        return historical

    except requests.exceptions.RequestException as e:

        raise ConnectionError(
            f"Historical weather API error: {e}"
        )


# ==========================================
# 6. FETCH WEATHER API FORECAST
# ==========================================

def get_weather_forecast(latitude, longitude, days=7):

    """
    Fetch Open-Meteo's weather forecast.
    """

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "et0_fao_evapotranspiration"
        ],

        "timezone": "auto",

        "forecast_days": days

    }

    try:

        response = requests.get(
            WEATHER_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        forecast = pd.DataFrame(
            data["daily"]
        )

        forecast["time"] = pd.to_datetime(
            forecast["time"]
        )

        return forecast

    except requests.exceptions.RequestException as e:

        raise ConnectionError(
            f"Weather forecast API error: {e}"
        )


# ==========================================
# 7. LSTM WEATHER PREDICTION
# ==========================================

def generate_lstm_predictions(historical, days=7):

    """
    Generate recursive LSTM predictions.

    Uses the latest 30 completed days
    to predict the following days.
    """

    if days < 1 or days > 7:
        raise ValueError(
            "Prediction days must be between 1 and 7."
        )

    recent_data = historical[
        FEATURES
    ].tail(SEQUENCE_LENGTH)

    if len(recent_data) < SEQUENCE_LENGTH:

        raise ValueError(
            "Insufficient historical records."
        )

    # Scale historical weather.
    scaled_data = scaler.transform(
        recent_data.values
    )

    current_sequence = scaled_data.copy()

    predictions = []

    last_date = historical["time"].iloc[-1]

    for day in range(days):

        # Reshape input for LSTM.
        input_data = current_sequence.reshape(
            1,
            SEQUENCE_LENGTH,
            len(FEATURES)
        )

        # Predict next day.
        predicted_scaled = model.predict(
            input_data,
            verbose=0
        )

        # Convert scaled values back to original units.
        predicted_values = scaler.inverse_transform(
            predicted_scaled
        )[0]

        # Ensure rainfall is not negative.
        predicted_values[2] = max(
            0,
            predicted_values[2]
        )

        # Ensure maximum temperature >= minimum temperature.
        if predicted_values[0] < predicted_values[1]:

            predicted_values[0], predicted_values[1] = (
                predicted_values[1],
                predicted_values[0]
            )

        prediction_date = (
            last_date + pd.Timedelta(days=day + 1)
        )

        predictions.append({

            "date": prediction_date,

            "temperature_2m_max": round(
                float(predicted_values[0]), 2
            ),

            "temperature_2m_min": round(
                float(predicted_values[1]), 2
            ),

            "rain_sum": round(
                float(predicted_values[2]), 2
            )

        })

        # Add prediction to sequence for next day.
        current_sequence = np.vstack([
            current_sequence[1:],
            predicted_scaled
        ])

    return pd.DataFrame(predictions)


# ==========================================
# 8. MAIN PREDICTION FUNCTION
# ==========================================

def predict_weather(
    city="Delhi",
    days=7,
    country=None,
    location_index=0
):

    """
    Complete AgriX weather prediction workflow.

    1. Search city dynamically.
    2. Fetch historical weather.
    3. Generate LSTM predictions.
    4. Fetch Open-Meteo forecast.
    5. Return all results.
    """

    print("\n================================")
    print("AGRI X WEATHER PREDICTION")
    print("================================")

    # Validate prediction period.
    if days < 1 or days > 7:
        raise ValueError(
            "Days must be between 1 and 7."
        )

    # Step 1: Find city.
    location = get_city_details(
        city,
        country,
        location_index
    )

    latitude = location["latitude"]
    longitude = location["longitude"]

    print("\nSelected Location:")
    print(location["name"], location["country"])

    print("Latitude:", latitude)
    print("Longitude:", longitude)

    # Step 2: Fetch recent history.
    historical = get_historical_weather(
        latitude,
        longitude,
        SEQUENCE_LENGTH
    )

    print("\nHistorical weather fetched successfully!")

    # Step 3: LSTM prediction.
    lstm_forecast = generate_lstm_predictions(
        historical,
        days
    )

    print("\nLSTM predictions generated!")

    # Step 4: Open-Meteo API forecast.
    api_forecast = get_weather_forecast(
        latitude,
        longitude,
        days
    )

    print("\nWeather API forecast fetched!")

    print("\nLSTM Forecast:")
    print(lstm_forecast.to_string(index=False))

    print("\nOpen-Meteo Forecast:")
    print(api_forecast.to_string(index=False))

    return {

        "location": location,

        "historical": historical,

        "lstm_forecast": lstm_forecast,

        "api_forecast": api_forecast

    }


# ==========================================
# 9. TEST THE MODEL
# ==========================================

if __name__ == "__main__":

    try:

        result = predict_weather(
            city="Kadapa",
            days=7,
            country="India"
        )

        print("\nWeather prediction completed!")

        print("\nLocation:")
        print(result["location"])

        print("\nLSTM Predictions:")
        print(result["lstm_forecast"])

        print("\nAPI Predictions:")
        print(result["api_forecast"])

    except Exception as e:

        print("\nWeather Prediction Error:")
        print(e)
