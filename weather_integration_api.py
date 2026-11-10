
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from datetime import timedelta
from tensorflow.keras.models import load_model

from weather_api import get_weather_forecast


# --------------------------------
# 1. Configuration
# --------------------------------

CITY = "Delhi"

LATITUDE = 28.6139
LONGITUDE = 77.2090

SEQUENCE_LENGTH = 30
FORECAST_DAYS = 7

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "rain_sum"
]


# --------------------------------
# 2. Load trained LSTM model
# --------------------------------

print("Loading trained LSTM model...")

model = load_model("trained_models/weather_model.keras")

scaler = joblib.load("trained_models/weather_scaler.pkl")

print("LSTM Model Loaded Successfully!")


# --------------------------------
# 3. Load historical dataset
# --------------------------------

print("\nLoading historical weather data...")

df = pd.read_csv("datasets/weather_data.csv")

df["date"] = pd.to_datetime(df["date"])

city_data = df[df["city"] == CITY].copy()

city_data = city_data.sort_values("date")

if len(city_data) < SEQUENCE_LENGTH:
    raise ValueError("Not enough historical data for LSTM prediction.")

data = city_data[FEATURES].values

# Last 30 days of historical data
sequence = scaler.transform(data[-SEQUENCE_LENGTH:])

last_date = city_data["date"].iloc[-1]

print("Historical Data Loaded!")
print("Last Dataset Date:", last_date.date())


# --------------------------------
# 4. Generate LSTM predictions
# --------------------------------

print("\nGenerating LSTM predictions...")

predictions = []

for day in range(FORECAST_DAYS):

    input_data = np.array([sequence])

    prediction = model.predict(input_data, verbose=0)[0]

    predictions.append(prediction)

    # Update sequence for next prediction
    sequence = np.vstack([
        sequence[1:],
        prediction
    ])


# Convert scaled predictions to actual values
predictions = scaler.inverse_transform(
    np.array(predictions)
)

# Rainfall cannot be negative
predictions[:, 2] = np.maximum(predictions[:, 2], 0)

lstm_dates = [
    last_date + timedelta(days=i + 1)
    for i in range(FORECAST_DAYS)
]

lstm_df = pd.DataFrame(
    predictions,
    columns=FEATURES
)

lstm_df.insert(0, "date", lstm_dates)

print("\nLSTM Predictions:")
print(lstm_df.round(2).to_string(index=False))


# --------------------------------
# 5. Fetch Weather API forecast
# --------------------------------

print("\nFetching Delhi Weather API forecast...")

api_forecast = get_weather_forecast(
    LATITUDE,
    LONGITUDE,
    days=FORECAST_DAYS
)

if api_forecast is None or api_forecast.empty:

    raise ValueError("Unable to fetch weather API data.")

print("\nWeather API Connected Successfully!")


# Select the same three features
api_df = api_forecast[
    ["time"] + FEATURES
].copy()

api_df = api_df.rename(
    columns={"time": "date"}
)

api_df["date"] = pd.to_datetime(api_df["date"])

print("\nAPI Weather Forecast:")
print(api_df.round(2).to_string(index=False))


# --------------------------------
# 6. Display forecast information
# --------------------------------

print("\n===================================")
print("       AGRIX WEATHER RESULTS")
print("===================================")

print("\nLSTM Forecast:")
print(lstm_df.round(2).to_string(index=False))

print("\nWeather API Forecast:")
print(api_df.round(2).to_string(index=False))


# --------------------------------
# 7. Check date alignment
# --------------------------------

api_start_date = api_df["date"].min().date()
lstm_start_date = lstm_df["date"].min().date()

print("\nDate Alignment Check:")

if api_start_date == lstm_start_date:

    comparison = pd.merge(
        api_df,
        lstm_df,
        on="date",
        suffixes=("_api", "_lstm")
    )

    print("\nAPI vs LSTM Comparison:")
    print(comparison.round(2).to_string(index=False))

else:

    print("The API and LSTM forecast dates do not match.")
    print("API starts:", api_start_date)
    print("LSTM starts:", lstm_start_date)

    print(
        "\nBoth forecasts are displayed separately. "
        "Do not calculate prediction accuracy until "
        "their dates and actual observations are aligned."
    )

print("\nAgriX Weather Integration Test Completed!")
