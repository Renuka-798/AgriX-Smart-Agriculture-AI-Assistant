
import requests
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo


# ---------------------------------------
# 1. Common Open-Meteo configuration
# ---------------------------------------

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_FEATURES = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration"
]


# ---------------------------------------
# 2. Get 7-day weather forecast
# ---------------------------------------

def get_weather_forecast(latitude, longitude, days=7):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": DAILY_FEATURES,
        "timezone": "auto",
        "forecast_days": days
    }

    try:

        response = requests.get(
            FORECAST_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        forecast = pd.DataFrame(data["daily"])

        return forecast

    except requests.exceptions.RequestException as e:

        print("Weather API Error:", e)
        return None

    except (KeyError, ValueError) as e:

        print("Invalid weather response:", e)
        return None


# ---------------------------------------
# 3. Get latest 30 days of historical data
# ---------------------------------------

def get_historical_weather(latitude, longitude, days=30):

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
            FORECAST_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        historical = pd.DataFrame(data["daily"])

        historical["time"] = pd.to_datetime(
            historical["time"]
        )

        # Determine today's date in the API location's timezone
        timezone = data.get("timezone", "UTC")

        today = datetime.now(
            ZoneInfo(timezone)
        ).date()

        # Keep completed days only
        historical = historical[
            historical["time"].dt.date < today
        ]

        historical = historical.sort_values("time")

        # Keep the latest requested number of days
        historical = historical.tail(days).reset_index(drop=True)

        if len(historical) < days:

            print(
                f"Warning: Only {len(historical)} "
                f"historical days were returned."
            )

            return None

        print(
            f"Retrieved {len(historical)} historical weather records."
        )

        return historical

    except requests.exceptions.RequestException as e:

        print("Historical Weather API Error:", e)
        return None

    except (KeyError, ValueError, TypeError) as e:

        print("Invalid historical weather response:", e)
        return None
