
from weather_api import (
    get_weather_forecast,
    get_historical_weather
)

# Delhi coordinates
latitude = 28.6139
longitude = 77.2090

# Test historical weather
print("\nFetching Historical Weather...")

history = get_historical_weather(
    latitude,
    longitude,
    days=30
)

if history is not None:

    print("\nHistorical Weather Retrieved Successfully!")

    print(history.to_string(index=False))

    print("\nHistorical Start:", history["time"].min())
    print("Historical End:", history["time"].max())

else:

    print("Unable to fetch historical weather.")


# Test future forecast
print("\nFetching 7-Day Forecast...")

forecast = get_weather_forecast(
    latitude,
    longitude,
    days=7
)

if forecast is not None:

    print("\nWeather Forecast Retrieved Successfully!")

    print(forecast.to_string(index=False))

else:

    print("Unable to fetch forecast.")
