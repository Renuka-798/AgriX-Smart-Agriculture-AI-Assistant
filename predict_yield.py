
import joblib

# Load optimized model
model = joblib.load(
    "trained_models/yield_model_optimized.pkl"
)

# Load matching encoders
region_encoder = joblib.load(
    "trained_models/region_encoder_optimized.pkl"
)

soil_encoder = joblib.load(
    "trained_models/soil_encoder_optimized.pkl"
)

crop_encoder = joblib.load(
    "trained_models/crop_encoder_optimized.pkl"
)

weather_encoder = joblib.load(
    "trained_models/weather_encoder_optimized.pkl"
)


def predict_yield(
    region,
    soil,
    crop,
    rainfall,
    temperature,
    fertilizer_used,
    irrigation_used,
    weather,
    days_to_harvest
):

    # Encode categorical inputs
    region = region_encoder.transform([region])[0]
    soil = soil_encoder.transform([soil])[0]
    crop = crop_encoder.transform([crop])[0]
    weather = weather_encoder.transform([weather])[0]

    # Convert boolean values into integers
    fertilizer_used = int(fertilizer_used)
    irrigation_used = int(irrigation_used)

    # Predict crop yield
    prediction = model.predict([[
        region,
        soil,
        crop,
        rainfall,
        temperature,
        fertilizer_used,
        irrigation_used,
        weather,
        days_to_harvest
    ]])[0]

    return round(prediction, 2)
