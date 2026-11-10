import joblib


# Load model
model = joblib.load(
    "trained_models/irrigation_model.pkl"
)

# Load encoders
region_encoder = joblib.load(
    "trained_models/region_encoder.pkl"
)

soil_encoder = joblib.load(
    "trained_models/soil_encoder.pkl"
)

crop_encoder = joblib.load(
    "trained_models/crop_encoder.pkl"
)

weather_encoder = joblib.load(
    "trained_models/weather_encoder.pkl"
)


def predict_irrigation(
    region,
    soil,
    crop,
    rainfall,
    temperature,
    fertilizer_used,
    weather,
    days_to_harvest,
    yield_tons
):

    # Encode inputs
    region = region_encoder.transform(
        [region]
    )[0]

    soil = soil_encoder.transform(
        [soil]
    )[0]

    crop = crop_encoder.transform(
        [crop]
    )[0]

    weather = weather_encoder.transform(
        [weather]
    )[0]

    fertilizer_used = int(
        fertilizer_used
    )

    prediction = model.predict(
        [[
            region,
            soil,
            crop,
            rainfall,
            temperature,
            fertilizer_used,
            weather,
            days_to_harvest,
            yield_tons
        ]]
    )[0]

    if prediction == 1:
        return "💧 Irrigation Required"

    return "✅ No Irrigation Needed"