import joblib


model = joblib.load(
    "trained_models/fertilizer_model.pkl"
)

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


def predict_fertilizer(
    region,
    soil,
    crop,
    rainfall,
    temperature,
    weather,
    days_to_harvest,
    yield_tons
):

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

    prediction = model.predict(
        [[
            region,
            soil,
            crop,
            rainfall,
            temperature,
            weather,
            days_to_harvest,
            yield_tons
        ]]
    )[0]

    if prediction == 1:
        return "🌱 Fertilizer Recommended"

    return "❌ Fertilizer Not Required"