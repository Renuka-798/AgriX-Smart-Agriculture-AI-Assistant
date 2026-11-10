
import os
import joblib
import streamlit as st

from huggingface_hub import hf_hub_download


@st.cache_resource
def load_fertilizer_assets():

    # Read the Hugging Face token from Streamlit secrets
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        hf_token = None

    # Download the large model from Hugging Face
    model_path = hf_hub_download(
        repo_id="Renuka-970/agrix-fertilizer-model",
        filename="fertilizer_model.pkl",
        repo_type="model",
        token=hf_token
    )

    # Load the model and encoders
    model = joblib.load(model_path)

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

    return (
        model,
        region_encoder,
        soil_encoder,
        crop_encoder,
        weather_encoder
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

    (
        model,
        region_encoder,
        soil_encoder,
        crop_encoder,
        weather_encoder
    ) = load_fertilizer_assets()

    region = region_encoder.transform([region])[0]
    soil = soil_encoder.transform([soil])[0]
    crop = crop_encoder.transform([crop])[0]
    weather = weather_encoder.transform([weather])[0]

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
