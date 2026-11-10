
import streamlit as st
import bcrypt
import joblib
import pandas as pd
from groq import Groq

from language import translate

from database import (
    add_user,
    login_user,
    save_prediction,
    get_prediction_history,
    clear_prediction_history
)

from predict_disease import predict_leaf_disease
from predict_irrigation import predict_irrigation
from predict_fertilizer import predict_fertilizer
from predict_yield import predict_yield
from predict_market import predict_market
from predict_weather import (
    predict_weather,
    search_city
)


# ---------------- AI CHATBOT ----------------

@st.cache_resource
def get_groq_client():
    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

def ask_agrix_chatbot(messages, language):

    latest_message = messages[-1]["content"].strip().lower()

    # Instant greeting response
    greetings = ["hi", "hello", "hey", "hii", "hlo"]

    if latest_message in greetings:
        return (
            f"Hello! 🌱 Welcome to AgriX. "
            f"How can I help you with farming today?"
        )

    client = get_groq_client()

    system_prompt = f"""
You are AgriX, a helpful AI agriculture assistant for farmers.

The selected language is {language}.
Always answer the farmer in {language}.

Your responsibilities:

1. Answer questions about crops, soil, irrigation,
   fertilizers, pests, plant diseases and farming.
2. Use simple, clear language.
3. Explain possible causes before suggesting solutions.
4. Give practical steps farmers can follow.
5. Ask follow-up questions when important details are missing.
6. Do not claim a certain disease diagnosis without evidence.
7. Do not invent pesticide names, doses or guaranteed results.
8. Encourage soil testing before fertilizer recommendations.
9. For severe crop damage, recommend contacting a local
   agricultural extension officer.
10. Politely redirect unrelated questions to agriculture.

Structure useful answers with:
- Possible cause
- Recommended steps
- Precautions

Be friendly, accurate and concise.
"""

    chat_history = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    for message in messages[-6:]:

        role = message["role"]

        if role in ["user", "assistant"]:
            chat_history.append({
                "role": role,
                "content": message["content"]
            })

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=chat_history,
        max_tokens=800,
        temperature=0.5,
        timeout=30
    )

    return response.choices[0].message.content




# ---------------- LOAD ENCODERS ----------------

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


# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="AgriX AI",
    page_icon="🌱",
    layout="wide"
)


# ---------------- SESSION STATE ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "language" not in st.session_state:
    st.session_state.language = "English"


# ---------------- LOGIN PAGE ----------------

def login():

    st.title("🌱 AgriX Login")

    email = st.text_input("Email")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        user = login_user(email, password)

        if user:

            stored_password = user[3]

            if bcrypt.checkpw(
                password.encode(),
                stored_password.encode()
            ):

                st.session_state.logged_in = True
                st.session_state.username = user[1]

                st.success(
                    "Login Successful"
                )

                st.rerun()

            else:
                st.error(
                    "Incorrect Password"
                )

        else:
            st.error(
                "User not found"
            )


# ---------------- REGISTER PAGE ----------------

def register():

    st.title("🌱 Register")

    username = st.text_input(
        "Username"
    )

    email = st.text_input(
        "Email"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    confirm_password = st.text_input(
        "Confirm Password",
        type="password"
    )

    if st.button("Register"):

        if password != confirm_password:

            st.error(
                "Passwords do not match"
            )

        else:

            hashed_password = bcrypt.hashpw(
                password.encode(),
                bcrypt.gensalt()
            ).decode()

            try:

                add_user(
                    username,
                    email,
                    hashed_password
                )

                st.success(
                    "Registration Successful"
                )

            except:
                st.error(
                    "Email already exists"
                )


# ---------------- DASHBOARD ----------------

def dashboard():

    # ---------------- LANGUAGE ----------------

    language_options = [
        "English",
        "Telugu",
        "Hindi"
    ]

    language = st.sidebar.selectbox(
        f"🌐 {translate('language', st.session_state.language)}",
        language_options,
        index=language_options.index(
            st.session_state.language
        )
    )

    st.session_state.language = language

    # ---------------- WELCOME ----------------

    st.title(
        f"🌱 {translate('welcome', language)}, "
        # f"{st.session_state.username}"
    )

    # ---------------- LOGOUT ----------------

    if st.sidebar.button(
        translate("logout", language)
    ):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.language = "English"
        st.session_state.chat_messages = []

        st.rerun()

    # ---------------- SIDEBAR MODULES ----------------

    module_keys = [
        "home",
        "disease",
        "irrigation",
        "fertilizer",
        "yield",
        "market",
        "weather",
        "chatbot",
        "history"
    ]

    module_labels = [
        translate(key, language)
        for key in module_keys
    ]

    selected_label = st.sidebar.selectbox(
        translate("choose_module", language),
        module_labels
    )

    menu = module_keys[
        module_labels.index(selected_label)
    ]

    # =========================================================
    # HOME
    # =========================================================

    if menu == "home":

        st.header(
            f"🏡 {translate('dashboard', language)}"
        )

        st.success(
            translate("login_success", language)
        )

        st.write(
            f"### {translate('available_features', language)}"
        )

        st.write(
            f"✅ {translate('disease', language)}"
        )
        st.write(
            f"✅ {translate('irrigation', language)}"
        )
        st.write(
            f"✅ {translate('fertilizer', language)}"
        )
        st.write(
            f"✅ {translate('yield', language)}"
        )
        st.write(
            f"✅ {translate('market', language)}"
        )
        st.write(
            f"✅ {translate('weather', language)}"
        )
        st.write(
            f"✅ {translate('chatbot', language)}"
        )
        st.write(
            f"✅ {translate('history', language)}"
        )

    # =========================================================
    # DISEASE DETECTION
    # =========================================================

    elif menu == "disease":

        st.header(
            f"🌿 {translate('crop_disease', language)}"
        )
        uploaded_file = st.file_uploader(
            translate("upload_leaf", language),
            type=["jpg", "jpeg", "png"],
            key="disease_image_uploader"
        )

        if uploaded_file is not None:
            from PIL import Image
            import io
            try:
                image_bytes = uploaded_file.getvalue()
                leaf_image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                st.image(
                    leaf_image,
                    caption="Uploaded Leaf Image",
                    width=350
                )

            except Exception as e:
                st.error(f"Unable to display image: {e}")
            if st.button(
                translate("predict_disease", language)
            ):
                with open("temp_image.jpg", "wb") as f:
                    f.write(image_bytes)

                disease, confidence = predict_leaf_disease(
                    "temp_image.jpg"
                )

                st.success(
                    f"{translate('detected_disease', language)}: {disease}"
                )

                st.info(
                    f"{translate('confidence', language)}: {confidence:.2f}%"
                )


        

                save_prediction(
                    st.session_state.username,
                    "Disease Detection",
                    "Leaf image uploaded",
                    f"{disease} ({confidence:.2f}% confidence)"
                )

    # =========================================================
    # IRRIGATION
    # =========================================================

    elif menu == "irrigation":

        st.header(
            f"💧 {translate('irrigation_prediction', language)}"
        )

        region = st.selectbox(
            translate("region", language),
            ["North", "South", "East", "West"]
        )

        soil = st.selectbox(
            translate("soil_type", language),
            [
                "Clay",
                "Loam",
                "Sandy",
                "Silt",
                "Peaty",
                "Chalky"
            ]
        )

        crop = st.selectbox(
            translate("crop", language),
            [
                "Wheat",
                "Rice",
                "Cotton",
                "Soybean",
                "Barley"
            ]
        )

        rainfall = st.number_input(
            translate("rainfall", language),
            min_value=0.0
        )

        temperature = st.number_input(
            translate("temperature", language)
        )

        fertilizer_used = st.selectbox(
            translate("fertilizer_used", language),
            [True, False],
            format_func=lambda x:
                translate("yes", language)
                if x
                else translate("no", language)
        )

        weather = st.selectbox(
            translate("weather_condition", language),
            [
                "Sunny",
                "Cloudy",
                "Rainy"
            ]
        )

        days_to_harvest = st.number_input(
            translate("days_harvest", language),
            min_value=0
        )

        yield_tons = st.number_input(
            translate("yield_hectare", language),
            min_value=0.0
        )

        if st.button(
            translate(
                "predict_irrigation",
                language
            )
        ):

            result = predict_irrigation(
                region,
                soil,
                crop,
                rainfall,
                temperature,
                fertilizer_used,
                weather,
                days_to_harvest,
                yield_tons
            )

            st.success(result)

            input_data = (
                f"Region: {region}, "
                f"Soil: {soil}, "
                f"Crop: {crop}, "
                f"Rainfall: {rainfall} mm, "
                f"Temperature: {temperature} °C, "
                f"Fertilizer Used: {fertilizer_used}, "
                f"Weather: {weather}, "
                f"Days to Harvest: {days_to_harvest}, "
                f"Yield: {yield_tons} tons/hectare"
            )

            save_prediction(
                st.session_state.username,
                "Irrigation Prediction",
                input_data,
                str(result)
            )

    # =========================================================
    # FERTILIZER
    # =========================================================

    elif menu == "fertilizer":

        st.header(
            f"🌾 {translate('fertilizer_recommendation', language)}"
        )

        region = st.selectbox(
            translate("region", language),
            ["North", "South", "East", "West"],
            key="fert_region"
        )

        soil = st.selectbox(
            translate("soil_type", language),
            [
                "Clay",
                "Loam",
                "Sandy",
                "Silt",
                "Peaty",
                "Chalky"
            ],
            key="fert_soil"
        )

        crop = st.selectbox(
            translate("crop", language),
            [
                "Wheat",
                "Rice",
                "Cotton",
                "Soybean",
                "Barley"
            ],
            key="fert_crop"
        )

        rainfall = st.number_input(
            translate("rainfall", language),
            min_value=0.0,
            key="fert_rainfall"
        )

        temperature = st.number_input(
            translate("temperature", language),
            key="fert_temp"
        )

        weather = st.selectbox(
            translate("weather_condition", language),
            [
                "Sunny",
                "Cloudy",
                "Rainy"
            ],
            key="fert_weather"
        )

        days_to_harvest = st.number_input(
            translate("days_harvest", language),
            min_value=0,
            key="fert_days"
        )

        yield_tons = st.number_input(
            translate("yield_hectare", language),
            min_value=0.0,
            key="fert_yield"
        )

        if st.button(
            translate(
                "recommend_fertilizer",
                language
            )
        ):

            result = predict_fertilizer(
                region,
                soil,
                crop,
                rainfall,
                temperature,
                weather,
                days_to_harvest,
                yield_tons
            )

            st.success(result)

            input_data = (
                f"Region: {region}, "
                f"Soil: {soil}, "
                f"Crop: {crop}, "
                f"Rainfall: {rainfall} mm, "
                f"Temperature: {temperature} °C, "
                f"Weather: {weather}, "
                f"Days to Harvest: {days_to_harvest}, "
                f"Yield: {yield_tons} tons/hectare"
            )

            save_prediction(
                st.session_state.username,
                "Fertilizer Recommendation",
                input_data,
                str(result)
            )

    # =========================================================
    # YIELD
    # =========================================================

    elif menu == "yield":

        st.title(
            f"🌾 {translate('yield_prediction', language)}"
        )

        st.write(
            translate(
                "yield_description",
                language
            )
        )

        region = st.selectbox(
            translate("region", language),
            region_encoder.classes_,
            key="yield_region"
        )

        soil = st.selectbox(
            translate("soil_type", language),
            soil_encoder.classes_,
            key="yield_soil"
        )

        crop = st.selectbox(
            translate("crop", language),
            crop_encoder.classes_,
            key="yield_crop"
        )

        rainfall = st.number_input(
            translate("rainfall", language),
            min_value=0.0,
            value=100.0,
            key="yield_rainfall"
        )

        temperature = st.number_input(
            translate("temperature", language),
            min_value=-10.0,
            max_value=60.0,
            value=25.0,
            key="yield_temperature"
        )

        fertilizer_used = st.selectbox(
            translate("fertilizer_used", language),
            [0, 1],
            format_func=lambda x:
                translate("yes", language)
                if x == 1
                else translate("no", language),
            key="yield_fertilizer"
        )

        irrigation_used = st.selectbox(
            translate("irrigation_used", language),
            [0, 1],
            format_func=lambda x:
                translate("yes", language)
                if x == 1
                else translate("no", language),
            key="yield_irrigation"
        )

        weather = st.selectbox(
            translate("weather_condition", language),
            weather_encoder.classes_,
            key="yield_weather"
        )

        days_to_harvest = st.number_input(
            translate("days_harvest", language),
            min_value=1,
            value=100,
            key="yield_days"
        )

        if st.button(
            translate(
                "predict_yield",
                language
            )
        ):

            try:

                result = predict_yield(
                    region,
                    soil,
                    crop,
                    rainfall,
                    temperature,
                    fertilizer_used,
                    irrigation_used,
                    weather,
                    days_to_harvest
                )

                st.success(
                    f"🌾 {translate('predicted_yield', language)}: "
                    f"{result} tons/hectare"
                )

                input_data = (
                    f"Region: {region}, "
                    f"Soil: {soil}, "
                    f"Crop: {crop}, "
                    f"Rainfall: {rainfall} mm, "
                    f"Temperature: {temperature} °C, "
                    f"Fertilizer Used: "
                    f"{'Yes' if fertilizer_used == 1 else 'No'}, "
                    f"Irrigation Used: "
                    f"{'Yes' if irrigation_used == 1 else 'No'}, "
                    f"Weather: {weather}, "
                    f"Days to Harvest: {days_to_harvest}"
                )

                save_prediction(
                    st.session_state.username,
                    "Yield Prediction",
                    input_data,
                    f"{result} tons/hectare"
                )

            except Exception as e:

                st.error(
                    f"{translate('prediction_error', language)}: {e}"
                )

    # =========================================================
    # MARKET FORECASTING
    # =========================================================

    elif menu == "market":

        st.title(
            f"📈 {translate('market_forecasting', language)}"
        )

        st.write(
            translate(
                "market_description",
                language
            )
        )

        market_data = pd.read_csv(
            "datasets/market_prices.csv"
        )

        market_data["Date"] = pd.to_datetime(
            market_data["Date"]
        )

        crops = market_data["Crop"].unique()

        crop = st.selectbox(
            translate("select_crop", language),
            crops
        )

        days = st.slider(
            translate("forecast_days", language),
            1,
            30,
            7
        )

        crop_data = market_data[
            market_data["Crop"] == crop
        ].sort_values("Date")

        prices = crop_data["Price"].tolist()

        st.subheader(
            f"📊 {crop} "
            f"{translate('price_history', language)}"
        )

        chart_data = crop_data.set_index(
            "Date"
        )[["Price"]]

        st.line_chart(chart_data)

        if st.button(
            translate(
                "forecast_market",
                language
            )
        ):

            if len(prices) < 30:

                st.error(
                    translate(
                        "at_least_30",
                        language
                    )
                )

            else:

                predictions = predict_market(
                    prices,
                    days
                )

                st.subheader(
                    f"🔮 {translate('forecasted_prices', language)}"
                )

                for i, price in enumerate(
                    predictions,
                    start=1
                ):

                    st.write(
                        f"Day {i}: ₹{price:.2f}"
                    )

                input_data = (
                    f"Crop: {crop}, "
                    f"Forecast Days: {days}"
                )

                save_prediction(
                    st.session_state.username,
                    "Market Forecasting",
                    input_data,
                    str(predictions)
                )


    # =========================================================
    # WEATHER PREDICTION - LSTM + OPEN-METEO API
    # =========================================================

    elif menu == "weather":

        st.title(
            f"🌦️ {translate('weather_prediction', language)}"
        )

        st.write(
            "Search for any supported city to view "
            "weather predictions using AgriX LSTM "
            "and the Open-Meteo Weather API."
        )

        st.info(
            "AgriX combines recent historical weather "
            "data, LSTM predictions and live API forecasts."
        )

        # -----------------------------------------
        # 1. DYNAMIC CITY SEARCH
        # -----------------------------------------

        st.subheader("📍 Search Your Location")

        city_query = st.text_input(
            "Enter city name",
            placeholder="Example: Kadapa, Hyderabad, Delhi",
            key="weather_city_search"
        )

        if st.button(
            "🔍 Search City",
            key="search_weather_city"
        ):

            if not city_query.strip():

                st.warning("Please enter a city name.")

            else:

                with st.spinner("Searching for city..."):

                    try:

                        results = search_city(city_query)

                        if results:

                            st.session_state.weather_city_results = results

                            st.success(
                                f"Found {len(results)} matching locations."
                            )

                        else:

                            st.session_state.weather_city_results = []

                            st.warning(
                                "No matching city found. "
                                "Please try another city name."
                            )

                    except Exception as e:

                        st.error(
                            f"City search failed: {e}"
                        )

        # -----------------------------------------
        # 2. SELECT MATCHING LOCATION
        # -----------------------------------------

        if "weather_city_results" not in st.session_state:
            st.session_state.weather_city_results = []

        city_results = st.session_state.weather_city_results

        if city_results:

            def city_label(item):

                name = item.get("name", "Unknown")

                admin = item.get("admin1", "")

                country = item.get("country", "")

                parts = [
                    part for part in [name, admin, country]
                    if part
                ]

                return ", ".join(parts)

            city_labels = [
                city_label(item)
                for item in city_results
            ]

            selected_city_label = st.selectbox(
                "Select the correct location",
                city_labels,
                key="weather_selected_city"
            )

            selected_index = city_labels.index(
                selected_city_label
            )

            selected_location = city_results[
                selected_index
            ]

            city = selected_location["name"]

            country = selected_location.get("country")

            latitude = selected_location["latitude"]

            longitude = selected_location["longitude"]

            st.success(
                f"📍 Selected: {city}, {country}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Latitude",
                    round(latitude, 4)
                )

            with col2:

                st.metric(
                    "Longitude",
                    round(longitude, 4)
                )

            # -----------------------------------------
            # 3. FORECAST DAYS
            # -----------------------------------------

            days = st.selectbox(
                "Select forecast period",
                [3, 5, 7],
                index=2,
                key="weather_forecast_days"
            )

            # -----------------------------------------
            # 4. GENERATE WEATHER PREDICTION
            # -----------------------------------------

            if st.button(
                "🔮 Generate Weather Prediction",
                key="generate_weather_prediction"
            ):

                try:

                    with st.spinner(
                        "Fetching historical weather and generating predictions..."
                    ):

                        result = predict_weather(
                            city=city,
                            days=int(days),
                            country=country,
                            location_index=0
                        )

                    # Store prediction results.
                    st.session_state.weather_result = result

                    st.success(
                        "Weather predictions generated successfully!"
                    )

                    # Save prediction history.
                    save_prediction(
                        st.session_state.username,
                        "Weather Prediction",
                        (
                            f"City: {city}, "
                            f"Country: {country}, "
                            f"Forecast Days: {days}"
                        ),
                        (
                            "LSTM Forecast:\n"
                            + result["lstm_forecast"].to_string(index=False)
                            + "\n\nAPI Forecast:\n"
                            + result["api_forecast"].to_string(index=False)
                        )
                    )

                except Exception as e:

                    st.error(
                        f"Weather prediction error: {e}"
                    )

            # -----------------------------------------
            # 5. DISPLAY PREDICTION RESULTS
            # -----------------------------------------

            if "weather_result" in st.session_state:

                result = st.session_state.weather_result

                result_location = result["location"]

                # Prevent showing an earlier city's result
                # after the user selects a different location.
                if (
                    result_location["name"] == city
                    and result_location["country"] == country
                ):

                    lstm_df = result["lstm_forecast"].copy()

                    api_df = result["api_forecast"].copy()

                    historical_df = result["historical"].copy()

                    # ---------------------------------
                    # LOCATION DETAILS
                    # ---------------------------------

                    st.divider()

                    st.header(
                        f"🌍 Weather Forecast: {city}, {country}"
                    )

                    st.caption(
                        f"Coordinates: {latitude}, {longitude}"
                    )

                    # ---------------------------------
                    # HISTORICAL WEATHER
                    # ---------------------------------

                    with st.expander(
                        "📅 View Recent Historical Weather"
                    ):

                        st.write(
                            "Latest completed historical weather records "
                            "used as LSTM input."
                        )

                        st.dataframe(
                            historical_df,
                            use_container_width=True
                        )

                    # ---------------------------------
                    # LSTM PREDICTION
                    # ---------------------------------

                    st.subheader(
                        "🤖 AgriX LSTM Predictions"
                    )

                    st.caption(
                        "These are model-generated predictions based "
                        "on the recent historical weather."
                    )

                    lstm_display = lstm_df.rename(
                        columns={
                            "date": "Date",
                            "temperature_2m_max": "Maximum Temperature (°C)",
                            "temperature_2m_min": "Minimum Temperature (°C)",
                            "rain_sum": "Rainfall (mm)"
                        }
                    )

                    st.dataframe(
                        lstm_display,
                        use_container_width=True,
                        hide_index=True
                    )

                    # ---------------------------------
                    # OPEN-METEO API FORECAST
                    # ---------------------------------

                    st.subheader(
                        "🌤️ Open-Meteo Weather Forecast"
                    )

                    st.caption(
                        "Independent weather forecast obtained "
                        "from the Open-Meteo API."
                    )

                    api_display_columns = [
                        "time",
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_sum",
                        "rain_sum",
                        "precipitation_probability_max",
                        "wind_speed_10m_max",
                        "et0_fao_evapotranspiration"
                    ]

                    api_display = api_df[
                        [
                            col for col in api_display_columns
                            if col in api_df.columns
                        ]
                    ].rename(
                        columns={
                            "time": "Date",
                            "weather_code": "Weather Code",
                            "temperature_2m_max": "Maximum Temperature (°C)",
                            "temperature_2m_min": "Minimum Temperature (°C)",
                            "precipitation_sum": "Precipitation (mm)",
                            "rain_sum": "Rainfall (mm)",
                            "precipitation_probability_max": "Rain Probability (%)",
                            "wind_speed_10m_max": "Wind Speed (km/h)",
                            "et0_fao_evapotranspiration": "Evapotranspiration (mm)"
                        }
                    )

                    st.dataframe(
                        api_display,
                        use_container_width=True,
                        hide_index=True
                    )

                    # ---------------------------------
                    # TEMPERATURE CHART
                    # ---------------------------------

                    st.divider()

                    st.subheader(
                        "🌡️ Temperature Comparison"
                    )

                    lstm_chart = lstm_df[
                        [
                            "date",
                            "temperature_2m_max",
                            "temperature_2m_min"
                        ]
                    ].copy()

                    lstm_chart = lstm_chart.rename(
                        columns={
                            "temperature_2m_max": "LSTM Maximum",
                            "temperature_2m_min": "LSTM Minimum"
                        }
                    )

                    api_chart = api_df[
                        [
                            "time",
                            "temperature_2m_max",
                            "temperature_2m_min"
                        ]
                    ].copy()

                    api_chart = api_chart.rename(
                        columns={
                            "time": "date",
                            "temperature_2m_max": "API Maximum",
                            "temperature_2m_min": "API Minimum"
                        }
                    )

                    comparison = pd.merge(
                        lstm_chart,
                        api_chart,
                        on="date",
                        how="inner"
                    )

                    if not comparison.empty:

                        temperature_chart = comparison.set_index(
                            "date"
                        )[
                            [
                                "LSTM Maximum",
                                "API Maximum",
                                "LSTM Minimum",
                                "API Minimum"
                            ]
                        ]

                        st.line_chart(
                            temperature_chart
                        )

                    else:

                        st.warning(
                            "The LSTM and API forecast dates do not overlap."
                        )

                    # ---------------------------------
                    # RAINFALL CHART
                    # ---------------------------------

                    st.subheader(
                        "🌧️ Rainfall Comparison"
                    )

                    rainfall_comparison = pd.merge(
                        lstm_df[
                            ["date", "rain_sum"]
                        ].rename(
                            columns={
                                "rain_sum": "LSTM Rainfall"
                            }
                        ),

                        api_df[
                            ["time", "rain_sum"]
                        ].rename(
                            columns={
                                "time": "date",
                                "rain_sum": "API Rainfall"
                            }
                        ),

                        on="date",
                        how="inner"
                    )

                    if not rainfall_comparison.empty:

                        rainfall_chart = rainfall_comparison.set_index(
                            "date"
                        )

                        st.bar_chart(
                            rainfall_chart
                        )

                    else:

                        st.warning(
                            "No matching dates available for rainfall comparison."
                        )

                    # ---------------------------------
                    # WEATHER SUMMARY
                    # ---------------------------------

                    st.divider()

                    st.subheader(
                        "📊 Weather Summary"
                    )

                    total_rainfall = float(
                        api_df["rain_sum"].fillna(0).sum()
                    )

                    max_temperature = float(
                        api_df["temperature_2m_max"].max()
                    )

                    min_temperature = float(
                        api_df["temperature_2m_min"].min()
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Maximum Temperature",
                            f"{max_temperature:.1f} °C"
                        )

                    with col2:

                        st.metric(
                            "Minimum Temperature",
                            f"{min_temperature:.1f} °C"
                        )

                    with col3:

                        st.metric(
                            "Total Forecast Rainfall",
                            f"{total_rainfall:.1f} mm"
                        )

                    # ---------------------------------
                    # AGRICULTURAL ADVICE
                    # ---------------------------------

                    st.divider()

                    st.subheader(
                        "🌾 Agricultural Weather Advice"
                    )

                    if total_rainfall > 50:

                        if language == "Telugu":

                            advice = (
                                "🌧️ భారీ వర్షపాతం అంచనా వేయబడింది. "
                                "పొలంలో నీటి పారుదల సదుపాయాలను పరిశీలించండి. "
                                "నేల తేమను బట్టి నీటిపారుదల నిర్ణయించండి."
                            )

                        elif language == "Hindi":

                            advice = (
                                "🌧️ भारी वर्षा का पूर्वानुमान है। "
                                "खेत में जल निकासी की व्यवस्था जाँचें "
                                "और मिट्टी की नमी के अनुसार सिंचाई करें।"
                            )

                        else:

                            advice = (
                                "🌧️ Heavy rainfall is forecast. "
                                "Check field drainage and avoid unnecessary "
                                "irrigation. Monitor soil moisture."
                            )

                        st.warning(advice)

                    elif total_rainfall > 10:

                        if language == "Telugu":

                            advice = (
                                "🌦️ కొంత వర్షపాతం అంచనా ఉంది. "
                                "నేల తేమను పరిశీలించి నీటిపారుదల నిర్ణయించండి."
                            )

                        elif language == "Hindi":

                            advice = (
                                "🌦️ कुछ वर्षा का पूर्वानुमान है। "
                                "सिंचाई से पहले मिट्टी की नमी जाँचें।"
                            )

                        else:

                            advice = (
                                "🌦️ Some rainfall is expected. "
                                "Monitor soil moisture before deciding "
                                "whether irrigation is required."
                            )

                        st.info(advice)

                    else:

                        if language == "Telugu":

                            advice = (
                                "☀️ తక్కువ వర్షపాతం అంచనా ఉంది. "
                                "పంట నీటి అవసరాలను మరియు నేల తేమను పరిశీలించండి."
                            )

                        elif language == "Hindi":

                            advice = (
                                "☀️ कम वर्षा का पूर्वानुमान है। "
                                "मिट्टी की नमी और फसल की पानी की आवश्यकता पर ध्यान दें।"
                            )

                        else:

                            advice = (
                                "☀️ Little rainfall is expected. "
                                "Monitor soil moisture and crop water requirements."
                            )

                        st.info(advice)

                    if max_temperature > 38:

                        st.warning(
                            "🌡️ High temperatures are forecast. "
                            "Monitor crops for heat stress and maintain "
                            "adequate soil moisture."
                        )

                    elif max_temperature > 32:

                        st.info(
                            "🌡️ Warm conditions are forecast. "
                            "Monitor crop water requirements."
                        )

                    else:

                        st.success(
                            "🌱 Forecast temperatures are relatively moderate."
                        )

                else:

                    st.info(
                        "Select Generate Weather Prediction to "
                        "calculate weather for this location."
                    )


    # =========================================================
    # AI CHATBOT
    # =========================================================

    elif menu == "chatbot":

        st.title(
            f"🤖 {translate('agri_assistant', language)}"
        )

        st.write(
            translate(
                "chat_description",
                language
            )
        )

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        if st.button(
            f"🗑️ {translate('clear_chat', language)}"
        ):

            st.session_state.chat_messages = []

            st.rerun()

        for message in st.session_state.chat_messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        question = st.chat_input(
            translate(
                "chat_placeholder",
                language
            )
        )

        if question:

            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            try:

                with st.chat_message("assistant"):

                    with st.spinner(
                        translate(
                            "preparing_answer",
                            language
                        )
                    ):

                        answer = ask_agrix_chatbot(
                            st.session_state.chat_messages,
                            language
                        )

                        st.markdown(answer)

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:
                error_message = str(e)
                if "503" in error_message or "UNAVAILABLE" in error_message:
                    st.warning(
                        "AgriX AI is temporarily busy. "
                        "Please try asking your question again in a few moments."
                    )
                else:
                    st.error(
                        f"{translate('chatbot_error', language)}: {error_message}"
                    )

    # =========================================================
    # PREDICTION HISTORY
    # =========================================================

    elif menu == "history":

        st.header(
            f"📜 {translate('prediction_history', language)}"
        )

        history = get_prediction_history(
            st.session_state.username
        )

        if not history:

            st.info(
                translate(
                    "no_history",
                    language
                )
            )

        else:

            st.write(
                f"{translate('showing_predictions', language)} "
                f"**{st.session_state.username}**"
            )

            for prediction in history:

                prediction_type = prediction[0]
                input_data = prediction[1]
                result = prediction[2]
                created_at = prediction[3]

                with st.expander(
                    f"🔹 {prediction_type} — {created_at}"
                ):

                    st.write(
                        f"**{translate('input', language)}:**"
                    )

                    st.write(input_data)

                    st.write(
                        f"**{translate('result', language)}:**"
                    )

                    st.success(result)

            st.divider()

            if st.button(
                f"🗑️ {translate('clear_history', language)}"
            ):

                clear_prediction_history(
                    st.session_state.username
                )

                st.success(
                    translate(
                        "history_cleared",
                        language
                    )
                )

                st.rerun()


# ---------------- MAIN ----------------

if st.session_state.logged_in:

    dashboard()

else:

    option = st.sidebar.selectbox(
        "Choose Option",
        [
            "Login",
            "Register"
        ]
    )

    if option == "Login":

        login()

    else:

        register()
