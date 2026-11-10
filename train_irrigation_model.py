import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# ---------------- LOAD DATA ----------------

df = pd.read_csv(
    "datasets/crop_yield.csv"
)

print("Dataset Loaded")


# ---------------- BOOLEAN TO INT ----------------

df["Fertilizer_Used"] = df[
    "Fertilizer_Used"
].astype(int)

df["Irrigation_Used"] = df[
    "Irrigation_Used"
].astype(int)


# ---------------- ENCODERS ----------------

region_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
weather_encoder = LabelEncoder()


df["Region"] = region_encoder.fit_transform(
    df["Region"]
)

df["Soil_Type"] = soil_encoder.fit_transform(
    df["Soil_Type"]
)

df["Crop"] = crop_encoder.fit_transform(
    df["Crop"]
)

df["Weather_Condition"] = weather_encoder.fit_transform(
    df["Weather_Condition"]
)


# ---------------- FEATURES ----------------

X = df[
    [
        "Region",
        "Soil_Type",
        "Crop",
        "Rainfall_mm",
        "Temperature_Celsius",
        "Fertilizer_Used",
        "Weather_Condition",
        "Days_to_Harvest",
        "Yield_tons_per_hectare"
    ]
]

y = df["Irrigation_Used"]


# ---------------- SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------- MODEL ----------------

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

print("Training Started...")

model.fit(
    X_train,
    y_train
)


# ---------------- EVALUATION ----------------

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ---------------- SAVE ----------------

joblib.dump(
    model,
    "trained_models/irrigation_model.pkl"
)

joblib.dump(
    region_encoder,
    "trained_models/region_encoder.pkl"
)

joblib.dump(
    soil_encoder,
    "trained_models/soil_encoder.pkl"
)

joblib.dump(
    crop_encoder,
    "trained_models/crop_encoder.pkl"
)

joblib.dump(
    weather_encoder,
    "trained_models/weather_encoder.pkl"
)

print(
    "\nIrrigation Model Trained Successfully"
)