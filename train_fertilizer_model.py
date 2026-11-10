import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# ---------------- LOAD DATA ----------------

df = pd.read_csv(
    "datasets/crop_yield.csv"
)

print("Dataset Loaded")


# ---------------- BOOLEAN ----------------

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
        "Weather_Condition",
        "Days_to_Harvest",
        "Yield_tons_per_hectare"
    ]
]


# ---------------- TARGET ----------------

y = df["Fertilizer_Used"]


# ---------------- SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------- MODEL ----------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

print("Training Started...")

model.fit(
    X_train,
    y_train
)


# ---------------- ACCURACY ----------------

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
    "trained_models/fertilizer_model.pkl"
)

print(
    "\nFertilizer Model Trained Successfully"
)