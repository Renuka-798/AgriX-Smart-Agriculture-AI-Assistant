
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from math import sqrt

# ---------------- CONFIGURATION ----------------

DATASET_PATH = "datasets/crop_yield.csv"
MODEL_DIR = "trained_models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------

print("Loading crop yield dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)

# ---------------- BOOLEAN TO INT ----------------

df["Fertilizer_Used"] = df["Fertilizer_Used"].astype(int)
df["Irrigation_Used"] = df["Irrigation_Used"].astype(int)

# ---------------- ENCODERS ----------------

region_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
weather_encoder = LabelEncoder()

df["Region"] = region_encoder.fit_transform(df["Region"])
df["Soil_Type"] = soil_encoder.fit_transform(df["Soil_Type"])
df["Crop"] = crop_encoder.fit_transform(df["Crop"])
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
        "Irrigation_Used",
        "Weather_Condition",
        "Days_to_Harvest"
    ]
]

y = df["Yield_tons_per_hectare"]

# ---------------- SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ---------------- OPTIMIZED MODEL ----------------

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=18,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.8,
    max_leaf_nodes=5000,
    max_samples=0.7,
    random_state=42,
    n_jobs=2
)

print("\nOptimized Yield Model Training Started...")

model.fit(X_train, y_train)

print("Training Completed!")

# ---------------- EVALUATION ----------------

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = sqrt(mean_squared_error(y_test, predictions))

print("\n--------- MODEL RESULTS ---------")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# ---------------- SAVE NEW MODEL ----------------

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "yield_model_optimized.pkl"),
    compress=3
)

joblib.dump(
    region_encoder,
    os.path.join(MODEL_DIR, "region_encoder_optimized.pkl")
)

joblib.dump(
    soil_encoder,
    os.path.join(MODEL_DIR, "soil_encoder_optimized.pkl")
)

joblib.dump(
    crop_encoder,
    os.path.join(MODEL_DIR, "crop_encoder_optimized.pkl")
)

joblib.dump(
    weather_encoder,
    os.path.join(MODEL_DIR, "weather_encoder_optimized.pkl")
)

print("\nOptimized Yield Model and encoders saved successfully!")
