
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = "datasets/weather_data.csv"
MODEL_DIR = "trained_models"

SEQUENCE_LENGTH = 30
EPOCHS = 30
BATCH_SIZE = 32

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "rain_sum"
]

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading weather dataset...")

df = pd.read_csv(DATASET_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["city", "date"])

print("\nDataset Loaded Successfully!")

print("Total Rows:", len(df))
print("Total Cities:", df["city"].nunique())
print("\nAvailable Cities:")
print(df["city"].unique())


# ==========================================
# 3. DATA CLEANING
# ==========================================

print("\nCleaning weather dataset...")

required_columns = ["city", "date"] + FEATURES

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

df = df.dropna(subset=required_columns)

df = df.drop_duplicates(
    subset=["city", "date"]
)

for feature in FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

df = df.dropna(subset=FEATURES)

# Remove physically invalid values
df = df[
    (df["temperature_2m_max"] >= df["temperature_2m_min"]) &
    (df["rain_sum"] >= 0)
]

df = df.sort_values(["city", "date"])

print("Rows after cleaning:", len(df))


# ==========================================
# 4. TRAIN / VALIDATION SPLIT
# ==========================================

print("\nPreparing training data...")

# Split each city's timeline chronologically.
# This prevents future observations from
# entering the training set.

train_parts = []
val_parts = []

for city, city_data in df.groupby("city"):

    city_data = city_data.sort_values("date")

    if len(city_data) <= SEQUENCE_LENGTH + 1:
        print(f"Skipping {city}: insufficient records")
        continue

    split_index = int(len(city_data) * 0.8)

    if split_index <= SEQUENCE_LENGTH:
        print(f"Skipping {city}: insufficient training history")
        continue

    train_parts.append(
        city_data.iloc[:split_index]
    )

    val_parts.append(
        city_data.iloc[split_index:]
    )

if not train_parts:
    raise ValueError("No cities have sufficient training data.")

train_df = pd.concat(train_parts)

val_df = pd.concat(val_parts)

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))


# ==========================================
# 5. SCALE DATA
# ==========================================

print("\nScaling weather features...")

scaler = MinMaxScaler()

scaler.fit(train_df[FEATURES].values)

print("Scaler fitted on training data!")


# ==========================================
# 6. CREATE SEQUENCES
# ==========================================

X_train = []
y_train = []

X_val = []
y_val = []

for city in train_df["city"].unique():

    city_train = train_df[
        train_df["city"] == city
    ].sort_values("date")

    city_val = val_df[
        val_df["city"] == city
    ].sort_values("date")

    train_scaled = scaler.transform(
        city_train[FEATURES].values
    )

    val_scaled = scaler.transform(
        city_val[FEATURES].values
    )

    # Training sequences
    for i in range(SEQUENCE_LENGTH, len(train_scaled)):

        X_train.append(
            train_scaled[i-SEQUENCE_LENGTH:i]
        )

        y_train.append(
            train_scaled[i]
        )

    # Validation sequences use the last 30
    # training days as their initial context.
    if len(city_val) > 0:

        combined = np.vstack([
            train_scaled[-SEQUENCE_LENGTH:],
            val_scaled
        ])

        for i in range(SEQUENCE_LENGTH, len(combined)):

            X_val.append(
                combined[i-SEQUENCE_LENGTH:i]
            )

            y_val.append(
                combined[i]
            )


X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = np.array(X_val)
y_val = np.array(y_val)

if len(X_train) == 0 or len(X_val) == 0:
    raise ValueError("Insufficient sequences for training or validation.")

print("\nSequence preparation completed!")

print("Training Input Shape:", X_train.shape)
print("Training Output Shape:", y_train.shape)

print("Validation Input Shape:", X_val.shape)
print("Validation Output Shape:", y_val.shape)


# ==========================================
# 7. BUILD LSTM MODEL
# ==========================================

print("\nBuilding LSTM model...")

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, len(FEATURES))
    )
)

model.add(Dropout(0.2))

model.add(LSTM(32))

model.add(Dense(3))

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

model.summary()


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nTraining Started...")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)


# ==========================================
# 9. SAVE MODEL AND SCALER
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "weather_model.keras"
)

scaler_path = os.path.join(
    MODEL_DIR,
    "weather_scaler.pkl"
)

model.save(model_path)

joblib.dump(
    scaler,
    scaler_path
)

joblib.dump(
    FEATURES,
    os.path.join(MODEL_DIR, "weather_features.pkl")
)

print("\n===================================")
print("AGRI X WEATHER MODEL TRAINED!")
print("===================================")

print("Model:", model_path)
print("Scaler:", scaler_path)

print("\nTraining Completed Successfully!")
