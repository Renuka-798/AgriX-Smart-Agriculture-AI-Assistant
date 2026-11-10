import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ---------------- LOAD DATA ----------------

df = pd.read_csv(
    "datasets/market_prices.csv"
)

print("Dataset Loaded")


# ---------------- DATE ----------------

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df.sort_values(
    "Date"
)


# ---------------- SELECT CROP ----------------

crop = "Rice"

crop_data = df[
    df["Crop"] == crop
].copy()


prices = crop_data[
    "Price"
].values.reshape(-1, 1)


# ---------------- SCALING ----------------

scaler = MinMaxScaler()

scaled_prices = scaler.fit_transform(
    prices
)


# ---------------- CREATE SEQUENCES ----------------

sequence_length = 30

X = []
y = []

for i in range(
    sequence_length,
    len(scaled_prices)
):

    X.append(
        scaled_prices[
            i-sequence_length:i
        ]
    )

    y.append(
        scaled_prices[i]
    )


X = np.array(X)
y = np.array(y)


# ---------------- MODEL ----------------

model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(
            X.shape[1],
            1
        )
    )
)

model.add(
    LSTM(50)
)

model.add(
    Dense(1)
)


model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


print("Training Started...")


model.fit(
    X,
    y,
    epochs=20,
    batch_size=32,
    verbose=1
)


# ---------------- SAVE ----------------

model.save(
    "trained_models/market_model.keras"
)

joblib.dump(
    scaler,
    "trained_models/market_scaler.pkl"
)

print(
    "\nMarket Model Trained Successfully"
)