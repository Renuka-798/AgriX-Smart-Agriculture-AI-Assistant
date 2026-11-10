import joblib
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("trained_models/market_model.keras")
scaler = joblib.load("trained_models/market_scaler.pkl")


def predict_market(prices, days=7):

    prices = np.array(prices).reshape(-1, 1)

    scaled_prices = scaler.transform(prices)

    sequence = scaled_prices[-30:]

    predictions = []

    for _ in range(days):

        input_data = np.array([sequence])

        prediction = model.predict(input_data, verbose=0)[0][0]

        predictions.append(prediction)

        sequence = np.append(
            sequence[1:],
            [[prediction]],
            axis=0
        )

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions.flatten()


if __name__ == "__main__":

    test_prices = [
        2000, 2020, 1980, 2050, 2100,
        2070, 2150, 2200, 2180, 2250,
        2300, 2280, 2350, 2400, 2380,
        2450, 2500, 2480, 2550, 2600,
        2580, 2650, 2700, 2680, 2750,
        2800, 2780, 2850, 2900, 2950
    ]

    predictions = predict_market(test_prices, 7)

    print("\n🔮 Predicted Market Prices")

    for i, price in enumerate(predictions, start=1):
        print(f"Day {i}: ₹{price:.2f}")