import pandas as pd

df = pd.read_csv(
    "datasets/crop_yield.csv"
)

print(df["Irrigation_Used"].value_counts())

print("\nTotal rows:")
print(len(df))