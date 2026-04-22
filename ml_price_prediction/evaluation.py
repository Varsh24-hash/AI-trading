import pandas as pd
import os
import joblib

from sklearn.metrics import accuracy_score


processed_folder = "data/processed"

files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

all_data = []

for file in files:
    filepath = os.path.join(processed_folder, file)
    df = pd.read_csv(filepath)

    ticker = file.split("_")[0]
    df["ticker"] = ticker

    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)


features = [
    "return_1d",
    "MA_10",
    "MA_50",
    "volatility",
    "volume_change",
    "RSI"
]

X = data[features]
y = data["target"]


split = int(len(data) * 0.8)

X_test = X[split:]
y_test = y[split:]


models = {
    "Logistic Regression": joblib.load("models/logistic_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "XGBoost": joblib.load("models/xgb_model.pkl")
}


for name, model in models.items():

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(name, "Accuracy:", acc)