import pandas as pd
import os
import joblib


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


model = joblib.load("models/xgb_model.pkl")

probs = model.predict_proba(X)[:,1]


signals = []

for p in probs:

    if p > 0.6:
        signals.append("BUY")

    elif p < 0.4:
        signals.append("SELL")

    else:
        signals.append("HOLD")


results = pd.DataFrame({
    "probability_up": probs,
    "signal": signals
})


os.makedirs("signals", exist_ok=True)

results.to_csv("signals/trading_signals.csv", index=False)

print("Trading signals generated.")