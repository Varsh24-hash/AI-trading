import pandas as pd
import ta
import os

# folder paths
raw_folder = "StockProject-main/data/raw"
processed_folder = "StockProject-main/data/processed"

# get all csv files in raw folder
files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]

for file in files:

    filepath = os.path.join(raw_folder, file)

    data = pd.read_csv(filepath)

    # convert numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # feature engineering
    data["return_1d"] = data["Close"].pct_change()

    data["MA_10"] = data["Close"].rolling(10).mean()
    data["MA_50"] = data["Close"].rolling(50).mean()

    data["volatility"] = data["return_1d"].rolling(10).std()

    data["volume_change"] = data["Volume"].pct_change()

    data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"]).rsi()

    # create target
    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    data = data.dropna()

    # create output filename
    ticker = file.replace(".csv", "")
    output_file = f"{processed_folder}/{ticker}_features.csv"

    data.to_csv(output_file, index=False)

    print(f"{ticker} feature engineering completed")

print("All feature engineering completed.")