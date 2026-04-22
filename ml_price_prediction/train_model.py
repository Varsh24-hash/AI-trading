import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


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

print("Dataset shape:", data.shape)


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

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


# -------------------------
# Logistic Regression
# -------------------------

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# -------------------------
# Random Forest
# -------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

rf_model.fit(X_train, y_train)


# -------------------------
# XGBoost
# -------------------------

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

xgb_model.fit(X_train, y_train)


os.makedirs("models", exist_ok=True)

joblib.dump(log_model, "models/logistic_model.pkl")
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")

print("All models trained and saved.")