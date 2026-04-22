import pandas as pd
import numpy as np
import joblib

from ml_price_prediction.predict import predict
from ml_strategy_rl.trading_environment import TradingEnvRL
from ml_strategy_rl.rl_agent import train_agent


# =========================
# 📊 Load Data
# =========================

ticker = "AAPL"

df = pd.read_csv(f"data/processed/{ticker}_features.csv")

# =========================
# 🤖 Load Model
# =========================

model = joblib.load("models/xgb_model.pkl")

# =========================
# 🔮 Generate Predictions (CORRECT WAY)
# =========================

df_pred = predict(model, df)

# =========================
# 📈 Prepare Environment Inputs
# =========================

prices = df_pred["close"].values
predictions = df_pred["probability_up"].values

print(f"Loaded {len(prices)} data points")
print("Sample prices:", prices[:5])
print("Sample predictions:", predictions[:5])


# =========================
# 🧠 Create Environment
# =========================

env = TradingEnvRL(prices, predictions)


# =========================
# 🚀 Train RL Agent
# =========================

agent = train_agent(env, episodes=20)

print("RL Training Complete ✅")