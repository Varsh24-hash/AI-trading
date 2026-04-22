import pandas as pd
import numpy as np
import joblib

from ml_price_prediction.predict import predict
from ml_strategy_rl.trading_environment import TradingEnvRL
from ml_strategy_rl.rl_agent import train_agent


# =========================
# 📊 Load REAL data
# =========================

def load_data():
    df = pd.read_csv("data/processed/AAPL_features.csv")
    model = joblib.load("models/xgb_model.pkl")

    df_pred = predict(model, df)

    prices = df_pred["close"].values
    predictions = df_pred["probability_up"].values

    return prices, predictions


# =========================
# 🚀 Run experiment
# =========================

def run_experiment(episodes):

    prices, predictions = load_data()

    env = TradingEnvRL(prices, predictions)

    agent = train_agent(env, episodes=episodes)

    # Final portfolio value
    final_value = env.cash + env.shares * prices[-1]

    return final_value


# =========================
# 🔍 Hyperparameter tuning
# =========================

episode_options = [10, 20, 50]

best_score = -float("inf")
best_param = None

for ep in episode_options:

    score = run_experiment(ep)

    print(f"Episodes={ep}, Final Value={score}")

    if score > best_score:
        best_score = score
        best_param = ep


print("\n🔥 BEST PARAMETER:")
print(f"Episodes={best_param}")
print(f"Best Portfolio Value={best_score}")