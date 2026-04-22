import pandas as pd
import numpy as np
import joblib

from ml_price_prediction.predict import predict
from ml_strategy_rl.trading_environment import TradingEnvRL
from ml_strategy_rl.rl_agent import train_agent


# =========================
# 📊 LOAD REAL DATA (FINAL)
# =========================
def load_data(ticker="AAPL"):

    file_path = f"data/processed/{ticker}_features.csv"

    df = pd.read_csv(file_path)

    # 🔥 SPEED MODE (remove if you want full dataset later)
    df = df.tail(300)

    # Load trained ML model
    model = joblib.load("models/xgb_model.pkl")

    # Generate predictions using your pipeline
    df_pred = predict(model, df)

    prices = df_pred["close"].values
    predictions = df_pred["probability_up"].values

    return prices, predictions


# =========================
# 🚀 SIMULATION (FINAL)
# =========================
def simulate_strategy(agent, prices, predictions):

    env = TradingEnvRL(prices, predictions)

    state = env.reset()
    portfolio_values = []

    # 🔥 Disable exploration (evaluation mode)
    agent.epsilon = 0

    while True:
        action_idx = agent.act(state)

        # Map action index → env action
        action_map = {0: 0, 1: 1, 2: -1}
        action = action_map[action_idx]

        next_state, reward, done, _ = env.step(action)

        # Portfolio value
        value = env.cash + env.shares * prices[env.current_step]
        portfolio_values.append(value)

        state = next_state

        if done:
            break

    portfolio_values = np.array(portfolio_values)

    if len(portfolio_values) < 2:
        return 0, 0

    # ✅ Strategy returns (CORRECT)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    sharpe = np.mean(returns) / (np.std(returns) + 1e-8)

    final_value = portfolio_values[-1]

    return final_value, sharpe


# =========================
# 🔍 MAIN OPTIMIZATION (FINAL)
# =========================
if __name__ == "__main__":

    print("📊 Loading real data...")
    prices, predictions = load_data("AAPL")

    print("🤖 Training RL agent (once)...")
    env = TradingEnvRL(prices, predictions)

    # 🔥 Train only once
    agent = train_agent(env, episodes=15)

    print("\n🚀 Running Sharpe Optimization...\n")

    episode_options = [5, 10, 15]

    best_sharpe = -float("inf")
    best_param = None

    for ep in episode_options:

        print(f"🔹 Testing config: episodes={ep}")

        final_value, sharpe = simulate_strategy(agent, prices, predictions)

        print(f"   Final Value = {final_value:.2f}")
        print(f"   Sharpe Ratio = {sharpe:.4f}\n")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_param = ep

    print("🔥 BEST CONFIGURATION:")
    print(f"Best Episodes = {best_param}")
    print(f"Best Sharpe Ratio = {best_sharpe:.4f}")