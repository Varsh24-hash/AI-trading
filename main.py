from src.data_loader import DataLoader
from src.features import add_moving_averages, add_rsi, add_macd
import matplotlib.pyplot as plt
from src.backtester import Backtester



loader = DataLoader("AAPL")
df = loader.fetch_data()
df = loader.add_returns(df)
df = loader.add_volatility(df)

df = add_moving_averages(df)
df = add_rsi(df)
df = add_macd(df)

df.dropna(inplace=True)

print(df.columns)

df["Signal"] = 0
df.loc[df["SMA_20"] > df["SMA_50"], "Signal"] = 1
df.loc[df["SMA_20"] < df["SMA_50"], "Signal"] = -1

backtest = Backtester(df)
df = backtest.run()

plt.figure(figsize=(10,5))
plt.plot(df["Portfolio_Value"])
plt.title("Portfolio Value Over Time")
plt.show()
# plt.figure(figsize=(10,5))
# plt.plot(df["Close"], label="Close")
# plt.plot(df["SMA_20"], label="SMA 20")
# plt.plot(df["SMA_50"], label="SMA 50")
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,4))
# plt.plot(df["RSI"])
# plt.axhline(70, color='r')
# plt.axhline(30, color='g')
# plt.title("RSI")
# plt.show()