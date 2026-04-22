import yfinance as yf
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, ticker, start="2018-01-01", end="2023-12-31"):
        self.ticker = ticker
        self.start = start
        self.end = end

    def fetch_data(self):
      df = yf.download(self.ticker, start=self.start, end=self.end)

      # Flatten MultiIndex columns
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      df.dropna(inplace=True)
      return df

    def add_returns(self, df):
        df["Simple_Return"] = df["Close"].pct_change()
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df

    def add_volatility(self, df, window=20):
        df["Rolling_Volatility"] = df["Log_Return"].rolling(window).std()
        return df