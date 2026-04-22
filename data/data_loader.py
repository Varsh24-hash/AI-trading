import yfinance as yf
import pandas as pd

tickers = [
"AAPL","MSFT","GOOGL","AMZN","NVDA",
"META","TSLA","JPM","V","WMT"
]

def download_stock_data(ticker):

    data = yf.download(ticker, start="2005-01-01", end="2024-01-01")
    data.reset_index(inplace=True)

    data.to_csv(f"StockProject-main/data/raw/{ticker}.csv", index=False)

    print(f"{ticker} downloaded")


for ticker in tickers:
    download_stock_data(ticker)