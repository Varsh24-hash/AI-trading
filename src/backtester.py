class Backtester:
    def __init__(self, df, initial_capital=100000, transaction_cost=0.001):
        self.df = df
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run(self):
        capital = self.initial_capital
        position = 0
        portfolio_values = []

        for i in range(len(self.df)):
            price = self.df["Close"].iloc[i]
            signal = self.df["Signal"].iloc[i]

            # BUY
            if signal == 1 and position == 0:
                position = capital / price
                capital -= capital * self.transaction_cost

            # SELL
            elif signal == -1 and position > 0:
                capital = position * price
                capital -= capital * self.transaction_cost
                position = 0

            portfolio_value = capital + position * price
            portfolio_values.append(portfolio_value)

        self.df["Portfolio_Value"] = portfolio_values
        return self.df