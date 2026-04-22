import numpy as np
from portfolio.position_sizing import allocate_cash, calculate_shares
from risk_management.risk_controls import apply_stop_loss, apply_position_limit


class TradingEnvRL:

    def __init__(self, prices, predictions, initial_cash=10000):
        self.prices = prices
        self.predictions = predictions
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0
        self.current_step = 1
        return self._get_state()

    def _get_state(self):
        price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1]

        # returns
        return_1 = (price - prev_price) / prev_price

        # moving average
        if self.current_step >= 3:
            ma = np.mean(self.prices[self.current_step-3:self.current_step])
        else:
            ma = price

        prediction = self.predictions[self.current_step]

        # normalized state
        state = np.array([
            price / 1000,
            return_1,
            ma / 1000,
            prediction,
            self.cash / 10000,
            self.shares / 100
        ])

        return state

    def step(self, action):

        price = self.prices[self.current_step]
        prev_value = self.cash + self.shares * price
        cost = 0.001 * price

        # =========================
        # BUY
        # =========================
        if action == 1:
            prediction = self.predictions[self.current_step]

            cash_to_invest = allocate_cash(self.cash, prediction)
            num_shares = calculate_shares(cash_to_invest, price)

            if num_shares > 0 and self.cash >= num_shares * (price + cost):
                self.shares += num_shares
                self.cash -= num_shares * (price + cost)

        # =========================
        # SELL
        # =========================
        elif action == -1:
            if self.shares > 0:
                self.cash += self.shares * (price - cost)
                self.shares = 0

        # HOLD → do nothing

        # =========================
        # MOVE FORWARD
        # =========================
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        if done:
            return self._get_state(), 0, done, {}

        next_price = self.prices[self.current_step]
        new_value = self.cash + self.shares * next_price

        # =========================
        # REWARD
        # =========================
        reward = new_value - prev_value

        # holding penalty
        reward -= 0.1 * self.shares

        # risk controls
        if apply_stop_loss(self, price):
            reward -= 5

        reward += apply_position_limit(self)

        return self._get_state(), reward, done, {}