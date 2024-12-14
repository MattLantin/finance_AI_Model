import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gymnasium.spaces import Discrete, Box
import yfinance as yf

# Download stock data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2015-01-01'
end_date = '2016-01-01'
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
data.to_csv('stock_data.csv')
init_bal = 10000

class Portfolio:
    BUY = 1
    SELL = -1
    HOLD = 0

    def __init__(self, stocks, initial_balance=10000):
        self.stocks = stocks
        self.investment = {stock: 0 for stock in stocks}
        self.investment_shares = {stock: 0 for stock in stocks}
        self.total_investment = 0
        self.investment_percentage = {stock: 0 for stock in stocks}
        self.balance = initial_balance
        self.initial_balance = initial_balance

    def reset(self):
        self.investment = {stock: 0 for stock in self.stocks}
        self.total_investment = 0
        self.investment_percentage = {stock: 0 for stock in self.stocks}
        self.balance = self.initial_balance

    def allowed_actions(self):
        return [Portfolio.BUY, Portfolio.SELL, Portfolio.HOLD]

    def take_action(self, action, stock, amount, price):
        if action == Portfolio.BUY:
            self.buy(stock, amount, price)
        elif action == Portfolio.SELL:
            self.sell(stock, amount, price)
        else:
            self.hold(stock)

    def buy(self, stock, amount, price):
        if self.balance >= amount * price:
            self.investment[stock] += amount
            self.total_investment += amount
            self.investment_shares[stock] += amount
            self.balance -= amount * price
        self.update_investment_percentage()

    def sell(self, stock, amount, price):
        if self.investment[stock] >= amount:
            self.investment[stock] -= amount
            self.total_investment -= amount
            self.investment_shares[stock] -= amount
            self.balance += amount * price
        self.update_investment_percentage()

    def update_investment_percentage(self):
        for stock in self.stocks:
            if self.total_investment > 0:
                self.investment_percentage[stock] = self.investment[stock] / self.total_investment

    def hold(self, stock):
        return self.investment[stock]

    def report(self):
        print(f"Total investment: {self.total_investment}")
        print(f"Total balance: {self.balance}")
        print(f"Total value: {self.total_investment + self.balance}")
        print(f"Investment in each stock: {self.investment}")
        print(f"Number of shares in each stock: {self.investment_shares}")


class Market:
    def __init__(self, stock_data, window_size):
        self.stock_data = stock_data
        self.window_size = window_size
        self.current_step = 0

    def get_price_at_step(self, stock, step):
        return self.stock_data.iloc[step][stock]

    def get_state(self):
        state = [self.get_price_at_step(stock, self.current_step) for stock in self.stock_data.columns]
        return np.array(state)

    def get_price(self, stock):
        return self.get_price_at_step(stock, self.current_step)

    def reset(self):
        self.current_step = 0


class StockTradingEnv(gym.Env):
    def __init__(self, market: Market, portfolio: Portfolio, initial_balance=10000):
        self.market = market
        self.portfolio = portfolio
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.portfolio_value = 0
        self.action_space = Discrete(len(self.portfolio.stocks) * 3)
        self.observation_space = Box(low=0, high=np.inf, shape=(len(self.portfolio.stocks),), dtype=np.float32)
        self.current_step = 0
        self.done = False

    def curr_portfolio_balance(self):
        total_value = 0
        for stock in self.portfolio.stocks:
            total_value += self.market.get_price(stock) * self.portfolio.investment_shares[stock]
        return total_value

    def reset(self, seed=None, options=None):
    # Set the seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        self.current_balance = self.initial_balance
        self.current_step = 0
        self.done = False
        self.portfolio.reset()
        self.market.reset()
        self.portfolio_value = self.curr_portfolio_balance() + self.current_balance

        state = self.market.get_state()
        info = {}  # Return an empty info dictionary
        return state, info



    def step(self, action):
        stock_idx = action // 3
        action_type = action % 3
        stock = self.portfolio.stocks[stock_idx]
        price = self.market.get_price(stock)
        amount = 1
        if action_type == Portfolio.BUY:
            self.portfolio.take_action(Portfolio.BUY, stock, amount, price)
        elif action_type == Portfolio.SELL:
            self.portfolio.take_action(Portfolio.SELL, stock, amount, price)
        else:
            self.portfolio.take_action(Portfolio.HOLD, stock, amount, price)

        prev_balance = self.curr_portfolio_balance() + self.current_balance
        self.portfolio_value = self.curr_portfolio_balance() + self.current_balance
        reward = self.portfolio_value - prev_balance

        self.current_step += 1
        done = self.current_step >= len(self.market.stock_data) - 1

        return self.market.get_state(), reward, done, {}

# Initialize portfolio and market
portfolio = Portfolio(tickers, init_bal)
market = Market(data, window_size=10)
env = StockTradingEnv(market, portfolio)

# Define and train the model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Simulate trading
def simulate_trading(env, model):
    state = env.reset()
    portfolio_values = []
    done = False
    while not done:
        action, _ = model.predict(state)
        state, _, done, _ = env.step(action)
        portfolio_values.append(env.curr_portfolio_balance())
    return portfolio_values

portfolio_values = simulate_trading(env, model)

# Save portfolio value plot
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Step")
plt.ylabel("Portfolio Value")
plt.savefig("portfolio_value_over_time.png")
plt.close()
