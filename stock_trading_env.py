import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from entity import Portfolio, Market, StockTradingEnv

# Ensure images directory exists
images_dir = "images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Load data from the CSV file
filename = "stock_data.csv"
try:
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    print(f"Successfully loaded data from {filename}.")
except FileNotFoundError:
    print(f"Error: {filename} not found. Please run generate_dataset.py first to create the dataset.")
    exit()

# Initialize the portfolio and market
tickers = data.columns.tolist()
init_bal = 10000
portfolio = Portfolio(tickers, init_bal)
market = Market(data, window_size=10)
env = StockTradingEnv(market, portfolio)

# Define and train the model
model = PPO("MlpPolicy", env, device="cpu", verbose=1)
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

# Run simulation
portfolio_values = simulate_trading(env, model)

# Portfolio Value Over Time
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Step")
plt.ylabel("Portfolio Value")
plt.savefig(os.path.join(images_dir, "portfolio_value_over_time.png"))
plt.close()

# Calculate percentage returns
returns = pd.Series(portfolio_values).pct_change().dropna()

# Risk vs Return Scatter Plot
mean_return = returns.mean()
risk = returns.std()
plt.scatter(mean_return, risk, color='blue', label="Portfolio")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.title("Risk vs Return")
plt.xlabel("Average Return")
plt.ylabel("Risk (Standard Deviation)")
plt.legend()
plt.savefig(os.path.join(images_dir, "risk_vs_return.png"))
plt.close()

# Cumulative Returns Chart
cumulative_returns = (1 + returns).cumprod()
plt.plot(cumulative_returns, label="Cumulative Returns")
plt.title("Cumulative Returns Over Time")
plt.xlabel("Step")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.savefig(os.path.join(images_dir, "cumulative_returns.png"))
plt.close()


# Rolling Sharpe Ratio Chart
rolling_sharpe = returns.rolling(window=20).mean() / returns.rolling(window=20).std()
plt.plot(rolling_sharpe, label="Rolling Sharpe Ratio (20-step)")
plt.title("Rolling Sharpe Ratio")
plt.xlabel("Step")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.savefig(os.path.join(images_dir, "rolling_sharpe_ratio.png"))
plt.close()
