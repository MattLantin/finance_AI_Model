import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from entity import Portfolio, Market, StockTradingEnv

# Ensure images directory exists
images_dir = "images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Load data from the CSV file (simulated loading for demonstration)
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

# Simulate model training (mock)
model = PPO("MlpPolicy", env, device="cpu", verbose=1)
model.learn(total_timesteps=10000)

# Generate synthetic data for portfolio values
steps = 500
portfolio_values = np.cumsum(np.random.normal(loc=0.2, scale=1.5, size=steps)) + init_bal

# Portfolio Value Over Time
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Step")
plt.ylabel("Portfolio Value")
plt.savefig(os.path.join(images_dir, "portfolio_value_over_time.png"))
plt.close()

# Synthetic returns for Sharpe calculations
returns = pd.Series(portfolio_values).pct_change().dropna()

# Rolling Sharpe Ratio (Synthetic)
rolling_sharpe = returns.rolling(window=20).mean() / returns.rolling(window=20).std()
plt.plot(rolling_sharpe, label="Rolling Sharpe Ratio (20-step)")
plt.title("Rolling Sharpe Ratio")
plt.xlabel("Step")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.savefig(os.path.join(images_dir, "rolling_sharpe_ratio.png"))
plt.close()

# Probabilistic Sharpe Ratio (PSR) - Synthetic
def probabilistic_sharpe_ratio(mean_return, std_dev, sample_size, benchmark_sharpe):
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).
    """
    sharpe_ratio = mean_return / std_dev
    denominator = np.sqrt(1 - 1 / (4 * sample_size - 4))
    psr = 1 - 0.5 * (1 + np.math.erf((sharpe_ratio - benchmark_sharpe) * denominator / (np.sqrt(2) * std_dev)))
    return psr

# Generate a bell curve for PSR
mean_return = 0.2  # Mock average return
risk = 0.15        # Mock risk (standard deviation)
sample_size = 500  # Mock sample size
benchmark_sharpe_values = np.linspace(-3, 3, 500)
sharpe_ratio = mean_return / risk
psr_bell_curve = np.exp(-((benchmark_sharpe_values - sharpe_ratio) ** 2) / (2 * (risk ** 2)))

# Plot PSR Bell Curve
plt.figure(figsize=(10, 6))
plt.plot(benchmark_sharpe_values, psr_bell_curve, label="PSR Bell Curve")
plt.axvline(x=sharpe_ratio, color='red', linestyle='--', label="Portfolio Sharpe Ratio")
plt.title("Probabilistic Sharpe Ratio (PSR) Bell Curve")
plt.xlabel("Benchmark Sharpe Ratio")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.savefig(os.path.join(images_dir, "psr_bell_curve.png"))
plt.close()
