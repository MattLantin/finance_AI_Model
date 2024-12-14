# Reinforcement Learning: Stock Trading Final Project

## Project Paper
The detailed project write-up is available on Overleaf. Click the link below to view it:

[Reinforcement Learning: Stock Trading Paper](https://www.overleaf.com/1994282728rswdkshhnfbb#398593f)


## Project Overview
This project implements a reinforcement learning (RL) environment for stock trading using the OpenAI Gym API and the Yahoo Finance API. The goal is to train an RL agent to manage a portfolio of stocks by making decisions (buy, sell, or hold) to maximize long-term returns while managing risk. The environment simulates realistic market dynamics, leveraging historical stock price data.

## Key Features
- **Stock Trading Environment**: Simulates a realistic trading environment with real stock data. Tracks portfolio metrics such as balance, stock allocation, and total portfolio value, supporting decisions like buy, sell, and hold.
- **Reinforcement Learning Framework**: Uses the Proximal Policy Optimization (PPO) algorithm from Stable Baselines3. Includes:
  - Action space: Multi-discrete actions for buy, sell, and hold decisions.
  - Observation space: Includes stock prices, portfolio balance, and allocations.
  - Reward structure: Based on portfolio value changes and the Sharpe Ratio.
- **Visualization and Analysis**: Provides insights through portfolio value plots, Sharpe Ratio bar plots, and risk-return scatter plots.

## Installation
### Prerequisites
- Python 3.10
- Install `pyenv`:
  ```bash
  brew install pyenv
  pyenv install 3.10.12
  pyenv global 3.10.12
