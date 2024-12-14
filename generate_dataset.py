import yfinance as yf
import pandas as pd

def generate_stock_data(tickers, start_date, end_date, filename="stock_data.csv"):
    """
    Download stock data using yfinance and save it to a CSV file.

    Args:
        tickers (list): List of stock tickers to download.
        start_date (str): Start date for the data in YYYY-MM-DD format.
        end_date (str): End date for the data in YYYY-MM-DD format.
        filename (str): Name of the file to save the data.
    """
    print(f"Downloading data for: {tickers} from {start_date} to {end_date}")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data.to_csv(filename)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Define stock tickers, date range, and output file name
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start_date = '2015-01-01'
    end_date = '2016-01-01'
    filename = "stock_data.csv"

    generate_stock_data(tickers, start_date, end_date, filename)
