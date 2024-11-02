import yfinance as yf
import os

ticker_symbol = input("Enter the stock ticker symbol: ")
periodInput = input("Enter your desired time period: ")

ticker = yf.Ticker(ticker_symbol)
historical_data = ticker.history(period=periodInput)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, f'{ticker_symbol}_hist.csv')
historical_data.to_csv(csv_file_path)

print(f"Summary of Historical Data for {ticker_symbol}:")
print(historical_data[['Open', 'High', 'Low', 'Close', 'Volume']])
print(f"Historical data saved to {csv_file_path}")

def tickerFetch(tickerFetch):
    ticker = yf.Ticker(tickerFetch)
    historical_data = ticker.history(period="5y")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, f'{tickerFetch}_hist.csv')
    historical_data.to_csv(csv_file_path)
    return historical_data