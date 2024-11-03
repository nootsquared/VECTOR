import yfinance as yf
import os

def tickerFetch(tickerFetch, periodI, interval):
    ticker = yf.Ticker(tickerFetch)
    historical_data = ticker.history(period=periodI, interval=interval)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, f'{tickerFetch}_hist.csv')
    historical_data.to_csv(csv_file_path)