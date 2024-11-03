import StockAnalysis.tickerFetch as tkf
import pandas as pd
import os

tickerInput = input("ticker: ")
possible_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
print(f"Possible interval values: {possible_intervals}")
intervalInput = input("interval: ")

interval_timeframes = {
    '1m': '1d - 7d',
    '2m': '1d - 60d',
    '5m': '1d - 60d',
    '15m': '1d - 60d',
    '30m': '1d - 60d',
    '60m': '1d - 60d',
    '90m': '1d - 60d',
    '1h': '1d - 60d',
    '1d': '1mo - max',
    '5d': '1mo - max',
    '1wk': '1mo - max',
    '1mo': '3mo - max',
    '3mo': '6mo - max'
}

if intervalInput in interval_timeframes:
    print(f"Possible time frame range for interval '{intervalInput}': {interval_timeframes[intervalInput]}")
else:
    print("Invalid interval selected.")

timeFrameInput = input("time frame: ")

tkf.tickerFetch(tickerInput, timeFrameInput, intervalInput)

# Change the first column head from 'Datetime' to 'Date' if it is 'Datetime'
csv_file_path = f'{tickerInput}_hist.csv'
csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "StockAnalysis", csv_file_path)
df = pd.read_csv(csv_file_path)

if df.columns[0] == 'Datetime':
    df.rename(columns={'Datetime': 'Date'}, inplace=True)
    df.to_csv(csv_file_path, index=False)

# Reload the CSV file after renaming the column
df = pd.read_csv(csv_file_path)

if len(df) < 3:
    raise ValueError("CSV file does not contain enough entries to determine the start and end dates.")

start_date = df.iloc[3]['Date']
end_date = df.iloc[-1]['Date']

# Import the function here to avoid circular import
import StockAnalysis.lstmModelFunc as lmf
lmf.train_and_plot_lstm(f"G:\\Github\\VECTOR\\StockAnalysis\\{tickerInput}_hist.csv", start_date, end_date, window_size=3, learning_rate=0.001, epochs=500)