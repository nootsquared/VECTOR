import StockAnalysis.tickerFetch as tkf
import pandas as pd
import os

#TODO:
#1. Check how random set bias is being calculated and make sure it is modular so different things can add on more weight to pos or neg random scale
#2. Check if boll bands are calculated correctly for extrapolated data
#3. Check if RSI and sentiment are being used correctly on the random bias calculation
#4. When you select 1m and 5d, it says insufficient data (Might only be looking at the date for the previous 3 calculations)
#5. When you select 15m for AMZN, it has a flat line - not sure?

tickerInput = input("ENTER YOUR DESIRED TICKER SYMBOL: ")
possible_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
disabled_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']

while True:
    print(f"Possible interval values: {possible_intervals}")
    intervalInput = input("ENTER DESIRED INTERVAL: ")
    if intervalInput in disabled_intervals:
        print("Temporarily these options are disabled. Please select a different interval.")
    else:
        break

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

timeFrameInput = input("ENTER YOUR DESIRED TIME FRAME: ")

tkf.tickerFetch(tickerInput, timeFrameInput, intervalInput)

csv_file_path = f'{tickerInput}_hist.csv'
csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "StockAnalysis", csv_file_path)
df = pd.read_csv(csv_file_path)

if df.columns[0] == 'Datetime':
    df.rename(columns={'Datetime': 'Date'}, inplace=True)
    df.to_csv(csv_file_path, index=False)

df = pd.read_csv(csv_file_path)

if len(df) < 3:
    raise ValueError("CSV file does not contain enough entries to determine the start and end dates.")

start_date = df.iloc[3]['Date']
end_date = df.iloc[-1]['Date']

if df.empty or len(df) < 3:
    raise ValueError("Insufficient data fetched for the given interval and time frame.")

import StockAnalysis.istmModelFuncNew as lmf
lmf.train_and_plot_lstm(f"G:\\Github\\VECTOR\\StockAnalysis\\{tickerInput}_hist.csv", start_date, end_date, window_size=3, learning_rate=0.001, epochs=200, ticker=tickerInput)