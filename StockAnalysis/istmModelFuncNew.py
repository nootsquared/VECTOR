import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from keras import Sequential, optimizers, layers
from tensorflow.keras.optimizers import Adam
from copy import deepcopy
from SentimentAnalysis.tickerNewsFunc import get_stock_sentiment, get_stock_weights

from randBias import setIntervals, changeWeightCoeff, setWeightCoeff, changeSpreadCoeff, setSpreadCoeff, getRandomNumberFromArray

def str_to_datetime(s):
    date_part = s.split(" ")[0]
    split = date_part.split("-")
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day).date()

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date + datetime.timedelta(days=1):target_date + datetime.timedelta(days=7)]
        if next_week.empty:
            break
        target_date = next_week.index[0]

        if target_date > last_date:
            break

    return pd.DataFrame({'Date': dates, 'X': X, 'Y': Y})

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1]
    X = np.array(middle_matrix.tolist()).reshape((len(dates), -1, 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return rolling_mean, upper_band, lower_band

def train_and_plot_lstm(csv_file, first_date_str, last_date_str, ticker, window_size=3, learning_rate=0.001, epochs=200):
    df = pd.read_csv(csv_file)
    df = df[["Date", "Close"]]

    df["Date"] = df["Date"].apply(str_to_datetime)
    df.index = df.pop("Date")

    df['RSI'] = calculate_rsi(df['Close'])
    df['Rolling Mean'], df['Upper Band'], df['Lower Band'] = calculate_bollinger_bands(df['Close'])

    windowed_df = df_to_windowed_df(df, first_date_str, last_date_str, n=window_size)
    dates, X, Y = windowed_df_to_date_X_y(windowed_df)

    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], Y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], Y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], Y[q_90:]

    model = Sequential([layers.Input((window_size, 1)),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

    train_predictions = model.predict(X_train).flatten()
    val_predictions = model.predict(X_val).flatten()
    test_predictions = model.predict(X_test).flatten()

    changes = df['Close'].diff().dropna().to_numpy()
    changes_sorted = np.sort(changes)

    extrapolated_dates = []
    extrapolated_predictions = []

    localWeightCoeff = 0.0
    localSpreadCoeff = 4

    setWeightCoeff(localWeightCoeff)
    setSpreadCoeff(localSpreadCoeff)

    sentiment = get_stock_sentiment(ticker)
    sentimentWeight = get_stock_weights(sentiment)

    localWeightCoeff += sentimentWeight
    setWeightCoeff(localWeightCoeff)
    
    # For the length of the dates_test array (and for each i in the rage of dates):
        # Calculate the RSI and the Bollinger Bands limits, and other indicators
        # Have a formula in a function that calculates all of these and assigns each a weights value, and calculates the total weight change for the factors
        # Use the total weight change to change the weights using changeWeightCoeff()
        # Generate a random integer between the change array created above
        # Add that (randomly generated change) to the current price and add that to the array of values

    for i in range(len(dates_test)): # add the interval I need inside the dates_test area
        if i < 3:
            next_prediction = test_predictions[i]
        else:
        

