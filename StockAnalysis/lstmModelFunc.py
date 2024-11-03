import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from keras import Sequential, optimizers, layers
from tensorflow.keras.optimizers import Adam
from copy import deepcopy

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
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date + datetime.timedelta(days=1):target_date + datetime.timedelta(days=7)]
        if next_week.empty:
            print("Next week is empty, breaking the loop.")
            break
        target_date = next_week.index[0]

        if target_date > last_date:
            print("Target date is beyond the last date, breaking the loop.")
            break

    return pd.DataFrame({'Date': dates, 'X': X, 'Y': Y})

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1]
    X = np.array(middle_matrix.tolist()).reshape((len(dates), -1, 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)

def train_and_plot_lstm(csv_file, first_date_str, last_date_str, window_size=3, learning_rate=0.001, epochs=100):
    df = pd.read_csv(csv_file)
    df = df[["Date", "Close"]]

    df["Date"] = df["Date"].apply(str_to_datetime)
    df.index = df.pop("Date")

    windowed_df = df_to_windowed_df(df, first_date_str, last_date_str, n=window_size)
    dates, X, Y = windowed_df_to_date_X_y(windowed_df)
    print(dates.shape, X.shape, Y.shape)

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

    plt.figure(figsize=(14, 7))
    plt.plot(dates_train, train_predictions, label='Training Predictions')
    plt.plot(dates_train, y_train, label='Training Observations')
    plt.plot(dates_val, val_predictions, label='Validation Predictions')
    plt.plot(dates_val, y_val, label='Validation Observations')
    plt.plot(dates_test, test_predictions, label='Testing Predictions')
    plt.plot(dates_test, y_test, label='Testing Observations')
    plt.legend()
    plt.show()

# Example of calling the function
#train_and_plot_lstm("G:\\Github\\VECTOR\\StockAnalysis\\NVDA_hist.csv", '2023-11-07', '2024-11-01', window_size=3, learning_rate=0.001, epochs=100)