import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from keras import Sequential, optimizers, layers
from tensorflow.keras.optimizers import Adam
from copy import deepcopy
from SentimentAnalysis.tickerNewsFunc import get_stock_sentiment

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

def train_and_plot_lstm(csv_file, first_date_str, last_date_str, ticker, window_size=3, learning_rate=0.001, epochs=100, sentiment_multiplier=1):
    df = pd.read_csv(csv_file)
    df = df[["Date", "Close"]]

    df["Date"] = df["Date"].apply(str_to_datetime)
    df.index = df.pop("Date")

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

    plt.figure(figsize=(14, 7))
    plt.plot(dates_train, train_predictions, label='Training Predictions')
    plt.plot(dates_train, y_train, label='Training Observations')
    plt.plot(dates_val, val_predictions, label='Validation Predictions')
    plt.plot(dates_val, y_val, label='Validation Observations')
    plt.plot(dates_test, test_predictions, label='Testing Predictions')
    plt.plot(dates_test, y_test, label='Testing Observations')

    sentiment_score = get_stock_sentiment(ticker)

    historical_changes = np.diff(df['Close'])
    std_change = np.std(historical_changes)
    max_change = np.max(np.abs(historical_changes))

    extrapolated_dates = []
    extrapolated_predictions = []

    current_window = X_test[:1]

    for i in range(len(dates_test)):
        if i < 3:
            next_prediction = test_predictions[i]
        else:
            next_prediction = model.predict(current_window).flatten()[0]

            change = np.random.normal(0, std_change)
            change = np.clip(change, -max_change, max_change)
            next_prediction += change

        extrapolated_dates.append(dates_test[i])
        extrapolated_predictions.append(next_prediction)

        next_prediction_reshaped = np.array([[next_prediction]]).reshape((1, 1, 1))
        new_window = np.append(current_window[:, 1:], next_prediction_reshaped, axis=1)
        current_window = new_window

    plt.plot(extrapolated_dates, extrapolated_predictions, label='Extrapolated Predictions', linestyle='dashed')

    predicted_dates = []
    predicted_values = []

    for i in range(len(extrapolated_predictions) - window_size):
        real_window = np.array(extrapolated_predictions[i:i + window_size]).reshape((1, window_size, 1))
        next_predicted_value = model.predict(real_window).flatten()[0]
        predicted_dates.append(extrapolated_dates[i + window_size])
        predicted_values.append(next_predicted_value)

    plt.plot(predicted_dates, predicted_values, label='Predicted from Random Values', linestyle='solid')
    plt.legend()
    plt.title(f"Sentiment: {sentiment_score}")
    plt.show()

