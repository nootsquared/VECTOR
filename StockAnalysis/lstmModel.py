import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

from keras import Sequential, optimizers, layers
from tensorflow.keras.optimizers import Adam

from copy import deepcopy

df = pd.read_csv("G:\\Github\\VECTOR\\StockAnalysis\\NVDA_hist.csv")
df = df[["Date", "Close"]]

def str_to_datetime(s):
    date_part = s.split(" ")[0]
    split = date_part.split("-")
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day).date()

df["Date"] = df["Date"].apply(str_to_datetime)
df.index = df.pop("Date")

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

windowed_df = df_to_windowed_df(df, '2023-11-07', '2024-11-01', n=3)
dates, X, Y = windowed_df_to_date_X_y(windowed_df)
print(dates.shape, X.shape, Y.shape)

q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], Y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], Y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], Y[q_90:]

# plt.plot(dates_train, y_train)
# plt.plot(dates_val, y_val)
# plt.plot(dates_test, y_test)
#
# plt.legend(['Train', 'Validation', 'Test'])
# plt.show()

model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001), #play around with learning_rate
              metrics=['mean_absolute_error'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

train_predictions = model.predict(X_train).flatten()
#
# plt.plot(dates_train, train_predictions)
# plt.plot(dates_train, y_train)
# plt.legend(['Training Predictions', 'Training Observations'])
#
val_predictions = model.predict(X_val).flatten()
#
# plt.plot(dates_val, val_predictions)
# plt.plot(dates_val, y_val)
# plt.legend(['Validation Predictions', 'Validation Observations'])

test_predictions = model.predict(X_test).flatten()

# plt.plot(dates_test, test_predictions)
# plt.plot(dates_test, y_test)
# plt.legend(['Testing Predictions', 'Testing Observations'])

# plt.plot(dates_train, train_predictions)
# plt.plot(dates_train, y_train)
# plt.plot(dates_val, val_predictions)
# plt.plot(dates_val, y_val)
# plt.plot(dates_test, test_predictions)
# plt.plot(dates_test, y_test)
# plt.legend(['Training Predictions',
#             'Training Observations',
#             'Validation Predictions',
#             'Validation Observations',
#             'Testing Predictions',
#             'Testing Observations'])

#you must have past 3 days of REAL data in order to know then next

plt.show()