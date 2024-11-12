import pandas as pd

class MiddleMan:
    def __init__(self, dataframe, mode='knownPrediction'):
        self.dataframe = dataframe
        self.mode = mode
        self.global_dates = []

    def set_global_dates(self, lookback_days, prediction_days):
        latest_date = pd.to_datetime(self.dataframe['date'].max())
        end_date = latest_date - pd.Timedelta(days=prediction_days)
        start_date = end_date - pd.Timedelta(days=lookback_days)
        predict_date = latest_date
        self.global_dates = [start_date, end_date, predict_date]

    def get_global_dates(self):
        return self.global_dates

    def get_data_points(self):
        if self.mode == 'futureExtrapolate':
            return self.dataframe
        elif self.mode == 'knownPrediction':
            if len(self.global_dates) < 3:
                raise ValueError("Not enough global dates set for knownPrediction mode")
            start_date, end_date, predict_date = self.global_dates
            data_points = self.dataframe[(self.dataframe['date'] >= start_date) & (self.dataframe['date'] <= end_date)]
            full_set = self.dataframe[(self.dataframe['date'] > end_date) & (self.dataframe['date'] <= predict_date)]
            return data_points, full_set
        else:
            raise ValueError("Invalid mode")

# Example usage:
# df = pd.read_csv('data.csv')
# middle_man = MiddleMan(df, mode='knownPrediction')
# middle_man.set_global_dates(7, 3)
# data_points, full_set = middle_man.get_data_points()
# print(data_points)
# print(full_set)