from math import sqrt

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

class Analyzer:
    def __init__(self, pathToData: str) -> None:
        super().__init__()
        self.df = pd.read_csv(pathToData, index_col=0, parse_dates=True)
        self.model = ARIMA(self.df.High, order=(5, 1, 0))
        self.model_fit = self.model.fit()

    def __str__(self) -> str:
        return str(self.model_fit.summary())

    def get_residuals(self):
        residuals = DataFrame(self.model_fit.resid)
        return residuals.describe()

    def split_data(self, split_proportion=0.7):
        data = self.df.High.values
        size = int(len(data) * split_proportion)
        train, test = data[0:size], data[size:len(data)]
        return [train, test]

    def train_and_predict(self, train_data, test_data):
        history = [x for x in train_data]
        predictions = list()
        for t in range(len(test_data)):
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            y_hat = output[0]
            predictions.append(y_hat)
            obs = test_data[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (y_hat, obs))
        rmse = sqrt(mean_squared_error(test_data, predictions))
        print('Test RMSE: %.3f' % rmse)


if __name__ == '__main__':
    analyzer = Analyzer('~/Desktop/GOOGL.csv')
    [train, test] = analyzer.split_data()
    analyzer.train_and_predict(train, test)
