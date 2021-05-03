import pandas as pd
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("/Users/saif/Downloads/GOOGL.csv", index_col=0, parse_dates=True)
# df["Date"] = df.apply(lambda x: datetime.strptime(x["Date"][0:7], "%d%b%y"), axis=1)

# Autocorrelation

autocorrelation_plot(df.High)
# plt.legend()
# plt.show()

# Rolling Means

# df = df.assign(rs7=df.High.rolling(window=7).std())
# df = df.assign(rm7=df.High.rolling(window=7).mean())
# df = df.assign(rs31=df.High.rolling(window=31).std())
# df = df.assign(rm31=df.High.rolling(window=31).mean())

model = ARIMA(df.High, order=(5, 1, 0))
model_fit = model.fit()
# print(model_fit.summary())

# Integration

residuals = DataFrame(model_fit.resid)
residuals.plot()
# plt.show()

residuals.plot(kind='kde')
# plt.show()

print(residuals.describe())
# Note: the mean is very close to 0 and therefore there is not much bias in the residuals.

'''
plt.plot(df.index, df.High, color='#000000', linestyle='-', label='OG')
plt.plot(df.index, df.rm7, color='#010101', linestyle='-', label='Weekly Rolling Mean')
plt.plot(df.index, df.rm31, color='#010101', linestyle='-', label='Monthly Rolling Mean')
plt.plot(df.index, df.rs7, color='#110011', linestyle='-', label='Weekly Rolling SD')
plt.plot(df.index, df.rs31, color='#000000', linestyle='-', label='Monthly Rolling SD')
'''
