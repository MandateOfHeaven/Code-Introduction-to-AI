import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa as tsa
import numpy as np
import matplotlib.pyplot as plt
import arch

"Loading data and plot the series"
Path = '/home/daiyi/Desktop/TSA/TSA_10/Data_10.txt'
# Path = '/home/daiyi/Desktop/TSA/TSA_5/Data_5.txt'
# Path = '/home/daiyi/Desktop/TSA/TSA_Data_2/Data_2.txt'

Open = []
Close = []
Ratio = []
Date = []
for OriginalLine in open(Path):
    Splited = OriginalLine.split()
    Open.append(float(Splited[2]))
    Close.append(float(Splited[5]))
    Ratio.append(float(Splited[6]))
    Date.append(str(Splited[1]))

Open.reverse()
Close.reverse()
Ratio.reverse()
Date.reverse()


"Indexing with Time-series Data"
Open = pd.Series(Open)
Close = pd.Series(Close)
Ratio = pd.Series(Ratio)
Open.index = pd.Index(Date)
Ratio.index = pd.Index(Date)
# Open.plot()
# plt.show()

"Handling Missing Values in Time-series Data"
print Open.isnull().sum()

from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):

    # #Determing rolling statistics
    # rolmean = pd.rolling_mean(timeseries, window=12)
    # rolstd = pd.rolling_std(timeseries, window=12)

    # #Plot rolling statistics:
    # orig = plt.plot(timeseries, color='blue',label='Original')
    # mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    # std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)

    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput

diff1 = Open.diff(1)
diff2 = Open.diff(2)
diff1 = diff1.dropna()
diff2 = diff2.dropna()
test_stationarity(Open)
test_stationarity(diff1)
# test_stationarity(Close)
# test_stationarity(Ratio)

diff1.plot()
plt.show()

"Analysis with ARIMA Model"

# from statsmodels.tsa.stattools import acf, pacf
# lag_acf = acf(diff1, nlags=20)
# lag_pacf = pacf(diff1, nlags=20, method='ols')
# # Plot ACF:
# plt.subplot(121)
# plt.plot(lag_acf, 'o')
# plt.axhline(y=0, linestyle='--', color='gray')
# plt.axhline(y=-1.96 / np.sqrt(len(diff1)), linestyle='--', color='gray')
# plt.axhline(y=1.96 / np.sqrt(len(diff1)), linestyle='--', color='gray')
# plt.title('Autocorrelation Function')
# # Plot PACF:
# plt.subplot(122)
# plt.plot(lag_pacf, 'or')
# plt.axhline(y=0, linestyle='--', color='gray')
# plt.axhline(y=-1.96 / np.sqrt(len(diff1)), linestyle='--', color='gray')
# plt.axhline(y=1.96 / np.sqrt(len(diff1)), linestyle='--', color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()

# # plt.show()

# "Estimation"
# data = Ratio
