import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np

data = pd.read_csv('S&P500_test.csv')
window_size = 16
start = window_size - 1
steps = 20

# Converting date to pandas datetime format
data['Date'] = pd.to_datetime(data['Date'])
data["Date"] = data["Date"].apply(mdates.date2num)

# Creating required data in new DataFrame OHLC
ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']]

# In case you want to check for shorter timespan
ohlc = ohlc[15: 35]
ohlc = ohlc.values
ohlc = ohlc.tolist()

for i in range(0, 20):
    ohlc[i][0] = ohlc[0][0] + i
    ohlc[i] = tuple(ohlc[i])

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))

# plot the candlesticks
candlestick_ohlc(ax, ohlc, width=.6)

for label in ax.xaxis.get_ticklabels():
    label.set_rotation(45)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax.grid(True)

plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('S&P500')
plt.legend()
# plt.xticks(list(data['Date'].head(20)))
# locs, labels = plt.xticks()
# plt.xticks(list(data['Date'].head(20)), [labels])
plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
plt.show()