import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import datetime
import numpy as np
from Candlesticks import candle
import random
import os
import errno

window_size = 16

df = pd.read_csv("S&P500_train.csv")
start_date = df['Date'][window_size]
print(start_date)


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = []

for i in range (0,10):
    y.append(random.randint(1, 10))

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
ax.xaxis.set_major_locator(mticker.MaxNLocator(len(x)))
plt.plot(x, y)
plt.grid(True)

graphpath = 'Graphs/Some fig.png'
if not os.path.exists(os.path.dirname(graphpath)):
    try:
        os.makedirs(os.path.dirname(graphpath))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

plt.savefig(graphpath)
plt.show()