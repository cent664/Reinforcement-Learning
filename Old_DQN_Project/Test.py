import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import datetime
import numpy as np
from Candlesticks import candle
import os
import os
import errno
from np_array_data import compute_array, reduce_dim, coloring, make_graph

window_size = 16
steps = 20
df = pd.read_csv('S&P500_Stock_Test.csv')

start = window_size
test_array = compute_array(df, start, window_size)


# To plot a stacked bar graph based on the test array for visualization
make_graph(test_array, start, window_size)
plt.show()