import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import datetime
import numpy as np

def candle():
    df = pd.read_csv('S&P500_test.csv')
    window_size = 16
    start = window_size - 1
    steps = 20

    # Converting date to pandas datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    og_dates = []

    # Converting datetime objects to the correct date format
    for i in range(0, len(df['Date'])):
        og_dates.append(df['Date'][i].strftime('%d-%m-%Y'))

    df["Date"] = df["Date"].apply(mdates.date2num)

    # Creating required data in new DataFrame OHLC
    ohlc = df[['Date', 'Open', 'High', 'Low', 'Close']]

    # In case you want to check for shorter timespan
    ohlc = ohlc[start: start + steps]
    ohlc = ohlc.values
    ohlc = ohlc.tolist()

    # Replacing x axis values with consecutive dates to eliminate gaps
    for i in range(0, steps):
        ohlc[i][0] = ohlc[0][0] + i
        ohlc[i] = tuple(ohlc[i])

    fig = plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0))

    # Plot the candlesticks
    candlestick_ohlc(ax, ohlc, width=.6)

    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(steps + 3))
    ax.grid(True)

    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title('S&P500')

    # Replacing x labels with the correct dates
    # locs, labels = plt.xticks()
    # plt.xticks(locs, list(og_dates[start - 2: start + steps + 2]))
    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    plt.show()

if __name__ == '__main__':
    candle()