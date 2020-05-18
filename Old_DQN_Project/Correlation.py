import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import errno


def plot(stockname, trendname):
    dfx = pd.read_csv('{}_Stock.csv'.format(stockname))
    dfy = pd.read_csv('{}_Trend.csv'.format(trendname))

    x_high = dfx['High'].values
    y_high = dfy['High'].values
    x_low = dfx['Low'].values
    y_low = dfy['Low'].values
    x_open = dfx['Open'].values
    y_open = dfy['Open'].values
    x_close = dfx['Close'].values
    y_close = dfy['Close'].values

    scaling_factor_x = (max(x_high) - min(x_low))/16
    scaling_factor_x = 1 / scaling_factor_x
    scaling_factor_x = scaling_factor_x / 2

    scaling_factor_y = (max(y_high) - min(y_low)) / 16
    scaling_factor_y = 1 / scaling_factor_y
    scaling_factor_y = scaling_factor_y / 2

    x_high *= scaling_factor_x
    x_low *= scaling_factor_x
    x_open *= scaling_factor_x
    x_close *= scaling_factor_x
    y_high *= scaling_factor_y
    y_low *= scaling_factor_y
    y_open *= scaling_factor_y
    y_close *= scaling_factor_y

    cc = np.corrcoef(x_high, y_high)
    cc = round(cc[0][1], 2)

    # High
    plt.plot(np.unique(x_high), np.poly1d(np.polyfit(x_high, y_high, 1))(np.unique(x_high)))
    plt.scatter(x_high, y_high, label="High", alpha=0.3, s=50, edgecolors='none')
    plt.xlabel("Stock Data Values (High) - {} days".format(len(x_high)))
    plt.ylabel("Trend Data Values (High) - {} days".format(len(y_high)))
    plt.title("Correlation Coefficient = {}".format(cc))

    # # Low
    # plt.plot(np.unique(x_low), np.poly1d(np.polyfit(x_low, y_low, 1))(np.unique(x_low)))
    # plt.scatter(x_low, y_low, label="Low", alpha=0.3, s=50, edgecolors='none')
    #
    # # Open
    # plt.plot(np.unique(x_open), np.poly1d(np.polyfit(x_open, y_open, 1))(np.unique(x_open)))
    # plt.scatter(x_open, y_open, label="Open", alpha=0.3, s=50, edgecolors='none')
    #
    # # Close
    # plt.plot(np.unique(x_close), np.poly1d(np.polyfit(x_close, y_close, 1))(np.unique(x_close)))
    # plt.scatter(x_close, y_close, label="Close", alpha=0.3, s=50, edgecolors='none')

    plt.legend()

    plt.savefig('Correlation - {} vs {} 2020.png'.format(stockname, trendname))
    # plt.show()


if __name__ == '__main__':
    stockname = "NFLX"
    trendname = "HBO"
    plot(stockname, trendname)