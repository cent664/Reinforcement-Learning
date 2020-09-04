import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import errno
from matplotlib.ticker import MaxNLocator
from PIL import Image
from np_array_data import compute_array, reduce_dim, coloring
from Trends import get_trends


def candlesticks():
    df = pd.read_csv('temp.csv')
    df = df[222:len(df)-1]
    print(len(df))

    window_size = 16
    fig, ax = plt.subplots()
    steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    start = window_size
    closes = []
    opens = []
    col = []
    bottoms = []
    colors = []
    w = 0.6

    # Creating the array
    for i in range(0, len(df)):
        if i < 0:
            print("Indexing error")
            break

        data = df[['Low', 'Close', 'Open', 'High']].iloc[i].values

        low = data[0]
        close = data[1]
        open = data[2]
        high = data[3]

        closes.append(close)
        opens.append(open)

    # Converting date to pandas datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    og_dates = [(df['Date'].iloc[window_size - 1]).strftime('%d-%m-%Y')]

    # Converting datetime objects to the correct date format
    for i in range(0, len(df)):
        og_dates.append((df['Date'].iloc[i]).strftime('%d-%m-%Y'))

    # Graph drawing parameters
    lw = 0.5

    for i in range(0, len(closes)):  # 0 -> Total number of columns
        col.append(abs(closes[i] - opens[i]))
        if closes[i] > opens[i]:
            bottoms.append(opens[i])
            colors.append('Black')
        else:
            bottoms.append(closes[i])
            colors.append('Red')

    ax.bar(steps, col, bottom=bottoms, color=colors, width=w)
    ax.set_xticklabels(og_dates)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax.set_xticks(np.arange(len(og_dates)))  # Forcing every label

    ax.grid(True)

    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title('S&P500')

    plt.show()


def bitmap():

    steps = 7
    window_size = 16
    index = window_size - 1

    df = pd.read_csv('temp.csv')
    df_window = df[222:len(df) - 1]

    df = df[len(df) - (steps + window_size): len(df)]

    df_high = df_window['High'].values
    df_low = df_window['Low'].values

    # Calculating Scaling factor for stocks
    maxRange = -1000000
    for i in range(0, steps):
        maxRange = max(maxRange, max(df_high) - min(df_low))

    dollars_per_pixel = maxRange / window_size
    scaling_factor = 1 / dollars_per_pixel
    scaling_factor = scaling_factor / 2  # To account for the shift from centering close

    test_array = compute_array(df_window, index, window_size)
    test_array = reduce_dim(test_array, scaling_factor)
    im_data = coloring(test_array, (16, 16), 'Simple')
    stock_image = Image.fromarray(np.uint8(im_data), 'L')
    stock_image.save("idk.bmp")


if __name__ == '__main__':
    # candlesticks()
    bitmap()