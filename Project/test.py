import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("TSLA.csv")  # Reading the data

current_index = 1500
window_size = 500
w = 1.0
lw = 0.5

for i in range(current_index - window_size + 1, current_index + 1):
    data = df[['Low', 'Close', 'Open', 'High']].iloc[i].values

    #print("Low={}, Close={}, Open={}, High={}".format(data[0], data[1], data[2], data[3]))

    if data[2] > data[1]:
        plt.bar(i + 1, data[1] - data[0], width=w, bottom=data[0], color='#be2409', edgecolor='Black', linewidth=lw)
        plt.bar(i + 1, data[2] - data[1], width=w, bottom=data[1], color='White', edgecolor='Black', linewidth=lw)
        plt.bar(i + 1, data[3] - data[2], width=w, bottom=data[2], color='#fddc54', edgecolor='Black', linewidth=lw)
    else:
        plt.bar(i + 1, data[2] - data[0], width=w, bottom=data[0], color='#be2409', edgecolor='Black', linewidth=lw)
        plt.bar(i + 1, data[1] - data[2], width=w, bottom=data[2], color='Black', edgecolor='Black', linewidth=lw)
        plt.bar(i + 1, data[3] - data[1], width=w, bottom=data[1], color='#fddc54', edgecolor='Black', linewidth=lw)

plt.show()