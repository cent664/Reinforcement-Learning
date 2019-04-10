import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

df = pd.read_csv("NFLX.csv")  # Reading the data

current_index = 1500
window_size = 5
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
"""
fig = plt.figure()
fig.canvas.draw()

# Now we can save it to a numpy array.
data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
print((data).shape)

plt.imshow(data, interpolation='nearest')
plt.show()

#img = Image.fromarray(data, 'RGB')
#img.save('my.png')
#img.show()
"""

#plt.plot()
canvas = plt.get_current_fig_manager().canvas

agg = canvas.switch_backends(FigureCanvasAgg)
#agg.draw()
s, (width, height) = agg.print_to_buffer()

# Convert to a NumPy array.
im_data = np.frombuffer(s, np.uint8).reshape((height, width, 4))

# Pass off to PIL.
im = Image.frombytes("RGBA", (width, height), s)
im.show()