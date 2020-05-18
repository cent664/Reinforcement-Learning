import numpy as np
import matplotlib.pyplot as plt

creward7 = [3.92, 3.92, 20.32, -10.73, -13.59, 9.6, 13.59, 2.82, -4.8, 12.25]
cc7 = [0.32, 0.56, 0.73, 0.08, 0.26, -0.38, 0.49, 0.18, 0.16, 0.41]

creward14 = [6.97, 45.81, 35.84, -13.4, 7.72, 15, 25.87, -2.61, -24.92, 28.08]
cc14 = [0.32, 0.57, 0.73, 0.07, 0.25, -0.38, 0.48, 0.18, 0.16, 0.41]

plt.plot(np.unique(cc7), np.poly1d(np.polyfit(cc7, creward7, 2))(np.unique(cc7)))
plt.scatter(cc7, creward7, label="1 week")

plt.plot(np.unique(cc14), np.poly1d(np.polyfit(cc14, creward14, 2))(np.unique(cc14)))
plt.scatter(cc14, creward14, label="2 weeks")

plt.xlabel("Correlation Coefficient")
plt.ylabel("Cumulative Reward")
plt.title("Correlation Coefficient vs Cumulative Reward")
plt.legend()
plt.grid()
plt.savefig("cc_vs_creward_combo.png")
plt.show()