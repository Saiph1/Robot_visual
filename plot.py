import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mov_avg_x = np.loadtxt('mov_avg_x.txt')
mov_avg_y = np.loadtxt('mov_avg_y.txt')
dx = np.loadtxt('dx.txt')
dy = np.loadtxt('dy.txt')
x = np.loadtxt('x.txt')
x = x - 154
y = np.loadtxt('y.txt')
y = y - 149
# Calculate dy/dx
# dy_dx = [dy[i]/dx[i] for i in range(len(dx))]
# move_dy_dx = [mov_avg_y[i]/mov_avg_x[i] for i in range(len(mov_avg_x))]
index = range(len(dy))
# window_size = 3
# rolling_mean = pd.Series(x).rolling(window_size).mean()

# Calculate the exponential moving average of values with smoothing factor alpha=0.5
alpha = 0.2
ema = pd.Series(dy, index).ewm(alpha=alpha).mean()

# Plot the original y values and rolling mean
# plt.plot(range(1, len(x)+1), index, linestyle="solid")
plt.plot(range(1, 601), dy[:600], label=f'dy', linestyle="--")
plt.plot(range(1, 601), ema[:600], label=f'Exponential moving average dy',color="red")
plt.legend()
plt.xlabel('Time (frames)')
plt.ylabel('dy (cm)')
plt.title('dy with exponential moving average (alpha = 0.2)')
plt.show()
# Plot dy/dx
# plt.plot(range(1, len(mov_avg_x)+1), mov_avg_x, linestyle="solid")
# plt.plot(range(1, len(x)+1), x, linestyle="--")
# plt.xlabel('i')
# plt.ylabel('dx')
# plt.title('dx')
# plt.show()