import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

filename = 'exportfile(9).txt'
with open(filename, 'r') as f:
    lines = f.readlines()

Free_Acc_X, Free_Acc_Y, Free_Acc_Z = [], [], []
pattern = r'[-+]?\d+\.\d+'

for line in lines:
    line = line.strip()
    parts = line.split('|')

    acc_data = parts[0].split(',')

    Free_Acc_X.append(float(re.search(r'FREE Acc X: (\S+)', acc_data[0]).group(1)))
    Free_Acc_Y.append(float(re.search(r'Free Acc Y: (\S+)', acc_data[1]).group(1)))
    Free_Acc_Z.append(float(re.search(r'Free Acc Z: (\S+)', acc_data[2]).group(1)))

accel_data = np.column_stack((Free_Acc_X, Free_Acc_Y, Free_Acc_Z))

fs = 100
t_acc = np.arange(len(Free_Acc_X)) / fs

vel_x = cumulative_trapezoid(Free_Acc_X, t_acc, initial=0)
pos_x = cumulative_trapezoid(vel_x, t_acc, initial=0)

vel_y = cumulative_trapezoid(Free_Acc_Y, t_acc, initial=0)
pos_y = cumulative_trapezoid(vel_y, t_acc, initial=0)

vel_z = cumulative_trapezoid(Free_Acc_Z, t_acc, initial=0)
pos_z = cumulative_trapezoid(vel_z, t_acc, initial=0)

plt.subplot(311)
plt.plot(t_acc, pos_x)
plt.xlabel('t/s')
plt.ylabel('x/m')

plt.subplot(312)
plt.plot(t_acc, pos_y)
plt.xlabel('t/s')
plt.ylabel('y/m')

plt.subplot(313)
plt.plot(t_acc, pos_z)
plt.xlabel('t/s')
plt.ylabel('z/m')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(pos_x, pos_y, pos_z)
ax.set_xlabel('x/m')
ax.set_ylabel('y/m')
ax.set_zlabel('z/m')

plt.show()