

import numpy as np
import matplotlib.pyplot as plt

# plot cal data


thrust_cal_data = np.load('app/results/thrust_cal.npy')
torque_cal_data = np.load('app/results/torque_cal.npy')
fig, axes = plt.subplots(2, 1)

def sortx(xdata, ydata):

    args_sorted = np.argsort(xdata)
    xdata = xdata[args_sorted]
    ydata = ydata[args_sorted]

    return xdata, ydata


axes[0].plot(*sortx(thrust_cal_data[:,0,0], thrust_cal_data[:,1,0]), '-o', label='Thrust calibration')
axes[0].plot(torque_cal_data[:,0,0], torque_cal_data[:,1,0], '-o', label="Variation from torque calibration")
axes[0].set_ylabel('Force (N)')
axes[1].plot(*sortx(torque_cal_data[:,0,1], torque_cal_data[:,1,1]), '-o', label='Torque calibration')
axes[1].set_ylabel('Moment (Nm)')
axes[1].plot(thrust_cal_data[:,0,1], thrust_cal_data[:,1,1], '-o', label='Variation from thrust calibration')
axes[0].grid()
axes[1].grid()
axes[0].legend()
axes[1].legend()

axes[1].set_xlabel('Raw ADC value')
plt.show()
