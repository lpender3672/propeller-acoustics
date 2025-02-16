
import matplotlib.pyplot as plt
import numpy as np


data = np.load('app/results/aero.npz')
force_data = data['force_data']
motor_data = data['motor_data']

print(force_data.shape)
print(motor_data.shape)

avg_speed = np.mean(motor_data[:,:,1], axis=1)
avg_forces = np.mean(force_data[:,:,1:], axis=1)

fig, [ax1, ax2] = plt.subplots(2, 1)
ax1.loglog(avg_speed, avg_forces[:,0], '-o', color='r')
ax1.grid(which='both')
ax2.loglog(avg_speed, avg_forces[:,1], '-o', color='b')
ax2.grid(which='both')
plt.show()
