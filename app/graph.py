
import matplotlib.pyplot as plt
import numpy as np

cal_data = np.load('app/calibration.npy')

fig, ax = plt.subplots()
ax.plot(cal_data[:,0], cal_data[:,1], '-o')
ax.grid()




def plot_prop(aero_data, ax1, ax2, label=None):

    force_data = aero_data['force_data']
    motor_data = aero_data['motor_data']

    avg_speed = np.mean(motor_data[:,:,1], axis=1) * 2 * np.pi / 60
    avg_raw_forces = np.mean(force_data[:,:,1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:,0], cal_data[:,0,0], cal_data[:,1,0])
    avg_torque = np.interp(avg_raw_forces[:,1], cal_data[:,0,1], cal_data[:,1,1])

    ax1.loglog(avg_speed, avg_raw_forces[:,0], '-o', label = label)
    ax1.grid(which='both')
    ax2.loglog(avg_speed, 1e6 - avg_raw_forces[:,1], '-o', label = label)
    ax2.grid(which='both')

    return ax1, ax2


def plot_folder(folder):

    fig, [ax1, ax2] = plt.subplots(2, 1)

    toroidal_data = np.load(folder + '/foxeer_toroidal.npz')
    dalprop_2blade = np.load(folder + '/dalprop5045.npz')
    dalprop_3blade = np.load(folder + '/dalprop5045bnr.npz')

    ax1.set_title('Thrust vs Speed')
    ax1.set_xlabel('Speed (rad/s)')
    ax1.set_ylabel('Thrust (N)')
    ax2.set_title('Torque vs Speed')
    ax2.set_xlabel('Speed (rad/s)')
    ax1.set_ylabel('Torque (Nm)')

    ax1, ax2 = plot_prop(toroidal_data, ax1, ax2, label='Toroidal')
    ax1, ax2 = plot_prop(dalprop_2blade, ax1, ax2, label='DAL 5045')
    ax1, ax2 = plot_prop(dalprop_3blade, ax1, ax2, label='DAL 5045BNR')

    ax1.set_xlim([100, 14000])
    ax2.set_xlim([100, 14000])

    ax2.legend(loc='lower left')

plot_folder(folder='app/results/1kg_torque')
plot_folder(folder='app/results/0.2kg_torque')

plt.show()
