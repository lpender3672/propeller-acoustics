
import matplotlib.pyplot as plt
import numpy as np

tcal_data = np.load('app/bcal_thrust.npy')
qcal_data = np.load('app/bcal_torque.npy')
fig, ax = plt.subplots()
ax.plot(tcal_data[:,0], tcal_data[:,1], '-o')
ax.plot(qcal_data[:,0], qcal_data[:,1], '-o')
ax.grid()


def plot_prop(aero_data, ax1, ax2, label=None):

    force_data = aero_data['force_data']
    motor_data = aero_data['motor_data']

    avg_speed = np.mean(motor_data[:,:,1], axis=1) * 2 * np.pi / 60
    avg_speed = np.abs(avg_speed)
    avg_raw_forces = np.mean(force_data[:,:,1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:,0], tcal_data[:,0,0], tcal_data[:,1,0])
    avg_torque = np.interp(avg_raw_forces[:,1], qcal_data[:,0,1], qcal_data[:,1,1])

    D = 0.127
    rho = 1.225
    A = np.pi * (D/2)**2
    CT = avg_thrust / (rho * A * avg_speed**2)
    CQ = avg_torque / (rho * A * avg_speed**2 * D / 2)

    ax1.loglog(avg_speed, CT, '-o', label = label)
    ax1.grid(which='both')
    ax2.loglog(avg_speed, CQ, '-o', label = label)
    ax2.grid(which='both')

    return ax1, ax2


def plot_FM(aero_data, ax, label=None):

    force_data = aero_data['force_data']
    motor_data = aero_data['motor_data']

    avg_speed = np.mean(motor_data[:,:,1], axis=1) * 2 * np.pi / 60
    avg_speed = np.abs(avg_speed)
    avg_raw_forces = np.mean(force_data[:,:,1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:,0], tcal_data[:,0,0], tcal_data[:,1,0])
    avg_torque = np.interp(avg_raw_forces[:,1], qcal_data[:,0,1], qcal_data[:,1,1])

    D = 0.127
    rho = 1.225
    A = np.pi * (D/2)**2
    CT = avg_thrust / (rho * A * avg_speed**2)
    CQ = avg_torque / (rho * A * avg_speed**2 * D / 2)

    FM = CT ** (3/2) / CQ

    ax.semilogx(avg_speed, FM, '-o', label = label)

    return ax


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



dalprop5045 = np.load('app/results/brng_torque/dalprop5045.npz')
dalprop5045bnr = np.load('app/results/brng_torque/dalprop5045bnr.npz')
foxeer_toroidal = np.load('app/results/brng_torque/foxeer_toroidal.npz')

d100clarky = np.load('app/results/brng_torque/D100clarkY.npz')
d100naca0018 = np.load('app/results/brng_torque/D100naca0018.npz')

fig, ax = plt.subplots( 2, 1)
plot_prop(dalprop5045, ax[0], ax[1], label='2 blade')
plot_prop(dalprop5045bnr, ax[0], ax[1], label='3 blade')
plot_prop(foxeer_toroidal, ax[0], ax[1], label='toroidal')

plot_prop(d100clarky, ax[0], ax[1], label='clarkY')
plot_prop(d100naca0018, ax[0], ax[1], label='naca0018')

ax[0].set_xlim([50, 2000])
ax[1].set_xlim([100, 2000])

ax[1].legend(loc='lower left')

fig, ax = plt.subplots()

plot_FM(dalprop5045, ax, label='2 blade')
plot_FM(dalprop5045bnr, ax, label='3 blade')
plot_FM(foxeer_toroidal, ax, label='toroidal')


ax.grid()
ax.legend()

plt.show()
