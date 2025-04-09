
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from routines import (
    load_prop_from_file,
)

tcal_data = np.load('app/bcal_thrust.npy')
qcal_data = np.load('app/bcal_torque.npy')
fig, ax = plt.subplots()
ax.plot(tcal_data[:,0], tcal_data[:,1], '-o')
ax.plot(qcal_data[:,0], qcal_data[:,1], '-o')
ax.grid()


def calc_forces(aero_data):

    force_data = aero_data['force_data']
    motor_data = aero_data['motor_data']

    avg_speed = np.mean(motor_data[:,:,1], axis=1) * 2 * np.pi / 60
    avg_speed = np.abs(avg_speed)
    avg_raw_forces = np.mean(force_data[:,:,1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:,0], tcal_data[:,0,0], tcal_data[:,1,0], np.NaN, np.NaN)
    avg_torque = np.interp(avg_raw_forces[:,1], qcal_data[:,0,1], qcal_data[:,1,1], np.NaN, np.NaN)

    return avg_speed, avg_thrust, avg_torque

def plot_prop(prop_result_path, ax1, ax2, label=None):

    full_path = Path(prop_result_path).resolve()
    fprop = full_path.parent.parent / "props" / full_path.name
    prop = load_prop_from_file(fprop)

    candidates = list(full_path.glob("aero_*"))
    if not candidates:
        print("Warning: No aero data found for", prop_result_path)
        return None
    
    faero = max(candidates, key=lambda f: f.stat().st_mtime)

    aero_data = np.load(faero)

    avg_speed, avg_thrust, avg_torque = calc_forces(aero_data)

    rho = 1.225
    rt = prop['rt']
    A = np.pi * rt**2

    filtered_speed_t = avg_speed[avg_thrust > 1e-2]
    filtered_thrust = avg_thrust[avg_thrust > 1e-2]
    filtered_speed_q = avg_speed[avg_torque > 1e-4]
    filtered_torque = avg_torque[avg_torque > 1e-4]

    CT = filtered_thrust / (rho * A * filtered_speed_t**2 * rt**2)
    CQ = filtered_torque / (rho * A * filtered_speed_q**2 * rt**3)

    ax1.semilogx(filtered_speed_t, CT, '-o', label = label)
    ax1.grid(which='both')
    ax2.semilogx(filtered_speed_q, CQ, '-o', label = label)
    ax2.grid(which='both')

    return ax1, ax2


def plot_FM(prop_result_path, ax, label=None):

    full_path = Path(prop_result_path).resolve()
    fprop = full_path.parent.parent / "props" / full_path.name
    prop = load_prop_from_file(fprop)

    candidates = list(full_path.glob("aero_*"))
    if not candidates:
        print("Warning: No aero data found for", prop_result_path)
        return None
    
    faero = max(candidates, key=lambda f: f.stat().st_mtime)

    aero_data = np.load(faero)

    force_data = aero_data['force_data']
    motor_data = aero_data['motor_data']

    avg_speed = np.mean(motor_data[:,:,1], axis=1) * 2 * np.pi / 60
    avg_speed = np.abs(avg_speed)
    avg_raw_forces = np.mean(force_data[:,:,1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:,0], tcal_data[:,0,0], tcal_data[:,1,0])
    avg_torque = np.interp(avg_raw_forces[:,1], qcal_data[:,0,1], qcal_data[:,1,1])

    filtered_speed = avg_speed[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]
    filtered_thrust = avg_thrust[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]
    filtered_torque = avg_torque[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]

    D = 0.127
    rho = 1.225
    A = np.pi * (D/2)**2
    CT = filtered_thrust / (rho * A * filtered_speed**2 * D**2)
    CQ = filtered_torque / (rho * A * filtered_speed**2 * D**2 * D / 2)

    FM = CT ** (3/2) / CQ

    ax.semilogx(filtered_speed, FM, '-o', label = label)

    return ax


fig, ax = plt.subplots( 2, 1)
plot_prop('app/results/dalprop5045.prop', ax[0], ax[1], label='dalprop5045')
plot_prop('app/results/printed5045.prop', ax[0], ax[1], label='printed 5045')
plot_prop('app/results/dalprop4045.prop', ax[0], ax[1], label='4045')
#plot_prop('app/results/dalprop5045bnr.prop', ax[0], ax[1], label='3 blade')
#plot_prop('app/results/d100clarkY.prop', ax[0], ax[1], label='clarkY')

plot_prop('app/results/dalprop6045.prop', ax[0], ax[1], label='6045')

ax[0].set_xlim([50, 2000])
ax[1].set_xlim([100, 2000])
ax[0].set_ylim([0, 0.05])
ax[1].set_ylim([0, 0.01])

ax[1].legend(loc='lower left')

fig, ax = plt.subplots( 1, 1)

plot_FM('app/results/dalprop5045.prop', ax)
plot_FM('app/results/printed5045.prop', ax)
plot_FM('app/results/dalprop4045.prop', ax)
plot_FM('app/results/dalprop6045.prop', ax)



plt.show()
