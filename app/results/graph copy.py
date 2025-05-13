import matplotlib.pyplot as plt
import numpy as np

import app.routines

toroidal_data = np.load("app/results/foxeer_toroidal.npz")
dalprop_2blade = np.load("app/results/dalprop5045.npz")
dalprop_3blade = np.load("app/results/dalprop5045bnr.npz")

cal_data = np.load("app/calibration.npy")

fig, ax = plt.subplots()
ax.plot(cal_data[:, 0], cal_data[:, 1], "-o")
ax.grid()


fig, ax1 = plt.subplots()


def plot_prop(aero_data, ax1, label=None):

    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    avg_speed = np.mean(motor_data[:, :, 1], axis=1) * 2 * np.pi / 60
    avg_raw_forces = np.mean(force_data[:, :, 1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:, 0], cal_data[:, 0, 0], cal_data[:, 1, 0])
    avg_torque = np.interp(avg_raw_forces[:, 1], cal_data[:, 0, 1], cal_data[:, 1, 1])

    filtered_speed = avg_speed[avg_thrust > 0.15]
    filtered_thrust = avg_thrust[avg_thrust > 0.15]

    ax1.loglog(filtered_speed, filtered_thrust, "-o", label=label)
    ax1.grid(which="both")

    return ax1


ax1.set_title("Thrust vs Speed")
ax1.set_xlabel("Speed (rad/s)")
ax1.set_ylabel("Thrust (N)")

ax1 = plot_prop(toroidal_data, ax1, label="Toroidal")
ax1 = plot_prop(dalprop_2blade, ax1, label="DAL 5045")
ax1 = plot_prop(dalprop_3blade, ax1, label="DAL 5045BNR")


def plot_prop_nondim(aero_data, prop_data, ax1, label=None):

    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    rt = prop_data["rt"]
    A = np.pi * rt**2
    rho = 1.225

    avg_speed = np.mean(motor_data[:, :, 1], axis=1) * 2 * np.pi / 60
    avg_raw_forces = np.mean(force_data[:, :, 1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:, 0], cal_data[:, 0, 0], cal_data[:, 1, 0])
    avg_torque = np.interp(avg_raw_forces[:, 1], cal_data[:, 0, 1], cal_data[:, 1, 1])

    filtered_speed = avg_speed[avg_thrust > 0.15]
    filtered_thrust = avg_thrust[avg_thrust > 0.15]

    linear_speed = filtered_speed * rt
    non_dim_thrust = filtered_thrust / (0.5 * rho * A * filtered_speed**2)

    ax1.loglog(linear_speed, non_dim_thrust, "-o", label=label)
    ax1.grid(which="both")

    return ax1


fig, ax1 = plt.subplots()
prop_data = routines.load_prop_from_file("app/props/constant_chord_hires.prop")

plot_prop_nondim(dalprop_2blade, prop_data, ax1, label="DAL 5045")
plot_prop_nondim(dalprop_3blade, prop_data, ax1, label="DAL 5045 3blade")
plot_prop_nondim(toroidal_data, prop_data, ax1, label="toroidal")


ax1.legend()


plt.show()
