from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.optimize import curve_fit


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routines import (
    load_prop_from_file,
)

tcal_data = np.load("app/bcal_thrust.npy")
qcal_data = np.load("app/bcal_torque.npy")
fig, ax = plt.subplots()
ax.plot(tcal_data[:, 0], tcal_data[:, 1], "-o")
ax.plot(qcal_data[:, 0], qcal_data[:, 1], "-o")
ax.grid()

cal_data = (tcal_data, qcal_data)


def cexp(x, a, b, c):
    """Converging exponential function."""
    return a * (1 - np.exp(-b * x)) + c

def fit_cexp(x, y):
    """Fits a converging exponential to the data."""

    initial_guess = [np.max(y), 0.1, 0]

    # sort x and y to ensure they are in the same order
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    try:
        params, _ = curve_fit(cexp, x, y, p0=initial_guess)
    except RuntimeError:
        
        print("Error: Curve fitting failed. Returning initial guess.")
        return initial_guess

    return params


def calc_mean_forces(aero_data, tcal_data, qcal_data):
    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    mean_speed = np.mean(motor_data[:, :, 1], axis=1) * 2 * np.pi / 60
    mean_speed = np.abs(mean_speed)

    mean_raw_forces = np.mean(force_data[:, :, 1:], axis=1)

    # interpolate to calibration data
    mean_thrust = np.interp(
        mean_raw_forces[:, 0], tcal_data[:, 0, 0], tcal_data[:, 1, 0], np.nan, np.nan
    )
    mean_torque = np.interp(
        mean_raw_forces[:, 1], qcal_data[:, 0, 1], qcal_data[:, 1, 1], np.nan, np.nan
    )
    return mean_speed, mean_thrust, mean_torque


def calc_std_forces(aero_data, tcal_data, qcal_data):

    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    # speed samples and std
    speed_samples = motor_data[:, :, 1] * 2 * np.pi / 60
    std_speed = np.std(speed_samples, axis=1)

    raw_thrust = force_data[:, :, 1]
    raw_torque = force_data[:, :, 2]

    thrust_samples_cal = np.stack(
        [
            np.interp(
                raw_thrust[i, :], tcal_data[:, 0, 0], tcal_data[:, 1, 0], np.nan, np.nan
            )
            for i in range(raw_thrust.shape[0])
        ],
        axis=0,
    )
    torque_samples_cal = np.stack(
        [
            np.interp(
                raw_torque[i, :], qcal_data[:, 0, 1], qcal_data[:, 1, 1], np.nan, np.nan
            )
            for i in range(raw_torque.shape[0])
        ],
        axis=0,
    )

    std_thrust = np.std(thrust_samples_cal, axis=1)
    std_torque = np.std(torque_samples_cal, axis=1)

    return std_speed, std_thrust, std_torque


def plot_prop(
    prop_result_path,
    ax1,
    ax2,
    cal_data,
    label=None,
    tcal_data=None,
    qcal_data=None,
    rho=1.225,
):
    """
    Plot thrust and torque coefficients vs. speed with error bars.
    """
    tcal_data, qcal_data = cal_data
    full_path = Path(prop_result_path).resolve()
    fprop = full_path.parent.parent / "props" / full_path.name
    prop = load_prop_from_file(fprop)

    # Get the next color from the current color cycle using the correct method
    prop_color = ax1._get_lines.get_next_color()

    # find latest aero file for this prop
    candidates = list(full_path.glob("aero_*"))
    if not candidates:
        print("Warning: No aero data found for", prop_result_path)
        return None
    faero = max(candidates, key=lambda f: f.stat().st_mtime)
    aero_data = np.load(faero)

    # calculate means and stds
    mean_speed, mean_thrust, mean_torque = calc_mean_forces(
        aero_data, tcal_data, qcal_data
    )
    std_speed, std_thrust, std_torque = calc_std_forces(aero_data, tcal_data, qcal_data)

    rt = prop["rt"]
    A = np.pi * rt**2

    mask_t = mean_thrust > 1e-2
    mask_q = mean_torque > 1e-4

    speed_t = mean_speed[mask_t]
    thrust = mean_thrust[mask_t]
    err_speed_t = std_speed[mask_t]
    err_thrust = std_thrust[mask_t]

    speed_q = mean_speed[mask_q]
    torque = mean_torque[mask_q]
    err_speed_q = std_speed[mask_q]
    err_torque = std_torque[mask_q]

    # compute coefficients
    CT = thrust / (rho * A * speed_t**2 * rt**2)
    CQ = torque / (rho * A * speed_q**2 * rt**3)
    # propagate errors (approx)

    yerr_CT = err_thrust / (rho * A * speed_t**2 * rt**2)
    yerr_CQ = err_torque / (rho * A * speed_q**2 * rt**3)

    # Use explicit color for thrust coefficient plot
    ax1.plot(speed_t, CT, "o", markersize=5, label=label, zorder=1, color=prop_color)
    # ax1.errorbar(speed_t, CT, xerr=err_speed_t, yerr=yerr_CT, fmt='none', ecolor='black', zorder=2)
    ax1.set_xscale("log")

    if speed_t.size:
        curr_min_t, curr_max_t = ax1.get_xlim()
        xmin_t = np.min(speed_t - err_speed_t) * 0.9
        xmax_t = np.max(speed_t + err_speed_t) * 1.1
        ax1.set_xlim(min(curr_min_t, xmin_t), max(curr_max_t, xmax_t))
    ax1.grid(True, which="both")
    ax1.set_xlabel("Speed [rad/s]")
    ax1.set_ylabel("Thrust Coefficient $C_T$")

    # Use same color for torque coefficient plot
    ax2.plot(speed_q, CQ, "o", markersize=5, label=label, zorder=1, color=prop_color)

    # ax2.errorbar(speed_q, CQ, xerr=err_speed_q, yerr=yerr_CQ, fmt='none', ecolor='black', zorder=2)
    ax2.set_xscale("log")
    if speed_q.size:
        curr_min_q, curr_max_q = ax2.get_xlim()
        xmin_q = np.min(speed_q - err_speed_q) * 0.9
        xmax_q = np.max(speed_q + err_speed_q) * 1.1
        ax2.set_xlim(min(curr_min_q, xmin_q), max(curr_max_q, xmax_q))

    # ok now fit curves to estimate the constant coefficients
    # thrust curve is constant enough to be averaged

    converged_thrust_coefficient = np.mean(CT)
    
    # fit torque curve
    params_q = fit_cexp(np.log10(speed_q), CQ)
    speed_cont_q = np.logspace(
        np.log10(np.min(speed_q)), np.log10(np.max(speed_q)), 1000
    )
    CQ_cont = cexp(np.log10(speed_cont_q), *params_q)
    
    # Use same color for fit line with solid style
    ax2.plot(speed_cont_q, CQ_cont, "-", color=prop_color)

    converged_torque_coefficient = params_q[0] + params_q[2]

    print(
        f"Prop: {label} | CT: {converged_thrust_coefficient:.7f} | CQ: {converged_torque_coefficient:.7f}"
    )
    
    ax2.grid(True, which="both")
    ax2.set_xlabel("Speed [rad/s]")
    ax2.set_ylabel("Torque Coefficient $C_Q$")

    return ax1, ax2


def plot_FM(prop_result_path, ax, cal_data, label=None):

    tcal_data, qcal_data = cal_data

    full_path = Path(prop_result_path).resolve()
    fprop = full_path.parent.parent / "props" / full_path.name
    prop = load_prop_from_file(fprop)

    candidates = list(full_path.glob("aero_*"))
    if not candidates:
        print("Warning: No aero data found for", prop_result_path)
        return None

    faero = max(candidates, key=lambda f: f.stat().st_mtime)

    aero_data = np.load(faero)

    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    avg_speed = np.mean(motor_data[:, :, 1], axis=1) * 2 * np.pi / 60
    avg_speed = np.abs(avg_speed)
    avg_raw_forces = np.mean(force_data[:, :, 1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:, 0], tcal_data[:, 0, 0], tcal_data[:, 1, 0])
    avg_torque = np.interp(avg_raw_forces[:, 1], qcal_data[:, 0, 1], qcal_data[:, 1, 1])

    filtered_speed = avg_speed[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]
    filtered_thrust = avg_thrust[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]
    filtered_torque = avg_torque[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]

    D = 0.127
    rho = 1.225
    A = np.pi * (D / 2) ** 2
    CT = filtered_thrust / (rho * A * filtered_speed**2 * D**2)
    CQ = filtered_torque / (rho * A * filtered_speed**2 * D**2 * D / 2)

    FM = CT ** (3 / 2) / CQ

    ax.semilogx(filtered_speed, FM, "-o", label=label)

    ax.grid(True, which="both")
    ax.set_xlabel("Speed [rad/s]")
    ax.set_ylabel("Figure of Merit (FM)")

    return ax


fig, ax = plt.subplots(2, 1)
plot_prop("app/results/dalprop5045.prop", ax[0], ax[1], cal_data, label="dalprop5045")
plot_prop("app/results/printed5045.prop", ax[0], ax[1], cal_data, label="printed 5045")
plot_prop("app/results/dalprop4045.prop", ax[0], ax[1], cal_data, label="4045")
plot_prop('app/results/dalprop5045bnr.prop', ax[0], ax[1], cal_data, label='3 blade')
plot_prop("app/results/dalprop6045.prop", ax[0], ax[1], cal_data, label="6045")

ax[1].legend(loc="upper left")

fig, ax = plt.subplots(2, 1)

plot_prop("app/results/printed5045.prop", ax[0], ax[1], cal_data, label="0 deg sweep")
plot_prop("app/results/5045_s15.prop", ax[0], ax[1], cal_data, label="15 deg sweep")
plot_prop("app/results/5045_s30.prop", ax[0], ax[1], cal_data, label="30 deg sweep")
plot_prop("app/results/5045_s45.prop", ax[0], ax[1], cal_data, label="45 deg sweep")

# ax[0].set_xlim([50, 2000])
# ax[1].set_xlim([100, 2000])
# ax[0].set_ylim([0, 0.05])
# ax[1].set_ylim([0, 0.01])

ax[1].legend(loc="upper left")

fig, ax = plt.subplots(1, 1)

plot_FM("app/results/dalprop5045.prop", ax, cal_data)
plot_FM("app/results/printed5045.prop", ax, cal_data)
plot_FM("app/results/dalprop4045.prop", ax, cal_data)
plot_FM("app/results/dalprop6045.prop", ax, cal_data)

ax.legend(loc="upper left")

plt.show()
