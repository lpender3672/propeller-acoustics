# This file will investigate if odrive torque agrees with odrive torque
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from app.routines_aero import (
    load_cell_calibration,
    calc_mean_forces
)

def torque_from_motor_data(prop_result_path='app/results/dalprop5045.prop', plot=False):

    noloadf = "app/results/noprop.prop/aero_2025-05-21-22-57-36.npz"

    full_path = Path(prop_result_path).resolve()
    candidates = list(full_path.glob("aero_*"))
    if not candidates:
        print("Warning: No aero data found for", prop_result_path)
        return None

    loadf = max(candidates, key=lambda f: f.stat().st_mtime)

    noload_motor_data = np.load(noloadf)["motor_data"]
    load_motor_data = np.load(loadf)["motor_data"]

    kt = 3/2 * 1/np.sqrt(3) * 60 / (2*np.pi) * 1/1700 # torque constant
    # conflicting information on the current measurement
    # confirmed to be current ampltude, not RMS so to get rms, dividing by sqrt(2) is correct
    kt /= np.sqrt(2)

    load_speeds = np.mean(load_motor_data[:, :, 1], axis=1) * -2 * np.pi / 60
    noload_speeds = np.mean(noload_motor_data[:, :, 1], axis=1) * -2 * np.pi / 60
    # assert speeds are clos

    abs_diff = np.abs(load_speeds - noload_speeds)
    rel_diff = abs_diff / np.maximum(load_speeds, noload_speeds)
    valid_mask = rel_diff < 1

    # Apply mask
    load_motor_data = load_motor_data[valid_mask]
    noload_motor_data = noload_motor_data[valid_mask]


    load_currents = np.mean(load_motor_data[:, :, 2], axis=1)
    noload_currents = np.mean(noload_motor_data[:, :, 2], axis=1)

    dcurrent = load_currents - noload_currents
    torques = kt * dcurrent

    speeds_rads = load_speeds[valid_mask]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        ax.plot(speeds_rads, torques, "o", label="dcurrent")
        ax.set_xscale("log")
        ax.set_yscale("log")

        xlo, xhi = ax.get_xlim()
        x = np.linspace(xlo, xhi, 100)
        ax.plot(x, 1e-8 * x**2, label="x^2")
        ax.legend()
        ax.grid(True, which="both")

        plt.show()

    return speeds_rads, torques


def torque_from_aero_data(prop_result_path = 'app/results/dalprop5045.prop' , plot=False):

    full_path = Path(prop_result_path).resolve()
    candidates = list(full_path.glob("aero_*"))
    if not candidates:
        print("Warning: No aero data found for", prop_result_path)
        return None

    aerof = max(candidates, key=lambda f: f.stat().st_mtime)

    aero_data = np.load(aerof)

    tcal_data, qcal_data = load_cell_calibration()

    mean_speed, _, mean_torque = calc_mean_forces(
        aero_data, tcal_data, qcal_data
    )

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        ax.plot(mean_speed, mean_torque, "o", label="dcurrent")
        ax.set_xscale("log")
        ax.set_yscale("log")

        xlo, xhi = ax.get_xlim()
        x = np.linspace(xlo, xhi, 100)
        ax.plot(x, 1e-8 * x**2, label="x^2")
        ax.legend()
        ax.grid(True, which="both")

        plt.show()

    return mean_speed, mean_torque


def compare_torque(prop_result_path):
    
    msp, mtq = torque_from_motor_data(
        plot=False, prop_result_path=prop_result_path)
    asp, atq = torque_from_aero_data(
        plot=False, prop_result_path=prop_result_path)
    
    print(mtq, atq)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.plot(msp, mtq, "o", label="Current measurement")
    ax.plot(asp, atq, "o", label="Load cell measurement")
    
    ax.set_xscale("log")
    ax.set_yscale("log")

    xlo, xhi = ax.get_xlim()
    x = np.linspace(xlo, xhi, 100)
    k = mtq[0] / msp[0]**2

    ax.plot(x, k * x**2, label="x^2")
    ax.legend()
    ax.grid(True, which="both")

    ax.set_xlabel("Speed [rad/s]")
    ax.set_ylabel("Torque [Nm]")

    return fig, ax


if __name__ == "__main__":

    fig, ax = compare_torque(
        'app/results/dalprop5045bnr.prop'
    )

    fig.savefig(
        'deliverables/final_report/figures/controller_vs_loadcell_torque.png',
        dpi=300
    )

    plt.show()

