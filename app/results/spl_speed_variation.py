
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from app.routines import (
    load_meta_data
)
from app.routines_audio import (
    load_microphone_calibration,
    rms_butter,
)

FREQUENCY = 51200  # Hz
TOTAL_CHANNELS = 7

def rmsndp_speed(prop_results_path, microphone_state, mic_idx, ax=None):
    # TODO: convert this function to use the df based routines
    
    meta, prop = load_meta_data(prop_results_path)

    # filter meta files by mic file matching mic_bds and mic_angle
    # the mic files are in meta[:, 4]

    speeds = np.array([])
    rmses = np.array([])

    for row in meta:
        mic_path = Path(row[4])
        mic_path = mic_path.relative_to(mic_path.parent.parent.parent.parent)
        if not (microphone_state in mic_path.name):
            continue

        correct_mic_path = mic_path.parent / microphone_state

        audiof = Path(row[0])
        try:
            relative_audiof = audiof.relative_to(audiof.parent.parent.parent.parent)
            data = np.fromfile(relative_audiof, dtype=np.float64).reshape(
                -1, TOTAL_CHANNELS
            )
        except FileNotFoundError:
            print(f"Warning: File not found {relative_audiof}, skipping...")
            continue

        speed = np.abs(row[1].astype(float))  # RPM
        speed = speed * 2 * np.pi / 60
        speed = np.abs(speed)  # rad / s

        if speed < 10:
            continue

        data = data[:, mic_idx]
        rms = rms_butter(data, speed, FREQUENCY)

        speeds = np.append(speeds, speed)
        rmses = np.append(rmses, rms)
        
    # plot now
    if ax is None:
        fig, ax = plt.subplots()

    prop_name = Path(prop_results_path).name

    microphone_positions = np.loadtxt(correct_mic_path, delimiter=",", skiprows=1, dtype='object')
    label_str = f"{prop_name} Mic {mic_idx + 1} " + r"$\theta =" + f"{microphone_positions[mic_idx, 2]}$"

    ax.loglog(speeds, rmses, "o-", label=label_str)
    ax.set_xlabel("Speed (rad/s)")
    ax.set_ylabel("RMS (V)")

    ax.grid(True, which="both")

    return ax


def plot_rmsndp_speed_nonlinearity():
    # this is incredible!!!
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for i in [5]:
        rmsndp_speed("app/results/dalprop5045.prop", microphone_state="gantry67.5_8bd.csv", mic_idx=i, ax=ax)
    for i in [5]:
        rmsndp_speed("app/results/dalprop5045_nonlinear_test.prop", microphone_state="gantry67.5_8bd.csv", mic_idx=i, ax=ax)
    for i in [5]:
        rmsndp_speed("app/results/dalprop5045_nonlinear_test2.prop", microphone_state="gantry67.5_8bd.csv", mic_idx=i, ax=ax)

    # get lower and upper limits of the y axis
    xmin = np.min(ax.get_xlim())
    xmax = np.max(ax.get_xlim())

    x = np.linspace(xmin, xmax, 1000)

    ax.plot(x, 1e-5 * x**2, "k--", label="x^2")

    ax.legend()

def plot_rmsndp_speed_sweeps():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for i in [0, 1, 3, 5]:
        rmsndp_speed("app/results/dalprop5045.prop", microphone_state="gantry45_8bd.csv", mic_idx=i, ax=ax)
    

    xmin = np.min(ax.get_xlim())
    xmax = np.max(ax.get_xlim())

    x = np.linspace(xmin, xmax, 1000)

    ax.plot(x, 1e-6 * x**2, "k--", label="x^2")

    ax.legend()


if __name__ == "__main__":
    #plot_rmsndp_speed_nonlinearity()
    plot_rmsndp_speed_sweeps()
    plt.show()

