

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from app.routines import (
    load_prop_from_file
)

# ok so open aero data of dalprop5045

def get_prop_speed_time_data(
    prop_result_path,
    speed_index
    ):
    """
    Plot thrust and torque coefficients vs. speed with error bars.
    """
    full_path = Path(prop_result_path).resolve()

    candidates = list(full_path.glob("aero_*"))
    if not candidates:
        print("Warning: No aero data found for", prop_result_path)
        return None
    faero = max(candidates, key=lambda f: f.stat().st_mtime)

    aero_data = np.load(faero)

    motor_data = aero_data["motor_data"]

    time_data = motor_data[speed_index, :, 0]
    speed_data = motor_data[speed_index, :, 1] * 2 * np.pi / 60

    return time_data, speed_data


def plot_rfft_speed(
    prop_result_path,
    speed_index,
    ax=None,
    label=None,
):
    time_data, speed_data = get_prop_speed_time_data(prop_result_path, speed_index)

    hanning = np.hanning(len(time_data))
    hanning = hanning / np.mean(hanning)

    time_data = time_data - time_data[0]
    time_data = time_data * hanning

    ft_data = np.fft.rfft(time_data)
    frequency = 100 # Hz

    freq_data = np.fft.rfftfreq(len(time_data), 1 / 100)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(freq_data, np.abs(ft_data), label=label)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

def plot_prop_freq():

    prop_result_path = Path("app/results/dalprop5045.prop/")
    speed_index = 1
    # identified a problem with the speed index

    time_data, speed_data = get_prop_speed_time_data(prop_result_path, speed_index)
    
    print(time_data.shape)

    plt.plot(time_data, speed_data)

    plot_rfft_speed(prop_result_path, speed_index)

    plt.show()


def specific_test_speed_time_data(fpath):
    # this loads the the specific cogging test data
    fpath = Path(fpath).resolve()

    motor_data = np.load(fpath)
    time_data = motor_data[:, 0]
    speed_data = motor_data[:, 1] * 2 * np.pi / 60

    #
    sample_freq = np.mean( 1 / np.diff(time_data) )
    #sample_freq = np.max( 1 / np.diff(time_data))


    print("Average sample frequency: ", sample_freq)

    ft_data = np.fft.rfft(speed_data)
    freq = np.fft.rfftfreq(len(speed_data), 1 / sample_freq)

    fig, ax = plt.subplots()
    ax.plot(time_data, speed_data)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (rad/s)")
    ax.set_title(fpath.name)


    fig, ax = plt.subplots()
    ax.plot(freq, np.abs(ft_data))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(0, 600)
    ax.set_title(fpath.name)

if __name__ == "__main__":

    specific_test_speed_time_data("app/results/cogging/0speed.npy")
    specific_test_speed_time_data("app/results/cogging/3000speed.npy")
    specific_test_speed_time_data("app/results/cogging/6000speed.npy")
    specific_test_speed_time_data("app/results/cogging/12000speed.npy")

    plt.show()
