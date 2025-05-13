

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from app.routines import (
    load_meta_data
)

from app.routines_audio import (
    load_and_compute_rfft,
    integrate_harmonics,
    load_calibration_data,
    apply_calib
)


view_channels = [0, 1, 5]


def plot_harmonics(prop_result_path, max_harmonic=20, ax=None, mcalib_data=None):

    if mcalib_data is None:
        mcalib_data = load_calibration_data()

    if ax is None:
        fig, ax = plt.subplots()

    meta, prop = load_meta_data(prop_result_path)
    speed = np.abs(meta[:, 1].astype(float))  # RPM
    bpfs = speed * prop["B"] / 60  # blade passes / s

    data_freq, ft_data = load_and_compute_rfft(prop_result_path)

    ft_data = apply_calib(data_freq, ft_data, mcalib_data)

    integrated_harmonics = integrate_harmonics(data_freq, ft_data, bpfs)

    # start off with plotting harmonics at single speed for a few microphones
    hmnics = np.arange(0, max_harmonic)

    for i, idx in enumerate(view_channels):
        channel_harmonics = integrated_harmonics[0][:max_harmonic, idx].astype(float)
        channel_harmonics_dB = 10 * np.log10(channel_harmonics)
        label = f"Channel {idx + 1} " + r"$\theta =" + f"{microphone_positions[idx]}$"
        ax.plot(hmnics, channel_harmonics_dB, label=label)

    ax.set_xlabel("Harmonic")
    ax.set_ylabel("Power spectral density (dB)")
    ax.legend()
    ax.grid(True)


def plot_radar_harmonics(prop_result_path, harmonics_of_interest=(1, 2, 3), ax=None, mcalib_data=None):
    """
    Plots a half-plane 'radar'-style graph of the integrated harmonics
    for the first speed (index=0), across the 7 microphone positions.

    Parameters:
    -----------
    prop_result_path : str
        Path to the data for propeller acoustics
    harmonics_of_interest : tuple/list of int
        Which harmonic indices (e.g., 1,2,3,...) to plot in radar form
    """

    if mcalib_data is None:
        mcalib_data = load_calibration_data()

    if ax is None:
        fig, ax = plt.subplots()

    # -- Load data as before --
    meta, prop = load_meta_data(prop_result_path)
    speed = np.abs(meta[:, 1].astype(float))  # RPM
    bpfs = speed * prop["B"] / 60  # blade passes / s

    data_freq, ft_data = load_and_compute_rfft(prop_result_path)
    ft_data = apply_calib(data_freq, ft_data, mcalib_data)

    # -- Integrate harmonics for all speeds --
    integrated_harmonics = integrate_harmonics(data_freq, ft_data, bpfs)
    # integrated_harmonics[0] is the integrated data (n_harmonics, n_channels)
    # for speed index=0

    # We have 7 "view channels" at angles in microphone_positions (in degrees):
    # e.g. microphone_positions = [0, 30, 60, 90, 120, 150, 180]
    angles_deg = microphone_positions[:6]  # first 6 channels
    angles_rad = np.deg2rad(angles_deg)

    # Create a polar subplot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # For each harmonic index in harmonics_of_interest
    for h_idx in harmonics_of_interest:
        # integrated_harmonics[0][h_idx, :] gives us the amplitude at that harmonic across channels
        # Slicing [:6] to get the first 6 channels
        # the 7th is a duplicate angle
        channel_amps = integrated_harmonics[0][h_idx, :6].astype(float)
        channel_amps_dB = 10 * np.log10(channel_amps)

        # Plot a line (radar style). The radial distance is channel_amps_dB
        # The angle is angles_rad.
        ax.plot(angles_rad, channel_amps_dB, label=f"Harmonic {h_idx}")

        # Optionally, fill the area for a more "radar" look:
        # ax.fill(angles_rad, channel_amps_dB, alpha=0.3)

    # Restrict to half-plane: 0° <= theta <= 180°
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    ax.set_rlabel_position(90)  # Move radial labels off to the side
    ax.set_title("Radar Plot of Selected Harmonics (Half-Plane)")
    ax.legend(loc="upper right")




def plot_prop_harmonics(folder, ax=None, mcalib_data=None):

    if mcalib_data is None:
        mcalib_data = load_calibration_data()

    if ax is None:
        fig, ax = plt.subplots()

    freq, ft = load_and_compute_rfft(folder)

    ft = apply_calib(freq, ft, mcalib_data)


def plot_raw(data_freq, ft_data, ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(len(view_channels), 1, sharex=True, figsize=(10, 10))

    for i, idx in enumerate(view_channels):

        magnitude = np.abs(ft_data[0, :, idx])
        magnitude = np.maximum(magnitude, 1e-10)
        magnitude_db = 20 * np.log10(magnitude)
        ax[i].plot(data_freq, magnitude_db, **kwargs)
        ax[i].set_title(f"Channel {idx + 1}")
        ax[i].set_xlabel("Frequency (Hz)")
        ax[i].set_ylabel("Magnitude (dB)")
        ax[i].grid(True)
        ax[i].set_ylim(-20, 60)
        ax[i].set_xlim(0, 1000)

    return ax



if __name__ == "__main__":

    
    data_freq, ft_data = load_and_compute_rfft("app/results/dalprop5045.prop")

    mcalib_data = load_calibration_data()

    fig, ax = plt.subplots(len(view_channels), 1, sharex=True, figsize=(10, 10))
    plot_raw(data_freq, ft_data, ax, label="uncalibrated")
    cal_ft = apply_calib(data_freq, ft_data, mcalib_data)
    plot_raw(data_freq, cal_ft, ax, label="calibrated")

    plot_harmonics("app/results/dalprop5045.prop", mcalib_data=mcalib_data)
    plot_radar_harmonics(
        "app/results/dalprop5045.prop", harmonics_of_interest=list(range(1, 10)), mcalib_data=mcalib_data
    )

    plt.show()

    # plot_prop_sound(twin_data, ax, label='Twin', alpha=0.9)
    # plot_prop_sound(tri_data, ax, label='Tri', alpha=0.9, linestyle='--')
    # plot_prop_sound(loop_data, ax, label='Loop', alpha=0.9, linestyle='-.')
    # plot_prop_sound(naca0024_data, ax, label='NACA 0024', alpha=0.9)

    # plot_prop_sound(tri_data, ax, label='Tri 6000 RPM', alpha=0.9, linestyle='--')

    # plot_prop_sound(tri12_v, ax, label='Tri 12kRPM', alpha=0.9, linestyle='-')
    # plot_prop_sound(tri12_av, ax, label='Tri 12kRPM antivibration', alpha=0.9, linestyle='-')

    # plot_prop_sound(tri12, ax, label='Tri 12000 RPM', alpha=0.9, linestyle='-')
    # plot_prop_sound(tri12_shrouded, ax, label='Tri 12000 RPM shrouded', alpha=0.9, linestyle='--')
    # plot_prop_sound(motor, ax, label='Motor 12000 RPM', alpha=0.9, linestyle='-.')
    # plot_prop_sound(motor_shrouded, ax, label='Motor shrouded 12000 RPM', alpha=0.9, linestyle='--')
