import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from pathlib import Path

import pandas as pd

from routines import (
    load_prop_from_file
)

total_channels = 7
freq = 51200

view_channels = [0,1,5]

microphone_positions = np.array([
    0, 90, 135, 150, 165, 180, 90 ])

mcalibf = 'practical/microphones.xlsx'
mcalib_data = pd.read_excel(mcalibf, sheet_name='calibration', skiprows=2).to_numpy(dtype=float)


def load_meta_data(prop_result_path):

    abspath = Path(prop_result_path).resolve()
    meta = np.load(abspath / 'meta_data.npy', allow_pickle=True)

    propf = abspath.parent.parent / 'props' / abspath.name
    prop = load_prop_from_file(propf)

    return meta, prop

#mdat, _ = load_meta_data('app/results/c.prop')
#print(mdat)

def load_and_compute_rfft(prop_result_path):

    meta, _ = load_meta_data(prop_result_path)

    # 51200 Hz
    freq_cutoff_idx = 56000

    ft_data = np.zeros((len(meta), freq_cutoff_idx, total_channels))

    for i,row in enumerate(meta):
        # do for all speeds
        audiof = Path(row[0])
        data = np.fromfile(audiof, dtype=np.float64).reshape(-1, total_channels)

        window = np.hanning(data.shape[0]).reshape(-1, 1)
        windowed_data = data * window

        zero_padding = 2**np.ceil(np.log2(windowed_data.shape[0]) + 1).astype(int)
        
        ft_data[i, :freq_cutoff_idx, :] = np.fft.rfft(windowed_data, n=zero_padding, axis=0)[:freq_cutoff_idx, :]
    
    data_freq = np.fft.rfftfreq(zero_padding, d=1/freq)[:freq_cutoff_idx]

    return data_freq, ft_data


def apply_calib(data_freq, ft_data, calib_data):
    
    calib_freqs = calib_data[:, 0::2].astype(float)
    calib_data_dB = calib_data[:, 1::2].astype(float)
    calib_data = 10**(calib_data_dB / 20)

    assert ft_data.shape[2] == calib_data.shape[1]

    for i in range(ft_data.shape[2]):
        # we not in dB so division

        ft_data[:,:,i] = ft_data[:,:,i] / (1 + np.interp(data_freq, calib_freqs[:,i], calib_data[:,i]))
        # so this will apply the min calib value for freq below the min calib freq
        # and the max calib value for freq above the max calib freq

    return ft_data


def integrate_harmonics(freq, ft_data, bpfs, aband = 10):
    # bpfs must match the speeds

    assert ft_data.shape[0] == bpfs.shape[0]

    integrated_list = []

    for i in range(ft_data.shape[0]): # speeds

        harmonics = np.arange(bpfs[i], freq[-1], bpfs[i])

        intergrated_harmonics = np.zeros((harmonics.shape[0], ft_data.shape[2]))

        for j in range(harmonics.shape[0]):

            mask = (freq >= harmonics[j] - aband) & (freq <= harmonics[j] + aband)
            fqharm = freq[mask]
            ftharm = ft_data[i, mask, :]

            # trapz power spectral density
            intergrated_harmonics[j, :] = np.trapz(np.abs(ftharm), fqharm**2, axis=0)

        integrated_list.append(intergrated_harmonics)

    return np.array(integrated_list, dtype=object)


def plot_harmonics(prop_result_path, max_harmonic = 20):

    fig, ax = plt.subplots()

    meta, prop = load_meta_data(prop_result_path)
    speed = np.abs(meta[:,1].astype(float)) # RPM
    bpfs = speed * prop['B'] / 60 # blade passes / s

    data_freq, ft_data = load_and_compute_rfft(prop_result_path)

    ft_data = apply_calib(data_freq, ft_data, mcalib_data)

    integrated_harmonics = integrate_harmonics(data_freq, ft_data, bpfs)

    # start off with plotting harmonics at single speed for a few microphones
    hmnics = np.arange(0, max_harmonic)

    for i, idx in enumerate(view_channels):
        channel_harmonics = integrated_harmonics[0, :max_harmonic, idx].astype(float)
        channel_harmonics_dB = 10 * np.log10(channel_harmonics)
        label = f'Channel {idx+1} ' + r'$\theta =' + f'{microphone_positions[idx]}$'
        ax.plot(hmnics, channel_harmonics_dB, label=label)

    ax.set_xlabel('Harmonic')
    ax.set_ylabel('Power spectral density (dB)')
    ax.legend()
    ax.grid(True)


def plot_radar_harmonics(prop_result_path, harmonics_of_interest=(1, 2, 3)):
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

    # -- Load data as before --
    meta, prop = load_meta_data(prop_result_path)
    speed = np.abs(meta[:,1].astype(float))      # RPM
    bpfs = speed * prop['B'] / 60                # blade passes / s

    data_freq, ft_data = load_and_compute_rfft(prop_result_path)
    ft_data = apply_calib(data_freq, ft_data, mcalib_data)

    # -- Integrate harmonics for all speeds --
    integrated_harmonics = integrate_harmonics(data_freq, ft_data, bpfs)
    # integrated_harmonics[0] is the integrated data (n_harmonics, n_channels) for speed index=0

    # We have 7 "view channels" at angles in microphone_positions (in degrees):
    # e.g. microphone_positions = [0, 30, 60, 90, 120, 150, 180]
    angles_deg = microphone_positions[:6]  # first 6 channels
    angles_rad = np.deg2rad(angles_deg)

    # Create a polar subplot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # For each harmonic index in harmonics_of_interest
    for h_idx in harmonics_of_interest:
        # integrated_harmonics[0][h_idx, :] gives us the amplitude at that harmonic across channels
        # Slicing [:7] to get the first 7 channels
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
    ax.legend(loc='upper right')
    plt.show()

def plot_prop_harmonics(folder, ax):

    freq, ft = load_and_compute_rfft(folder)

    ft = apply_calib(freq, ft, mcalib_data)    

def plot_raw(data_freq, ft_data, ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(len(view_channels), 1, sharex=True, figsize=(10, 10))

    for i, idx in enumerate(view_channels):

        magnitude = np.abs(ft_data[0,:,idx])
        magnitude = np.maximum(magnitude, 1e-10)
        magnitude_db = 20 * np.log10(magnitude) 
        ax[i].plot(data_freq, magnitude_db, **kwargs)
        ax[i].set_title(f'Channel {idx+1}')
        ax[i].set_xlabel('Frequency (Hz)')
        ax[i].set_ylabel('Magnitude (dB)')
        ax[i].grid(True)
        ax[i].set_ylim(-20, 60)
        ax[i].set_xlim(0, 1000)

    return ax


if __name__ == "__main__":

    data_freq, ft_data = load_and_compute_rfft('app/results/dalprop5045.prop')

    fig, ax = plt.subplots(len(view_channels), 1, sharex=True, figsize=(10, 10))
    plot_raw(data_freq, ft_data, ax, label='uncalibrated')
    cal_ft = apply_calib(data_freq, ft_data, mcalib_data)
    plot_raw(data_freq, cal_ft, ax, label='calibrated')

    plot_harmonics('app/results/dalprop5045.prop')
    plot_radar_harmonics('app/results/dalprop5045.prop', harmonics_of_interest=(1, 5, 10))

    plt.tight_layout()
    plt.show()


    #plot_prop_sound(twin_data, ax, label='Twin', alpha=0.9)
    #plot_prop_sound(tri_data, ax, label='Tri', alpha=0.9, linestyle='--')
    #plot_prop_sound(loop_data, ax, label='Loop', alpha=0.9, linestyle='-.')
    #plot_prop_sound(naca0024_data, ax, label='NACA 0024', alpha=0.9)

    #plot_prop_sound(tri_data, ax, label='Tri 6000 RPM', alpha=0.9, linestyle='--')

    #plot_prop_sound(tri12_v, ax, label='Tri 12kRPM', alpha=0.9, linestyle='-')
    #plot_prop_sound(tri12_av, ax, label='Tri 12kRPM antivibration', alpha=0.9, linestyle='-')

    #plot_prop_sound(tri12, ax, label='Tri 12000 RPM', alpha=0.9, linestyle='-')
    #plot_prop_sound(tri12_shrouded, ax, label='Tri 12000 RPM shrouded', alpha=0.9, linestyle='--')
    #plot_prop_sound(motor, ax, label='Motor 12000 RPM', alpha=0.9, linestyle='-.')
    #plot_prop_sound(motor_shrouded, ax, label='Motor shrouded 12000 RPM', alpha=0.9, linestyle='--')
