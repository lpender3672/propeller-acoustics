


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from app.routines_audio import (
    parse_lookup_df,
    normalized_hanning,
    load_microphone_calibration,
    apply_calib_freq
)

def plot_audio_psd(audio_path, speed, ax, channel = 3, **kwargs):
    calib_data = load_microphone_calibration()

    audio_data = np.fromfile(audio_path, dtype=np.float64).reshape(-1, 7)

    hanning_window = normalized_hanning(audio_data.shape[0])

    windowed_data = audio_data * hanning_window
    zero_padding = 2 ** np.ceil(np.log2(windowed_data.shape[0]) + 1).astype(int)

    ft_data = np.fft.rfft(
        windowed_data, n=zero_padding, axis=0
    )
    freqs = np.fft.rfftfreq(zero_padding, d=1/51200)

    ft_data = apply_calib_freq(ft_data, freqs, calib_data)

    psd = np.abs(ft_data[:, channel])**2 / (zero_padding * 51200)
    psddB = 10 * np.log10(psd / (20e-6)**2)

    ax.plot(freqs / speed, psddB, **kwargs)

    return ax

def plot_motor_comparison_spectrums():

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    plot_audio_psd('app/results/floor/F_ambient.bin', 
                   200, axes[0], color='C1', alpha = 0.8, label='Ambient noise')
    plot_audio_psd('app/results/floor/F_motor12.bin', 
                   200, axes[0], color='C2', alpha = 0.5, label='Motor noise')

    plot_audio_psd('app/results/floor/F_motor12.bin', 
                   200, axes[1], color='C2', alpha = 0.8, label='Motor noise')
    plot_audio_psd('app/results/floor/F_5045_12.bin', 
                   200, axes[1], color='C3', alpha = 0.5, label='Propeller noise')

    axes[0].set_xlim(0, 10)
    axes[1].set_xlim(0, 10)

    axes[0].set_ylabel('$SPL [dB]$')
    axes[1].set_ylabel('$SPL [dB]$')

    axes[0].set_xlabel('$\omega / \Omega$ [-]')
    axes[1].set_xlabel('$\omega / \Omega$ [-]')

    axes[0].grid(True, which='both')
    axes[1].grid(True, which='both')

    axes[0].legend()
    axes[1].legend()

    fig.savefig(
        "deliverables/final_report/figures/spectrum_plot.png", dpi=300
    )


def plot_stream_comparison_spectrums():

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    plot_audio_psd('app/results/floor/F_motor12.bin', 
                   200, ax, color='C1', alpha = 0.8, label=r'$\theta=135^\circ$')
    plot_audio_psd('app/results/floor/F_motor12.bin',
                   200, ax, channel=5, color='C2', alpha = 0.5, label=r'$\theta=180^\circ$')

    ax.set_xlim(0, 10)
    ax.set_ylabel('$SPL [dB]$')
    ax.set_xlabel('$\omega / \Omega$ [-]')
    ax.grid(True, which='both')

    ax.legend()

    fig.savefig(
        "deliverables/final_report/figures/stream_spectrum_plot.png", dpi=300
    )


if __name__ == "__main__":

    plot_motor_comparison_spectrums()
    plot_stream_comparison_spectrums()

    plt.show()

