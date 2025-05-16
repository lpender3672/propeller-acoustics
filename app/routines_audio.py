import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, find_peaks

from app.routines import (
    load_prop_from_file,
    load_meta_data
)

TOTAL_CHANNELS = 7
FREQUENCY = 51200

def load_microphone_calibration():

    mcalibf = "practical/microphones.xlsx"
    mcalib_data = pd.read_excel(mcalibf, sheet_name="calibration", skiprows=2).to_numpy(
        dtype=float
    )
    return mcalib_data


def butter_filt(data, speed, freq):
    b, a = butter(
        1, [0.1 * speed / freq, 10 * speed / freq], btype="bandpass", analog=False
    )
    data_fltrd = filtfilt(b, a, data)
    return data_fltrd


def rms_butter(data, speed, freq):
    b, a = butter(
        1, [0.1 * speed / freq, 10 * speed / freq], btype="bandpass", analog=False
    )
    data_fltrd = filtfilt(b, a, data)
    rms = np.sqrt(np.mean(data_fltrd**2))
    return rms



def load_and_compute_rfft(prop_result_path):

    meta, _ = load_meta_data(prop_result_path)

    # 51200 Hz
    freq_cutoff_idx = 56000

    ft_data = np.zeros((len(meta), freq_cutoff_idx, TOTAL_CHANNELS))

    for i, row in enumerate(meta):
        # do for all speeds
        audiof = Path(row[0])
        try:
            relative_audiof = audiof.relative_to(audiof.parent.parent.parent.parent)
            data = np.fromfile(relative_audiof, dtype=np.float64).reshape(
                -1, TOTAL_CHANNELS
            )
        except FileNotFoundError:
            print(f"Warning: File not found {relative_audiof}, skipping...")
            continue

        window = np.hanning(data.shape[0]).reshape(-1, 1)
        windowed_data = data * window

        zero_padding = 2 ** np.ceil(np.log2(windowed_data.shape[0]) + 1).astype(int)

        ft_data[i, :freq_cutoff_idx, :] = np.fft.rfft(
            windowed_data, n=zero_padding, axis=0
        )[:freq_cutoff_idx, :]

    data_freq = np.fft.rfftfreq(zero_padding, d=1 / FREQUENCY)[:freq_cutoff_idx]

    return data_freq, ft_data


def apply_calib(data_freq, ft_data, calib_data):

    calib_freqs = calib_data[:, 0::2].astype(float)
    calib_data_dB = calib_data[:, 1::2].astype(float)
    calib_data = 10 ** (calib_data_dB / 20)

    assert ft_data.shape[2] == calib_data.shape[1]

    for i in range(ft_data.shape[2]):
        # we not in dB so division

        ft_data[:, :, i] = ft_data[:, :, i] / (
            1 + np.interp(data_freq, calib_freqs[:, i], calib_data[:, i])
        )
        # so this will apply the min calib value for freq below the min calib freq
        # and the max calib value for freq above the max calib freq

    return ft_data


def integrate_harmonics(freq, ft_data, bpfs, aband=10):
    # bpfs must match the speeds

    assert ft_data.shape[0] == bpfs.shape[0]

    integrated_list = []

    for i in range(ft_data.shape[0]):  # speeds

        if np.isclose(bpfs[i], 0):
            continue

        harmonics = np.arange(bpfs[i], freq[-1], bpfs[i])

        intergrated_harmonics = np.zeros((harmonics.shape[0], ft_data.shape[2]))

        for j in range(harmonics.shape[0]):

            mask = (freq >= harmonics[j] - aband) & (freq <= harmonics[j] + aband)
            fqharm = freq[mask]
            ftharm = ft_data[i, mask, :]

            # trapz power spectral density
            intergrated_harmonics[j, :] = np.trapezoid(ftharm**2, fqharm, axis=0)

        integrated_list.append(intergrated_harmonics)

    return np.array(integrated_list, dtype=object)

