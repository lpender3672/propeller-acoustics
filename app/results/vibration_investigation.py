
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from app.routines_audio import (
    parse_lookup_df,
    normalized_hanning
)


def plot_vibration_investigation(lookup_path, speed=1000):

    speed_to_find = 4250
    
    lookup_df = parse_lookup_df(lookup_path)

    # get closest matching speed
    row = lookup_df.iloc[(-lookup_df['speed'] - speed_to_find).abs().argsort()[:1]]

    audio_path = row['audio_path'].values[0]
    speed = row['speed'].values[0] * - 1 / 60
    print(f"Using audio file: {audio_path}")
    audio_data = np.fromfile(audio_path, dtype=np.float64).reshape(-1, 7)

    hanning_window = normalized_hanning(audio_data.shape[0])

    windowed_data = audio_data * hanning_window
    zero_padding = 2 ** np.ceil(np.log2(windowed_data.shape[0]) + 1).astype(int)

    ft_data = np.fft.rfft(
        windowed_data, n=zero_padding, axis=0
    )
    freqs = np.fft.rfftfreq(zero_padding, d=1/51200) / speed

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    plot_channel = 0 

    ax.plot(freqs, np.abs(ft_data[:, plot_channel]), label=f'Channel {plot_channel}')
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude')
    

if __name__ == "__main__":
    # Example usage
    lookup_path = 'app/results'

    plot_vibration_investigation(lookup_path)

    plt.show()

