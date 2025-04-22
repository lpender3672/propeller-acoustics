import numpy as np
import sounddevice as sd
from pathlib import Path
import os

from audio import (
    load_meta_data,
    butter_filt,
)

total_channels = 7
freq = 51200  # Hz

def listen_to_microphone(prop_result_path, mic_idx, speed = 1000):
    """
    Plays the audio from a specific microphone at a specific speed.

    Parameters:
    -----------
    prop_result_path : str
        Path to the data for propeller acoustics
    mic_idx : int
        Index of the microphone to play (0-6)
    speed : float
        Speed of test to play (RPM)
    """

    meta, _ = load_meta_data(prop_result_path)
    
    # get closest matching speed
    row = meta[np.argmin(np.abs(meta[:, 1] + speed))]
    audiof = Path(row[0])
    actual_speed = -row[1] * 2 * np.pi / 60 # rads/s
    relative_audiof = audiof.relative_to(audiof.parent.parent.parent.parent)
    print(relative_audiof)
    if os.path.exists(relative_audiof):
        print(f"Playing audio for speed {actual_speed:.2f} rad/s")

    try:
        data = np.fromfile(relative_audiof, dtype=np.float64).reshape(-1, total_channels)
        filtered = butter_filt(data[:, mic_idx], actual_speed, freq)
        normalised = filtered / np.max(np.abs(filtered))
        padded = np.concatenate([np.zeros(5120), normalised, np.zeros(5120)])
        sd.play(padded, samplerate=freq)
        sd.wait()
        
    except FileNotFoundError:
        print(f"Warning: File not found {audiof}, skipping...")


if __name__ == "__main__":
    listen_to_microphone('app/results/dalprop5045.prop', 1, 17000)
    