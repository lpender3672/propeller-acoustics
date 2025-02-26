import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile('app/results/audio.bin', dtype=np.float64)
total_channels = 1
freq = 54231
data = data.reshape(-1, total_channels)




#data_ft = np.fft.rfft(data, axis=0)
#data_freq = np.fft.rfftfreq(data.shape[0], d=1/freq)

fig, ax = plt.subplots(total_channels, 1, figsize=(10, 12))
if not isinstance(ax, np.ndarray):
    ax = np.array([ax])

for i in range(total_channels):
    channel_data = data[:, i]

    window = np.hanning(len(channel_data))
    windowed_data = channel_data * window

    zero_padding = 2**np.ceil(np.log2(len(windowed_data)) + 1).astype(int)
    data_ft = np.fft.rfft(windowed_data, n=zero_padding)
    data_freq = np.fft.rfftfreq(zero_padding, d=1/freq)

    magnitude = np.abs(data_ft)
    magnitude = np.maximum(magnitude, 1e-10)
    magnitude_db = 20 * np.log10(magnitude)
    
    ax[i].plot(data_freq, magnitude_db)
    ax[i].set_title(f'Channel {i+1}')
    ax[i].set_xlabel('Frequency (Hz)')
    ax[i].set_ylabel('Magnitude (dB)')
    ax[i].grid(True)
    ax[i].set_ylim(0, 50)

plt.tight_layout()
plt.show()

print(data.shape)