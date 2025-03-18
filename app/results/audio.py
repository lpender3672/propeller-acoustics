import numpy as np
import matplotlib.pyplot as plt

total_channels = 7
freq = 51200

view_channels = [1,5]

#data_ft = np.fft.rfft(data, axis=0)
#data_freq = np.fft.rfftfreq(data.shape[0], d=1/freq)

def plot_prop_sound(data, ax, **kwargs):

    for i, idx in enumerate(view_channels):
        channel_data = data[:, idx]

        window = np.hanning(len(channel_data))
        windowed_data = channel_data * window

        zero_padding = 2**np.ceil(np.log2(len(windowed_data)) + 1).astype(int)
        data_ft = np.fft.rfft(windowed_data, n=zero_padding)
        data_freq = np.fft.rfftfreq(zero_padding, d=1/freq)

        magnitude = np.abs(data_ft)
        magnitude = np.maximum(magnitude, 1e-10)
        magnitude_db = 20 * np.log10(magnitude) 
        ax[i].plot(data_freq, magnitude_db, **kwargs)
        ax[i].set_title(f'Channel {idx+1}')
        ax[i].set_xlabel('Frequency (Hz)')
        ax[i].set_ylabel('Magnitude (dB)')
        ax[i].grid(True)
        ax[i].set_ylim(0, 80)
        ax[i].set_xlim(0, 1000)

twin_data = np.fromfile('app/results/NF_twin.bin', dtype=np.float64).reshape(-1, total_channels)
tri_data = np.fromfile('app/results/NF_tri.bin', dtype=np.float64).reshape(-1, total_channels)
loop_data = np.fromfile('app/results/NF_loop.bin', dtype=np.float64).reshape(-1, total_channels)
naca0024_data = np.fromfile('app/results/NF_0024.bin', dtype=np.float64).reshape(-1, total_channels)

tri12 = np.fromfile('app/results/NF_tri12.bin', dtype=np.float64).reshape(-1, total_channels)
tri12_shrouded = np.fromfile('app/results/NF_tri12_shrouded.bin', dtype=np.float64).reshape(-1, total_channels)
motor = np.fromfile('app/results/NF_motor12.bin', dtype=np.float64).reshape(-1, total_channels)
motor_shrouded = np.fromfile('app/results/NF_motor12_shrouded.bin', dtype=np.float64).reshape(-1, total_channels)

tri12_av = np.fromfile('app/results/NF_AV_tri12.bin', dtype=np.float64).reshape(-1, total_channels)
tri12_v = np.fromfile('app/results/NF_V_tri12.bin', dtype=np.float64).reshape(-1, total_channels)

fig, ax = plt.subplots(len(view_channels), 1, figsize=(10, 12))
if not isinstance(ax, np.ndarray):
    ax = np.array([ax])

#plot_prop_sound(twin_data, ax, label='Twin', alpha=0.9)
#plot_prop_sound(tri_data, ax, label='Tri', alpha=0.9, linestyle='--')
#plot_prop_sound(loop_data, ax, label='Loop', alpha=0.9, linestyle='-.')
#plot_prop_sound(naca0024_data, ax, label='NACA 0024', alpha=0.9)

#plot_prop_sound(tri_data, ax, label='Tri 6000 RPM', alpha=0.9, linestyle='--')

plot_prop_sound(tri12_v, ax, label='Tri 12kRPM', alpha=0.9, linestyle='-')
plot_prop_sound(tri12_av, ax, label='Tri 12kRPM antivibration', alpha=0.9, linestyle='-')

#plot_prop_sound(tri12, ax, label='Tri 12000 RPM', alpha=0.9, linestyle='-')
#plot_prop_sound(tri12_shrouded, ax, label='Tri 12000 RPM shrouded', alpha=0.9, linestyle='--')
#plot_prop_sound(motor, ax, label='Motor 12000 RPM', alpha=0.9, linestyle='-.')
#plot_prop_sound(motor_shrouded, ax, label='Motor shrouded 12000 RPM', alpha=0.9, linestyle='--')

ax[-1].legend()

plt.tight_layout()
plt.show()
