
import numpy as np

from routines import *

from app.results.audio import (
    load_meta_data,
    load_and_compute_rfft,
)

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pandas as pd


tcal_data = np.load('app/bcal_thrust.npy')
qcal_data = np.load('app/bcal_torque.npy')


def load_aer_data(prop_result_path):

    full_path = Path(prop_result_path).resolve()
    fprop = full_path.parent.parent / "props" / full_path.name
    prop = load_prop_from_file(fprop)

    candidates = list(full_path.glob("aero_*"))
    if not candidates:
        print("Warning: No aero data found for", prop_result_path)
        return None
    
    faero = max(candidates, key=lambda f: f.stat().st_mtime)

    aero_data = np.load(faero)

    return aero_data, prop


def butter_filt(data, speed, freq):
    b, a = butter(1, [0.1*speed / freq, 10*speed / freq], btype='bandpass', analog=False)
    data_fltrd = filtfilt(b, a, data)
    rms = np.sqrt(np.mean(data_fltrd**2))
    return rms

def calculate_reference_pressures(prop_result_path):

    aero_data, prop = load_aer_data(prop_result_path)

    force_data = aero_data['force_data']
    motor_data = aero_data['motor_data']

    avg_speed = np.mean(motor_data[:,:,1], axis=1) * 2 * np.pi / 60
    avg_speed = np.abs(avg_speed)
    avg_raw_forces = np.mean(force_data[:,:,1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:,0], tcal_data[:,0,0], tcal_data[:,1,0])
    avg_torque = np.interp(avg_raw_forces[:,1], qcal_data[:,0,1], qcal_data[:,1,1])

    filtered_speed = avg_speed[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]
    filtered_thrust = avg_thrust[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]
    filtered_torque = avg_torque[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]


    meta, _ = load_meta_data(prop_result_path)

    total_channels = 7
    freq = 51200 # hz

    cutoff_freq = 50

    # 51200 Hz
    freq_cutoff_idx = 56000

    ft_data = np.zeros((len(meta), freq_cutoff_idx, total_channels))


    speeds = []
    rmses = []

    for i,row in enumerate(meta):
        # do for all speeds
        audiof = Path(row[0])
        relative_audiof = rebase_path(audiof)

        try:
            data = np.fromfile(relative_audiof, dtype=np.float64).reshape(-1, total_channels)
        except FileNotFoundError:
            continue

        speed = row[1] / -60
        speed = max(abs(speed), 10)
        rmses.append(butter_filt(data[:, 5], speed, freq))
        speeds.append(speed)

    rmses = np.array(rmses)
    speeds = np.array(speeds)

    a,b = np.polyfit(np.log(speeds[speeds > 10]), np.log(rmses[speeds > 10]), 1)
    
    print("Slope:", a)
    print("Intercept:", b)
    
    plt.loglog(speeds, rmses, 'o')
    plt.loglog(speeds, np.exp(b) * speeds**a, label="Fit")  
    plt.grid(which='both')
    plt.xlabel("Speed (RPM)")
    plt.ylabel("RMS")
    plt.legend()
    plt.show()


def rebase_path(path):

    if isinstance(path, str):
        path = Path(path)
        
    ans = path.relative_to(path.parent.parent.parent.parent)
    return str(ans)

def rebase_pathlist(pathlist):
    return [rebase_path(path) for path in pathlist]


def parse_lookup_df(results_folder):

    # Base directory
    base_dir = Path(results_folder)

    df_list = []

    for meta_path in base_dir.glob("**/*.prop/meta_data.npy"):
        try:

            prop_name = meta_path.parent.name #.replace(".prop", "")

            meta_array = np.load(meta_path, allow_pickle=True)

            df = pd.DataFrame()

            prop_path = base_dir.parent / "props" / prop_name

            df['audio_path'] = rebase_pathlist(meta_array[:, 0])
            df['speed'] = meta_array[:, 1]
            df['current'] = meta_array[:, 2]
            df['temperature'] = meta_array[:, 3]
            df['prop_path'] = prop_path
            df['mic_path'] = rebase_pathlist(meta_array[:, 4])

            df_list.append(df)

        except Exception as e:
            print(f"Error processing {meta_path}: {e}")

    if not df_list:
        return pd.DataFrame()
    
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df


def plot_radar_graph(lookup_df):
    grouped = lookup_df.groupby(['prop_path', 'speed'])

    total_channels = 7

    for (prop_path, speed), group in grouped:
        print(f"Processing {prop_path} at speed {speed} RPM...")

        rms_values = []
        angles = []

        speed = speed / -60
        if speed < 10:
            continue
        
        for _, row in group.iterrows():
            mic_path = row['mic_path']
            prop_path = row['prop_path']
            
            mic_data = np.genfromtxt(mic_path, delimiter=',', dtype=None, encoding=None)
            if not os.path.exists(row['audio_path']):
                continue

            audio_data = np.fromfile(row['audio_path'], dtype=np.float64).reshape(-1, total_channels)

            for i in range(total_channels):

                rms_values.append(butter_filt(audio_data[:, i], speed, 51200))
                angles.append(mic_data[i, 2])

        # plot radar graph


df = parse_lookup_df('app/results/')

plot_radar_graph(df)

#calculate_reference_pressures('app/results/dalprop5045.prop')


