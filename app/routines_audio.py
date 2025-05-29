import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import hashlib
import pickle

from scipy.signal import butter, filtfilt, find_peaks

from app.routines import (
    load_prop_from_file,
)


CACHE_DIR = Path("app/results/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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



def normalized_hanning(size):
    """
    Returns a normalized hanning window
    """

    hanning_window = np.hanning(size).reshape(-1, 1)
    # energy normalization of the window
    hanning_window /= np.sqrt(np.mean(hanning_window**2))

    return hanning_window

def apply_calib_time(data_time, sample_rate, raw_calib_data):
    """
    Applies microphone calibration on time domain data
    """

    # ensure number of channels match
    # dividing by 2 because freq data is in pairs
    assert data_time.shape[1] == raw_calib_data.shape[1] // 2

    hanning_window = normalized_hanning(data_time.shape[0])
    windowed_data = data_time * hanning_window
    zero_padding = 2 ** np.ceil(np.log2(windowed_data.shape[0]) + 1).astype(int)

    ft_data = np.fft.rfft(
            windowed_data, n=zero_padding, axis=0
        )
    freq_data = np.fft.rfftfreq(zero_padding, d=1 / sample_rate)

    cal_data = apply_calib_freq(ft_data, freq_data, raw_calib_data)

    # apply the inverse fft
    output_data = np.fft.irfft(cal_data, n=zero_padding, axis=0)

    return output_data[:data_time.shape[0], :]


def apply_calib_freq(ft_data, data_freq, raw_calib_data):
    """
    Applies microphone calibration on frequency domain data
    """

    calib_freqs = raw_calib_data[:, 0::2].astype(float)
    calib_data_dB = raw_calib_data[:, 1::2].astype(float)
    calib_data = 10 ** (calib_data_dB / 20)

    # ensure number of channels match
    assert ft_data.shape[1] == calib_data.shape[1]

    # loop over channels
    for i in range(ft_data.shape[1]):
        # we not in dB so division

        ft_data[:, i] = ft_data[:, i] / np.interp(
            data_freq, calib_freqs[:, i], calib_data[:, i]
            )
        # so this will apply the min calib value for freq below the min calib freq
        # and the max calib value for freq above the max calib freq

    return ft_data


def rebase_path(path):

    if isinstance(path, str):
        path = Path(path)

    ans = path.relative_to(path.parent.parent.parent.parent)
    return str(ans)


def rebase_pathlist(pathlist):
    return [rebase_path(path) for path in pathlist]

def extract_datetime(filename):

    datetime_str = filename.split('_')[-1].replace('-', ' ').split('.')[0]
    
    return pd.to_datetime(datetime_str, format=r'%Y %m %d %H %M %S')


def clean_lookup_df(lookup_df):

    lookup_df = lookup_df.sort_values(by='datetime')
    lookup_df['time_diff'] = lookup_df['datetime'].diff().fillna(pd.Timedelta(seconds=0))
    lookup_df['group'] = (lookup_df['time_diff'] > pd.Timedelta(seconds=5)).cumsum()
    # starts a new group if the time difference is greater than 5 seconds

    new_df = pd.DataFrame(columns=lookup_df.columns)

    for _, group_df in lookup_df.groupby(['prop_path', 'group']):

        if np.isclose(group_df['speed'].iloc[0], 0):

            group_df['speed'] = group_df['speed'].shift(-1) # shift
            group_df = group_df[:-1]  # remove last row

            new_df = pd.concat([new_df, group_df])

    return new_df


def parse_lookup_df(results_folder, clean=True):

    # Base directory
    base_dir = Path(results_folder)

    df_list = []

    for meta_path in base_dir.glob("**/*.prop/meta_data.npy"):

        if 'old' in meta_path.parts:
            continue

        try:

            prop_fname = meta_path.parent.name  # .replace(".prop", "")

            meta_array = np.load(meta_path, allow_pickle=True)

            df = pd.DataFrame()

            prop_path = base_dir.parent / "props" / prop_fname

            audio_path = rebase_pathlist(meta_array[:, 0])
            
            date_time = [extract_datetime(path) for path in audio_path]

            df["audio_path"] = audio_path
            df["speed"] = meta_array[:, 1]
            df["current"] = meta_array[:, 2]
            df["temperature"] = meta_array[:, 3]
            df["prop_path"] = prop_path
            df["mic_path"] = rebase_pathlist(meta_array[:, 4])
            df['datetime'] = date_time

            df_list.append(df)

        except Exception as e:
            print(f"Error processing {meta_path}: {e}")

    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)

    if clean:
        combined_df = clean_lookup_df(combined_df)
    
    return combined_df


def _generate_cache_key(lookup_df, aero_coefficients, reference_FOM, harmonics = 0):
        lookup_hash = hashlib.md5(pd.util.hash_pandas_object(lookup_df).values).hexdigest()
        aero_hash = hashlib.md5(pickle.dumps(aero_coefficients)).hexdigest()
        return f"{lookup_hash}_{aero_hash}_{reference_FOM + harmonics}"

def parse_harmonic_df(lookup_df, aero_coefficients, harmonics = 10, use_cache=True):

    MIN_SPEED_HZ = 10
    SAMPLE_FREQ_HZ = 51200 # Hz
    REF_PRESSURE = 20e-6  # Pa
    EXPECTED_CHANNELS = 7 # recorded 7 channels for nearly everything
    EPSILON_HZ = 2 # BPF bound for integration
    # the bins are 1 Hz so pm 2 Hz is a good bound
    rho = 1.225

    mic_calibration_data = load_microphone_calibration()

    KT, KQ = np.mean(list(aero_coefficients.values()), axis=0)
    reference_FOM = KT ** (3/2) / (2 ** (1/2) * KQ)

    # Generate a unique hash for the lookup_df and aero_coefficients
    cache_key = _generate_cache_key(lookup_df, aero_coefficients, reference_FOM, harmonics)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if use_cache and cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                processed_df = pickle.load(f)
                return processed_df
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading cache file: {e}. Reprocessing data...")
            # If there's an error loading the cache, continue with processing
    
    # caches to avoid repeated file loading
    prop_cache = {}
    mic_data_cache = {}
    hanning_cache = {} # ok so audio files might be different lengths so cache hanning windows of the same length
    
    # we dont know how many rows we will have so we need to count first
    total_rows = 0
    for _, row in lookup_df.iterrows():
        mic_path = row["mic_path"]
        prop_path = row["prop_path"]
        speed = np.abs(row["speed"])
        speed_hz = speed / 60  # Convert RPM to Hz
        
        # Skip if speed is too low
        if abs(speed_hz) < MIN_SPEED_HZ:
            continue

        prop_path_str = str(prop_path)
        if prop_path_str in prop_cache:
            prop = prop_cache[prop_path_str]
        else:
            prop = load_prop_from_file(prop_path)
            prop_cache[prop_path_str] = prop
        
        bpf_hz = speed_hz * prop['B']
        propeller_name = prop_path.name.replace(".prop", "")
        try:
            aero_coefficients[propeller_name]
        except KeyError:
            print(f"Warning: No aero coefficients found for {propeller_name}. Skipping.")
            continue

        # Count mic channels
        if str(mic_path) in mic_data_cache:
            mic_data = mic_data_cache[str(mic_path)]
        elif os.path.exists(mic_path):
                mic_data = np.genfromtxt(mic_path, delimiter=",", dtype=object, encoding=None, skip_header=1)
                mic_data_cache[str(mic_path)] = mic_data
        else:
            print(f"Warning: No mic data found for {mic_path}. Skipping.")
            continue
        
        audio_path = row["audio_path"]
        if not os.path.exists(audio_path):
            continue
            
        total_rows += mic_data.shape[0] * harmonics
    
    # allocate numpy arrays for all data
    propellers = np.empty(total_rows, dtype=object)
    speeds = np.empty(total_rows)
    angles = np.empty(total_rows)
    distances = np.empty(total_rows)
    harmonic_values = np.empty(total_rows)
    # harmonic sound pressure levels
    hrms_values = np.empty(total_rows)
    hspl_values = np.empty(total_rows)
    ndhspl_values = np.empty(total_rows)
    CTs = np.empty(total_rows)
    CQs = np.empty(total_rows)
    FMs = np.empty(total_rows)
    
    # Process data and fill arrays
    row_idx = 0
    
    for _, row in lookup_df.iterrows():
        mic_path = row["mic_path"]
        prop_path = row["prop_path"]
        speed = np.abs(row["speed"])
        speed_hz = speed / 60  # Convert RPM to Hz
        speed_rad = speed_hz * 2 * np.pi  # Convert Hz to rad/s

        # continue if speed bad
        if abs(speed_hz) < MIN_SPEED_HZ:
            continue

        prop_path_str = str(prop_path)
        if prop_path_str in prop_cache:
            prop = prop_cache[prop_path_str]
        else:
            # we've already loaded all the propeller files into cache so if we get here we have a problem
            raise RuntimeError(f"Error processing row with prop_path {prop_path}: prop not found.")
        
        bpf_hz = speed_hz * prop['B']
        propeller_name = prop_path.name.replace(".prop", "")
        try:
            CT, CQ = aero_coefficients[propeller_name]
        except KeyError:
            print(f"Warning: No aero coefficients found for {propeller_name}. Skipping.")
            continue

        mic_path_str = str(mic_path)
        if mic_path_str in mic_data_cache:
            mic_data = mic_data_cache[mic_path_str]
        else:
            # we again should have already loaded all the mic files into cache
            raise RuntimeError(f"Error processing row with mic_path {mic_path}: mic_data not found.")
            
        audio_path = row["audio_path"]
        if not os.path.exists(audio_path):
            continue

        total_channels = mic_data.shape[0] # could be len mic_data
        if total_channels != EXPECTED_CHANNELS:
            raise RuntimeError(f"Expected {EXPECTED_CHANNELS} channels, but got {total_channels} in {mic_path}.")
        
        audio_data = np.fromfile(audio_path, dtype=np.float64).reshape(-1, total_channels)

        # make a hanning cache
        if audio_data.shape[0] in hanning_cache:
            hanning_window = hanning_cache[audio_data.shape[0]]
        else:
            # rms normalized hanning window
            hanning_window = normalized_hanning(audio_data.shape[0])
            hanning_cache[audio_data.shape[0]] = hanning_window
        
        windowed_data = audio_data * hanning_window
        zero_padding = 2 ** np.ceil(np.log2(windowed_data.shape[0]) + 1).astype(int)

        ft_data = np.fft.rfft(
            windowed_data, n=zero_padding, axis=0
        )
        
        # bpf format
        freq_data = np.fft.rfftfreq(zero_padding, d=1 / SAMPLE_FREQ_HZ) # Hz

        ft_data = apply_calib_freq(ft_data, freq_data, mic_calibration_data)

        num_mics = len(mic_data)

        nnext = num_mics * harmonics

        propellers[row_idx:row_idx + nnext] = propeller_name
        speeds[row_idx:row_idx + nnext] = speed
        angles[row_idx:row_idx + nnext] = np.repeat(mic_data[:, 2], harmonics)  # angle from mic state file
        distances[row_idx:row_idx + nnext] = np.repeat(mic_data[:, 3], harmonics)  # distance from mic state file
        distances[row_idx:row_idx + nnext] /= 1e3 * prop["rt"]  # normalize distance by propeller radius

        CTs[row_idx:row_idx + nnext] = CT
        CQs[row_idx:row_idx + nnext] = CQ
        FMs[row_idx:row_idx + nnext] = CT ** (3/2) / (np.sqrt(2) * CQ)

        #fom = CT ** (3/2) / (2 ** (1/2) * CQ)
        #reff = prop["rt"] * (fom / reference_FOM)
        #nd_pressure = rho * (reff * speed_rad)**2

        ref_offset = 20 * np.log10(
            KQ / CQ * (CT / KT) ** 2
        )

        for i in range(num_mics):
            for j in range(harmonics):
                # i is mic number, n is harmonic number
                n = j + 1

                center_freq = n * bpf_hz
                mask = (freq_data >= center_freq - EPSILON_HZ) & (freq_data <= center_freq + EPSILON_HZ)
                # integrate mask with np.trap
                fqharm = freq_data[mask] / bpf_hz
                ftharm = ft_data[mask, i]

                # seems to be working
                integrated_value = np.trapezoid(
                    np.abs(ftharm)**2, fqharm)
                hspl = np.sqrt(integrated_value)

                harmonic_values[row_idx] = n
                hrms_values[row_idx] = hspl
                hspl_values[row_idx] = 20 * np.log10( hspl / REF_PRESSURE )
                ndhspl_values[row_idx] = hspl_values[row_idx] + ref_offset

                row_idx += 1

    if row_idx != total_rows:
        raise RuntimeError(f"Expected {total_rows} rows, but got {row_idx} rows.")
    
    # Create DataFrame from arrays
    if row_idx == 0:
        print("No data to plot after processing.")
        return None
    
    processed_df = pd.DataFrame({
        'propeller': propellers,
        'speed': speeds,
        'angle': angles,
        'distance': distances,
        'harmonic': harmonic_values,
        'RMS' : hrms_values,
        'SPL': hspl_values,
        'SPLref': ndhspl_values,
        'CT': CTs,
        'CQ': CQs,
        'FM': FMs
    })

    # Save processed data to cache if caching is enabled
    if use_cache:
        print(f"Saving processed data to cache file {cache_file}")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(processed_df, f)
        except (FileNotFoundError) as e:
            print(f"Error saving cache file: {e}")

    return processed_df

def parse_spl_df(lookup_df, aero_coefficients, use_cache=True):

    MIN_SPEED_HZ = 10
    SAMPLE_FREQ_HZ = 51200 # Hz
    REF_PRESSURE = 20e-6  # Pa
    EXPECTED_CHANNELS = 7 # recorded 7 channels for nearly everything
    rho = 1.225

    mic_calibration_data = load_microphone_calibration()

    KT, KQ = np.mean(list(aero_coefficients.values()), axis=0)
    reference_FOM = KT ** (3/2) / (2 ** (1/2) * KQ)

    # Generate a unique hash for the lookup_df and aero_coefficients
    cache_key = _generate_cache_key(lookup_df, aero_coefficients, reference_FOM)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if use_cache and cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                processed_df = pickle.load(f)
                return processed_df
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading cache file: {e}. Reprocessing data...")
            # If there's an error loading the cache, continue with processing
    
    # caches to avoid repeated file loading
    prop_cache = {}
    mic_data_cache = {}
    
    # we dont know how many rows we will have so we need to count first
    total_rows = 0
    for _, row in lookup_df.iterrows():
        mic_path = row["mic_path"]
        prop_path = row["prop_path"]
        speed = np.abs(row["speed"])
        speed_hz = speed / 60  # Convert RPM to Hz
        
        # Skip if speed is too low
        if abs(speed_hz) < MIN_SPEED_HZ:
            continue

        prop_path_str = str(prop_path)
        if prop_path_str in prop_cache:
            prop = prop_cache[prop_path_str]
        else:
            prop = load_prop_from_file(prop_path)
            prop_cache[prop_path_str] = prop
        
        propeller_name = prop_path.name.replace(".prop", "")
        try:
            aero_coefficients[propeller_name]
        except KeyError:
            print(f"Warning: No aero coefficients found for {propeller_name}. Skipping.")
            continue

        # Count mic channels
        if str(mic_path) in mic_data_cache:
            mic_data = mic_data_cache[str(mic_path)]
        elif os.path.exists(mic_path):
                mic_data = np.genfromtxt(mic_path, delimiter=",", dtype=object, encoding=None, skip_header=1)
                mic_data_cache[str(mic_path)] = mic_data
        else:
            print(f"Warning: No mic data found for {mic_path}. Skipping.")
            continue
        
        audio_path = row["audio_path"]
        if not os.path.exists(audio_path):
            continue
            
        total_rows += mic_data.shape[0]
    
    # Pre-allocate numpy arrays for all data
    propellers = np.empty(total_rows, dtype=object)
    speeds = np.empty(total_rows)
    angles = np.empty(total_rows)
    distances = np.empty(total_rows)
    rms_values = np.empty(total_rows)
    spl_values = np.empty(total_rows)
    ndspl_values = np.empty(total_rows)
    
    # Process data and fill arrays
    row_idx = 0
    
    for _, row in lookup_df.iterrows():
        mic_path = row["mic_path"]
        prop_path = row["prop_path"]
        speed = np.abs(row["speed"])
        speed_hz = speed / 60  # Convert RPM to Hz
        speed_rad = speed_hz * 2 * np.pi  # Convert Hz to rad/s

        # continue if speed bad
        if abs(speed_hz) < MIN_SPEED_HZ:
            continue

        # check cache for already loaded prop files
        prop_path_str = str(prop_path)
        if prop_path_str in prop_cache:
            prop = prop_cache[prop_path_str]
        else:
            raise RuntimeError(f"Error processing row with prop_path {prop_path}: prop not found.")
            
        propeller_name = prop_path.name.replace(".prop", "")
        try:
            CT, CQ = aero_coefficients[propeller_name]
        except KeyError:
            print(f"Warning: No aero coefficients found for {propeller_name}. Skipping.")
            continue

        # check cahce for already loaded mic files
        mic_path_str = str(mic_path)
        if mic_path_str in mic_data_cache:
            mic_data = mic_data_cache[mic_path_str]
        else:
            # we've already checked all the mic files so if we get here we have a problem
            raise RuntimeError(f"Error processing row with mic_path {mic_path}: mic_data not found.")
            
        audio_path = row["audio_path"]
        if not os.path.exists(audio_path):
            continue

        total_channels = mic_data.shape[0] # could be len mic_data
        if total_channels != EXPECTED_CHANNELS:
            raise RuntimeError(f"Expected {EXPECTED_CHANNELS} channels, but got {total_channels} in {mic_path}.")

        audio_data = np.fromfile(audio_path, dtype=np.float64).reshape(-1, total_channels)

        audio_data = apply_calib_time(audio_data, SAMPLE_FREQ_HZ, mic_calibration_data)

        #fom = CT ** (3/2) / (2 ** (1/2) * CQ)
        #reff = prop["rt"] * (fom / reference_FOM)
        #nd_pressure = rho * (reff * speed_rad)**2

        ref_offset = 20 * np.log10(
            KQ / CQ * (CT / KT) ** 2
        )
        
        num_mics = len(mic_data)
        for i in range(num_mics):
            # now store in the allocated arrays
            propellers[row_idx] = propeller_name
            speeds[row_idx] = speed_rad
            angles[row_idx] = float(mic_data[i][2])  # angle from mic state file
            distances[row_idx] = float(mic_data[i][3])  # distance from mic state file
            distances[row_idx] /= 1e3 * prop["rt"]  # normalize distance by propeller radius
            
            # Calculate RMS
            rms = rms_butter(audio_data[:, i], speed_hz, SAMPLE_FREQ_HZ)
            rms_values[row_idx] = rms
            
            # Calculate SPL and ndSPL
            spl_values[row_idx] = 20 * np.log10(rms / REF_PRESSURE)
            ndspl_values[row_idx] = spl_values[row_idx] + ref_offset
            
            # Increment row index
            row_idx += 1

    if row_idx != total_rows:
        raise RuntimeError(f"Expected {total_rows} rows, but got {row_idx} rows.")
    
    # Create DataFrame from arrays
    if row_idx == 0:
        print("No data to plot after processing.")
        return None
    
    processed_df = pd.DataFrame({
        'propeller': propellers,
        'speed': speeds,
        'angle': angles,
        'distance': distances,
        'RMS': rms_values,
        'OASPL': spl_values,
        'OASPLref': ndspl_values
    })

    # Save processed data to cache if caching is enabled
    if use_cache:
        print(f"Saving processed data to cache file {cache_file}")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(processed_df, f)
        except (FileNotFoundError) as e:
            print(f"Error saving cache file: {e}")

    return processed_df


def test_calibration():
    import matplotlib.pyplot as plt
    import numpy as np

    # Create white noise signal
    fs = 51200
    channels = 7
    duration = 1  # seconds
    
    #signal = np.ones((fs * duration, channels))
    signal = np.random.normal(0, 1, (fs * duration, channels))

    # Load microphone calibration
    mcalib_data = load_microphone_calibration()

    # Apply calibration
    calibrated_signal = apply_calib_time(signal, fs, mcalib_data)

    # Use same FFT length for fair comparison
    n_fft = 2 ** int(np.ceil(np.log2(signal.shape[0]) + 1))  # Same padding as apply_calib_time
    win = np.hanning(signal.shape[0])
    win_norm = np.mean(win)

    ft_uncal = np.fft.rfft(signal[:, 0] * win, n=n_fft) / win_norm
    ft_cal = np.fft.rfft(calibrated_signal[:, 0], n=n_fft)
    ft_freq = np.fft.rfftfreq(n_fft, d=1 / fs)

    # Convert to dB
    spectrum_uncal = 20 * np.log10(np.abs(ft_uncal) + 1e-12)
    spectrum_cal = 20 * np.log10(np.abs(ft_cal) + 1e-12)

    calib_freqs = mcalib_data[:, 0]
    calib_dB = mcalib_data[:, 1]
    interp_calib = np.interp(ft_freq, calib_freqs, calib_dB)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(ft_freq, spectrum_uncal, label="Original")
    ax.plot(ft_freq, spectrum_cal, label="Calibrated")
    plt.plot(ft_freq, -interp_calib, label="Expected spectral change (-Calibration dB)", color="black", linestyle="--")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title("Effect of Microphone Calibration (Channel 0)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_calibration()
    pass