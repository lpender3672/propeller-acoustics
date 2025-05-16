import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import hashlib
import pickle
from pathlib import Path
from matplotlib.colors import LogNorm

from app.routines import (
    load_prop_from_file
)
from app.routines_audio import (
    load_and_compute_rfft,
    rms_butter,
    butter_filt,
    load_meta_data,
    load_microphone_calibration,
)
from app.routines_aero import (
    calc_aero_coefficients,
    load_cell_calibration
)

# Create cache directory if it doesn't exist
CACHE_DIR = Path("app/results/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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

            prop_fname = meta_path.parent.name  # .replace(".prop", "")

            meta_array = np.load(meta_path, allow_pickle=True)

            df = pd.DataFrame()

            prop_path = base_dir.parent / "props" / prop_fname

            df["audio_path"] = rebase_pathlist(meta_array[:, 0])
            df["speed"] = meta_array[:, 1]
            df["current"] = meta_array[:, 2]
            df["temperature"] = meta_array[:, 3]
            df["prop_path"] = prop_path
            df["mic_path"] = rebase_pathlist(meta_array[:, 4])

            df_list.append(df)

        except Exception as e:
            print(f"Error processing {meta_path}: {e}")

    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    
    return combined_df


def _generate_cache_key(lookup_df, aero_coefficients, reference_FOM, harmonics = 0):
        lookup_hash = hashlib.md5(pd.util.hash_pandas_object(lookup_df).values).hexdigest()
        aero_hash = hashlib.md5(pickle.dumps(aero_coefficients)).hexdigest()
        return f"{lookup_hash}_{aero_hash}_{reference_FOM + harmonics}"

def parse_harmonic_df(lookup_df, aero_coefficients, harmonics = 10, reference_FOM = 0.5, use_cache=True):

    MIN_SPEED_HZ = 10
    SAMPLE_FREQ_HZ = 51200 # Hz
    REF_PRESSURE = 20e-6  # Pa
    EXPECTED_CHANNELS = 7 # recorded 7 channels for nearly everything
    EPSILON = 5e-2 # BPF bound for integration
    rho = 1.225

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
            hanning_window = np.hanning(audio_data.shape[0]).reshape(-1, 1)
            hanning_cache[audio_data.shape[0]] = hanning_window
        
        windowed_data = audio_data * hanning_window
        zero_padding = 2 ** np.ceil(np.log2(windowed_data.shape[0]) + 1).astype(int)

        ft_data = np.fft.rfft(
            windowed_data, n=zero_padding, axis=0
        )
        
        # bpf format
        freq_data = np.fft.rfftfreq(zero_padding, d=1 / SAMPLE_FREQ_HZ) / bpf_hz

        # TODO: apply calibration

        # integrate harmonics

        num_mics = len(mic_data)

        nnext = num_mics * harmonics

        propellers[row_idx:row_idx + nnext] = propeller_name
        speeds[row_idx:row_idx + nnext] = speed
        angles[row_idx:row_idx + nnext] = np.repeat(mic_data[:, 2], harmonics)  # angle from mic state file
        distances[row_idx:row_idx + nnext] = np.repeat(mic_data[:, 3], harmonics)  # distance from mic state file

        fom = CT ** (3/2) / (2 ** (1/2) * CQ)
        reff = prop["rt"] * (fom / reference_FOM)
        nd_pressure = rho * (reff * speed_rad)**2

        for i in range(num_mics):
            for j in range(harmonics):
                # i is mic number, n is harmonic number
                n = j + 1

                mask = (freq_data > (n - EPSILON)) & (freq_data < (n + EPSILON))
                # integrate mask with np.trap
                fqharm = freq_data[mask]
                ftharm = ft_data[mask, i]

                # seems to be working
                integrated_value = np.trapezoid(ftharm**2, fqharm)
                hspl = np.sqrt(integrated_value)

                harmonic_values[row_idx] = n
                hrms_values[row_idx] = hspl
                hspl_values[row_idx] = hspl / REF_PRESSURE
                ndhspl_values[row_idx] = hspl / nd_pressure

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
        'HSPL': hspl_values,
        'ndHSPL': ndhspl_values
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

def parse_spl_df(lookup_df, aero_coefficients, reference_FOM = 0.5, use_cache=True):

    MIN_SPEED_HZ = 10
    SAMPLE_FREQ_HZ = 51200 # Hz
    REF_PRESSURE = 20e-6  # Pa
    EXPECTED_CHANNELS = 7 # recorded 7 channels for nearly everything
    rho = 1.225

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

        fom = CT ** (3/2) / (2 ** (1/2) * CQ)
        reff = prop["rt"] * (fom / reference_FOM)
        nd_pressure = rho * (reff * speed_rad)**2
        
        num_mics = len(mic_data)
        for i in range(num_mics):
            # now store in the allocated arrays
            propellers[row_idx] = propeller_name
            speeds[row_idx] = speed
            angles[row_idx] = float(mic_data[i][2])  # angle from mic state file
            distances[row_idx] = float(mic_data[i][3])  # distance from mic state file
            
            # Calculate RMS
            rms = rms_butter(audio_data[:, i], speed_hz, SAMPLE_FREQ_HZ)
            rms_values[row_idx] = rms
            
            # Calculate SPL and ndSPL
            spl_values[row_idx] = 20 * np.log10(rms / REF_PRESSURE)
            ndspl_values[row_idx] = rms / nd_pressure
            
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
        'SPL': spl_values,
        'ndSPL': ndspl_values
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


def _plot_data(ax, group_df, x_var, y_var, label=None, plot_type='line', is_polar=False, alpha=0.7, 
             colour_by=None, cmap='viridis'):
    """
    Helper function to plot data with various plot types.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to plot on
    group_df : pandas DataFrame
        DataFrame containing the data to plot
    x_var : str
        Variable to plot on x-axis
    y_var : str
        Variable to plot on y-axis
    label : str, optional
        Label for the plot legend
    plot_type : str
        Type of plot ('line', 'scatter', 'polar', 'bar')
    is_polar : bool
        Whether this is a polar plot
    alpha : float
        Alpha transparency for the plot
    colour_by : str, optional
        Variable to use for colouring points (only for scatter plots)
    cmap : str, optional
        colourmap name to use for colour mapping
        
    Returns:
    --------
    scatter : matplotlib scatter plot object or None
        The scatter plot object (only when scatter plot with colour_by is used)
    """
    # Sort data if x_var is distance or angle for better line plots
    if x_var in ['distance', 'angle']:
        group_df = group_df.sort_values(by=x_var)
    
    x_data = group_df[x_var]
    y_data = group_df[y_var]
    
    scatter = None
    
    if is_polar:
        # Convert angles to radians for polar plot
        x_radians = np.deg2rad(x_data) if x_var == 'angle' else np.deg2rad(group_df['angle'])
        ax.plot(x_radians, y_data, 'o-', label=label)
    elif plot_type == 'scatter':
        if colour_by is not None and colour_by in group_df.columns:
            # Use the specified variable for colouring points
            c_data = group_df[colour_by]
            scatter = ax.scatter(x_data, y_data, c=c_data, cmap=cmap, label=label, alpha=alpha)
        else:
            ax.scatter(x_data, y_data, label=label, alpha=alpha)
    elif plot_type == 'bar':
        ax.bar(x_data, y_data, label=label, alpha=alpha)
    else:  # default to line plot
        ax.plot(x_data, y_data, 'o-', label=label)
        
    return scatter


def _format_group_name(name, modified_group_by, original_group_by, units):
    # Handle speed_bin conversion if 'speed' was in the original grouping
    if 'speed' in original_group_by:
        if isinstance(name, tuple):
            name_list = list(name)
            for i, g in enumerate(modified_group_by):
                if g == 'speed_bin':
                    name_list[i] = f"{int(round(name_list[i]))} {units['speed']}"
            name = tuple(name_list)
        else:
            name = f"{int(round(name))} {units['speed']}"
    
    formatted_name = str(name) if not isinstance(name, tuple) else ', '.join(map(str, name))
    return formatted_name


def multi_function_plot(processed_df, x_var, y_var, filter_dict=None, group_by=None, plot_type='line', 
                  title=None, fig_size=(10, 6), log_bin_factor=0.05,
                  max_groups=10, colour_by=None, cmap='viridis', colourbar_label=None,
                  log_colourbar=False, units=None):
    """
    AI generated docstring to help me remember what this function does.

    A multi purpose function for plotting propeller acoustics data with filtering and grouping capabilities.
    
    Parameters:
    -----------
    processed_df : pandas DataFrame
        DataFrame containing the propeller acoustics data
    x_var : str
        Variable to plot on x-axis ('angle', 'speed', 'distance', 'propeller', 'SPL')
    y_var : str
        Variable to plot on y-axis ('angle', 'speed', 'distance', 'propeller', 'SPL')
    filter_dict : dict, optional
        Dictionary of filters to apply, e.g., {'speed': (1000, 2000), 'angle': [0, 90, 180]}
    group_by : str or list, optional
        Variable(s) to group by for multiple plots/lines
    plot_type : str, optional
        Type of plot ('line', 'scatter', 'polar', 'bar')
    title : str, optional
        Plot title
    fig_size : tuple, optional
        Figure size as (width, height)
    log_bin_factor : float, optional
        Factor for logarithmic binning of speed values (default: 0.05). 
        Lower values create more bins, higher values create fewer bins.
        Examples: 0.05 ≈ 12% bin width, 0.10 ≈ 25% bin width.
    max_groups : int, optional
        Maximum number of groups to plot (default: 10). If more groups are found, only the most interesting ones are plotted.
    colour_by : str, optional
        Variable to use for colour mapping in scatter plots (e.g., 'speed', 'SPL')
    cmap : str, optional
        colourmap name to use for colour mapping (default: 'viridis')
    colourbar_label : str, optional
        Label for the colourbar. If None, the colour_by variable name is used.
    log_colourbar : bool, optional
        Whether to use a logarithmic scale for the colourbar (default: False)
    units : dict, optional
        Dictionary mapping variable names to their units (e.g., {'speed': 'RPM', 'SPL': 'dB'})
        If None, default units will be used.
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Default units dictionary
    units = {
        'speed': 'RPM',
        'SPL': 'dB',
        'angle': '°',
        'distance': 'mm',
        'RMS': 'Pa'
    }
    
    # Apply filters if provided
    if filter_dict:
        for key, value in filter_dict.items():
            if key in processed_df.columns:
                if isinstance(value, tuple) and len(value) == 2:
                    # range filter (min, max)
                    processed_df = processed_df[(processed_df[key] >= value[0]) & 
                                            (processed_df[key] <= value[1])]
                elif isinstance(value, list):
                    # list of values
                    processed_df = processed_df[processed_df[key].isin(value)]
                elif isinstance(value, float):
                    processed_df = processed_df[np.isclose(
                        processed_df[key], value, rtol=0.01)] # 1% probably a bit too much
                
                else:
                    # int or string probably
                    processed_df = processed_df[processed_df[key] == value]
    
    if processed_df.empty:
        print("No data left after applying filters.")
        return None, None
    
    # this is chatgpts idea to bin the speed values to help grouping
    with np.errstate(divide='ignore'):  # Ignore log10(0) warnings
        log_speed = np.log10(processed_df['speed'])
        log_speed[np.isneginf(log_speed)] = 0  # Handle log(0) = -inf
        log_speed_binned = np.round(log_speed / log_bin_factor) * log_bin_factor
        processed_df['speed_bin'] = np.power(10, log_speed_binned)
    

    fig, ax = plt.subplots(figsize=fig_size)
    
    is_polar = plot_type == 'polar' #or (x_var == 'angle' and plot_type == 'line')
    if is_polar:
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
    
    # only used if colour bar
    scatter_with_colourbar = None
    
    if group_by:

        if not isinstance(group_by, list):
            group_by = [group_by]
        
        # need to replace 'speed' with 'speed_bin' in group_by for relaxed grouping
        modified_group_by = []
        for g in group_by:
            if g == 'speed':
                modified_group_by.append('speed_bin')
                bin_size_percent = round((10**log_bin_factor - 1) * 100)
                print(f"Using logarithmic speed binning (bin width ~{bin_size_percent}%) for grouping")
            else:
                modified_group_by.append(g)
        
        # group by the correct grouping variables
        grouped = processed_df.groupby(modified_group_by)
        
        if len(grouped) > max_groups:
            print(f"Warning: Too many groups ({len(grouped)}). Limiting to {max_groups} most interesting groups.")
            
            # Get group sizes
            group_sizes = grouped.size()
            
            # Calculate the variance of y_var for each group to determine "interestingness"
            group_variances = grouped[y_var].var()
            
            # Combine size and variance for ranking - groups with more data points and higher variance are more "interesting"
            interestingness = group_sizes * group_variances
            
            # Get the keys (group names) of the top max_groups most interesting groups
            top_groups = interestingness.nlargest(max_groups).index.tolist()
            
            # Create a filtered groups dictionary with only the top groups
            filtered_groups = {key: grouped.get_group(key) for key in top_groups}
            
            # Iterate through the filtered groups
            for name, group in filtered_groups.items():

                label = _format_group_name(name, modified_group_by, group_by, units)
                
                scatter = _plot_data(ax, group, x_var, y_var, label=label, plot_type=plot_type, 
                                   is_polar=is_polar, colour_by=colour_by, cmap=cmap)
                
                if scatter is not None:
                    scatter_with_colourbar = scatter
        else:
            # plot all groups
            for name, group in grouped:

                label = _format_group_name(name, modified_group_by, group_by, units)
                
                scatter = _plot_data(ax, group, x_var, y_var, label=label, plot_type=plot_type, 
                                   is_polar=is_polar, colour_by=colour_by, cmap=cmap)
                
                if scatter is not None:
                    scatter_with_colourbar = scatter
    else:
        # no grouping, plot all data
        scatter_with_colourbar = _plot_data(ax, processed_df, x_var, y_var, plot_type=plot_type, 
                                         is_polar=is_polar, colour_by=colour_by, cmap=cmap)
        
    # now plotting done maybe add colourbar
    # add labels and title
    
    if colour_by is not None and scatter_with_colourbar is not None:
        # do a colourbar 
        if log_colourbar:
            scatter_with_colourbar.set_norm(LogNorm())
        
        cbar = fig.colorbar(scatter_with_colourbar, ax=ax)
        
        if colourbar_label is not None:
            cbar_label = colourbar_label
        elif colour_by in units:
            cbar_label = f"{colour_by} ({units[colour_by]})"
        else:
            cbar_label = colour_by
            
        cbar.set_label(cbar_label)
    
    # labels and title
    # most cases this should be overwritten
    if not is_polar:
        if x_var in units:
            ax.set_xlabel(f"{x_var} ({units[x_var]})")
        else:
            ax.set_xlabel(x_var)
            
        if y_var in units:
            ax.set_ylabel(f"{y_var} ({units[y_var]})")
        else:
            ax.set_ylabel(y_var)
    else:

        ax.set_theta_zero_location('N')  # 0 degrees at the top
        ax.set_theta_direction(-1)  # clockwise
        ax.set_xlim(0, 180)  # axisymmetric
        if y_var != 'angle' and y_var in units:
            ax.set_ylabel(f"{y_var} ({units[y_var]})")
        elif y_var != 'angle':
            ax.set_ylabel(y_var)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{y_var} vs {x_var}")
    
    # add legend if grouped
    if group_by:
        ax.legend()
    
    ax.grid(True)
    
    return fig, ax


if __name__ == "__main__":

    cal_data = load_cell_calibration()

    aero_coeffs = calc_aero_coefficients(
        'app/results',
        cal_data
    )

    df = parse_lookup_df("app/results/")

    pdf = parse_spl_df(df, aero_coeffs)

    hdf = parse_harmonic_df(df, aero_coeffs, harmonics=10)

 
    fig, ax = multi_function_plot(pdf, 
                            x_var='angle', 
                            y_var='SPL',
                            filter_dict={'distance' : 1270, 'speed' : (9500, 10500)},
                            group_by=['propeller'],
                            plot_type='line')
    

    fig, ax = multi_function_plot(pdf, 
                            x_var='distance', 
                            y_var='ndSPL',
                            filter_dict={ 'propeller' : 'dalprop5045', 'angle' : 90},
                            #group_by=[],
                            plot_type='scatter',
                            colour_by='speed',
                            log_bin_factor=0.01,
                            log_colourbar=True)
    
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    
    fig, ax = multi_function_plot(pdf, 
                            x_var='distance', 
                            y_var='ndSPL',
                            filter_dict={ 'propeller' : 'dalprop5045', 'angle' : 90, 'speed' : (10000, 13000)},
                            group_by=['speed'],
                            plot_type='line',
                            #colour_by='speed',
                            log_bin_factor=0.01,
                            log_colourbar=True)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')

    plt.show()
