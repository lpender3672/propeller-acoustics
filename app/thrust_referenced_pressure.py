import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
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
)

tcal_data = np.load("app/bcal_thrust.npy")
qcal_data = np.load("app/bcal_torque.npy")


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


def calculate_reference_pressures(prop_result_path):

    aero_data, prop = load_aer_data(prop_result_path)

    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    avg_speed = np.mean(motor_data[:, :, 1], axis=1) * 2 * np.pi / 60
    avg_speed = np.abs(avg_speed)
    avg_raw_forces = np.mean(force_data[:, :, 1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:, 0], tcal_data[:, 0, 0], tcal_data[:, 1, 0])
    avg_torque = np.interp(avg_raw_forces[:, 1], qcal_data[:, 0, 1], qcal_data[:, 1, 1])

    filtered_speed = avg_speed[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]
    filtered_thrust = avg_thrust[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]
    filtered_torque = avg_torque[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]

    meta, _ = load_meta_data(prop_result_path)

    total_channels = 7
    freq = 51200  # hz

    cutoff_freq = 50

    # 51200 Hz
    freq_cutoff_idx = 56000

    ft_data = np.zeros((len(meta), freq_cutoff_idx, total_channels))

    speeds = []
    rmses = []

    for i, row in enumerate(meta):
        # do for all speeds
        audiof = Path(row[0])
        relative_audiof = rebase_path(audiof)

        try:
            data = np.fromfile(relative_audiof, dtype=np.float64).reshape(
                -1, total_channels
            )
        except FileNotFoundError:
            continue

        speed = row[1] / -60
        speed = max(abs(speed), 10)
        rmses.append(rms_butter(data[:, 5], speed, freq))
        speeds.append(speed)

    rmses = np.array(rmses)
    speeds = np.array(speeds)

    a, b = np.polyfit(np.log(speeds[speeds > 10]), np.log(rmses[speeds > 10]), 1)

    print("Slope:", a)
    print("Intercept:", b)

    plt.loglog(speeds, rmses, "o")
    plt.loglog(speeds, np.exp(b) * speeds**a, label="Fit")
    plt.grid(which="both")
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

            prop_name = meta_path.parent.name  # .replace(".prop", "")

            meta_array = np.load(meta_path, allow_pickle=True)

            df = pd.DataFrame()

            prop_path = base_dir.parent / "props" / prop_name

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


def parse_spl_df(lookup_df, aero_coefficients):

    # TODO: calculate reference pressure from aero_coefficients
    ref_pressure = 2e-5  # for now: standard reference pressure in air

    # Create a copy of the dataframe for processing
    processed_df = pd.DataFrame()
    
    # Process each row to extract audio data, mic data, and calculate SPL
    data_records = []
    
    for _, row in lookup_df.iterrows():
        mic_path = row["mic_path"]
        prop_path = row["prop_path"]
        speed = row["speed"]
        speed_hz = speed / -60  # Convert RPM to Hz
        
        # Skip if speed is too low
        if abs(speed_hz) < 10:
            continue
            
        # Extract propeller name
        propeller = prop_path.name.replace(".prop", "")
        
        try:
            # Load microphone position data
            mic_data = np.genfromtxt(mic_path, delimiter=",", dtype=None, encoding=None, skip_header=1)
            
            # Skip if audio data doesn't exist
            if not os.path.exists(row["audio_path"]):
                continue
                
            # Load audio data
            total_channels = 7  # Assuming 7 channels as in the original code
            audio_data = np.fromfile(row["audio_path"], dtype=np.float64).reshape(-1, total_channels)
            
            # Process each microphone channel
            for i,_ in enumerate(mic_data):
                angle = int(mic_data[i][2])  # Angle from mic_data
                distance = float(mic_data[i][3])  # Distance from mic_data
                
                # Calculate RMS value
                rms = rms_butter(audio_data[:, i], abs(speed_hz), 51200)

                # Calculate SPL (Sound Pressure Level) in dB
                spl = 20 * np.log10(rms / ref_pressure)
                
                # Add to records
                data_records.append({
                    'propeller': propeller,
                    'speed': np.abs(speed),
                    'angle': angle,
                    'distance': distance,
                    'RMS': rms,
                    'SPL': spl
                })

        except Exception as e:
            print(f"Error processing row with mic_path {mic_path}: {e}")
            continue
    
    # Create DataFrame from processed data
    if not data_records:
        print("No data to plot after processing.")
        return None, None
        
    processed_df = pd.DataFrame(data_records)

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
    
    is_polar = plot_type == 'polar' or (x_var == 'angle' and plot_type == 'line')
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


    df = parse_lookup_df("app/results/")

    pdf = parse_spl_df(df, None)

    """
    fig, ax = multi_function_plot(pdf, 
                            x_var='angle', 
                            y_var='SPL',
                            filter_dict={'distance' : 1270, 'speed' : 10000},
                            group_by=['propeller'],
                            plot_type='polar')"""
    
    fig, ax = multi_function_plot(pdf, 
                            x_var='distance', 
                            y_var='SPL',
                            filter_dict={ 'propeller' : 'dalprop5045', 'angle' : 90},
                            #group_by=[],
                            plot_type='scatter',
                            colour_by='speed',
                            log_bin_factor=0.01,
                            log_colourbar=True)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    
    fig, ax = multi_function_plot(pdf, 
                            x_var='distance', 
                            y_var='SPL',
                            filter_dict={ 'propeller' : 'dalprop5045', 'angle' : 180},
                            #group_by=[],
                            plot_type='scatter',
                            colour_by='speed',
                            log_bin_factor=0.01,
                            log_colourbar=True)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')

    plt.show()
