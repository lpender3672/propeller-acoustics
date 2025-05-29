import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib.colors import (
    LogNorm,
    LinearSegmentedColormap,
    TwoSlopeNorm
)

from app.routines_audio import (
    parse_lookup_df,
    parse_spl_df,
    parse_harmonic_df
)
from app.routines_aero import (
    calc_aero_coefficients,
    load_cell_calibration
)

def filter_df(df, filter_dict, aggregate_dict = None):

    for key, value in filter_dict.items():
            if key in df.columns:
                if isinstance(value, tuple) and len(value) == 2:
                    # range filter (min, max)
                    df = df[(df[key] >= value[0]) & 
                                (df[key] <= value[1])]
                elif isinstance(value, list):
                    # list of values
                    df = df[df[key].isin(value)]
                elif isinstance(value, float):
                    if key == 'speed':
                        atol = 10
                    else:
                        atol = 0
                        
                    df = df[np.isclose(
                        df[key], value, rtol=0.01, atol=atol)] # 1% probably a bit too much
                
                else:
                    df = df[df[key] == value]

    if aggregate_dict is not None:
        # aggregate the data if specified
        df = df.groupby(list(aggregate_dict.keys())).agg(aggregate_dict).reset_index()
    
    return df

# TODO: raw spectrum plotting windows of key harmonics

def create_3point_colormap(name='custom_cmap', color_min='blue', color_mid='white', color_max='red'):

    colors = [color_min, color_mid, color_max]
    cmap = LinearSegmentedColormap.from_list(name, colors, N=256)
    return cmap


def _plot_data(ax, group_df, x_var, y_var, label=None, plot_type='line', is_polar=False, alpha=0.7, 
             colour_by=None, cmap='viridis', two_slope_normalize=False, **kwargs):
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

    # Evaluate kwargs if custom lambda function is provided
        # Evaluate kwargs if provided
    evaluated_kwargs = {}
    for key, value in kwargs.items():
        if callable(value):
            evaluated_kwargs[key] = value(group_df)
        else:
            evaluated_kwargs[key] = value


    # Sort data if x_var is distance or angle for better line plots
    if x_var in ['distance', 'angle']:
        group_df = group_df.sort_values(by=x_var)
    
    x_data = group_df[x_var]
    y_data = group_df[y_var]
    
    scatter = None

    if label:
        try: float(label)
        except ValueError:
            pass
        else:
            label = str(np.round(float(label), 2))  # Round numeric labels for better readability

    if is_polar:
        # Convert angles to radians for polar plot
        x_radians = np.deg2rad(x_data) if x_var == 'angle' else np.deg2rad(group_df['angle'])
        ax.plot(x_radians, y_data, 'o-', label=label)
    elif plot_type == 'scatter':
        if colour_by is not None and colour_by in group_df.columns:
            # Use the specified variable for colouring points
            c_data = group_df[colour_by]
            c_norm = None
            if two_slope_normalize:
                #c_norm = TwoSlopeNorm(vmin=c_data.min(), vcenter=0, vmax=c_data.max())
                c_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            scatter = ax.scatter(x_data, y_data, c=c_data, cmap=cmap, label=label, alpha=alpha, norm=c_norm, **evaluated_kwargs)
        else:
            ax.scatter(x_data, y_data, label=label, alpha=alpha, **evaluated_kwargs)
    elif plot_type == 'bar':
        ax.bar(x_data, y_data, label=label, alpha=alpha, **evaluated_kwargs)
    else:  # default to line plot
        ax.plot(x_data, y_data, 'o-', label=label, alpha=alpha, **evaluated_kwargs)
        
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
                  title=None, fig_size=(8, 6), log_bin_factor=0.05,
                  max_groups=10, colour_by=None, cmap='viridis', colourbar_label=None,
                  log_colourbar=False, units=None, **kwargs):
    """
    AI generated docstring to help me remember what this function does.

    A multi purpose function for plotting propeller acoustics data with filtering and grouping capabilities.
    
    Parameters:
    -----------
    processed_df : pandas DataFrame
        DataFrame containing the propeller acoustics data
    x_var : str
        Variable to plot on x-axis ('angle', 'speed', 'distance', 'propeller', 'OASPL')
    y_var : str
        Variable to plot on y-axis ('angle', 'speed', 'distance', 'propeller', 'OASPL')
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
        Variable to use for colour mapping in scatter plots (e.g., 'speed', 'OASPL')
    cmap : str, optional
        colourmap name to use for colour mapping (default: 'viridis')
    colourbar_label : str, optional
        Label for the colourbar. If None, the colour_by variable name is used.
    log_colourbar : bool, optional
        Whether to use a logarithmic scale for the colourbar (default: False)
    units : dict, optional
        Dictionary mapping variable names to their units (e.g., {'speed': 'rad/s', 'OASPL': 'dB'})
        If None, default units will be used.
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Default units dictionary
    units = {
        'speed': 'rad/s',
        'OASPL': 'dB',
        'angle': '°',
        'distance': 'mm',
        'RMS': 'Pa'
    }
    
    # Apply filters if provided
    if filter_dict:
        processed_df = filter_df(processed_df, filter_dict)
    
    if processed_df.empty:
        print("No data left after applying filters.")
        return None, None
    
    # this is chatgpts idea to bin the speed values to help grouping
    if group_by and 'speed' in group_by:
        with np.errstate(divide='ignore'):  # Ignore log10(0) warnings
            log_speed = np.log10(processed_df['speed'])
            log_speed[np.isneginf(log_speed)] = 0  # Handle log(0) = -inf
            log_speed_binned = np.round(log_speed / log_bin_factor) * log_bin_factor
            processed_df['speed_bin'] = np.power(10, log_speed_binned)
    

    fig, ax = plt.subplots(figsize=fig_size)
    
    is_polar = plot_type == 'polar' #or (x_var == 'angle' and plot_type == 'line')
    if is_polar:
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))

    two_slope_normalize = cmap == '3point'
    if two_slope_normalize:
        cmap = create_3point_colormap(name='custom_cmap', color_min='purple', color_mid='blue', color_max='red')
    
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
        # if single group, sort by the original group_by variable
        if len(grouped) == 1:
            grouped = processed_df.sort_values(by=group_by)

        
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
                                   is_polar=is_polar, colour_by=colour_by, cmap=cmap, two_slope_normalize=two_slope_normalize,
                                   **kwargs)
                
                if scatter is not None:
                    scatter_with_colourbar = scatter
        else:
            # plot all groups
            for name, group in grouped:

                label = _format_group_name(name, modified_group_by, group_by, units)
                
                scatter = _plot_data(ax, group, x_var, y_var, label=label, plot_type=plot_type, 
                                   is_polar=is_polar, colour_by=colour_by, cmap=cmap, two_slope_normalize=two_slope_normalize,
                                   **kwargs)
                
                if scatter is not None:
                    scatter_with_colourbar = scatter
    else:
        # no grouping, plot all data
        scatter_with_colourbar = _plot_data(ax, processed_df, x_var, y_var, plot_type=plot_type, 
                                         is_polar=is_polar, colour_by=colour_by, cmap=cmap, two_slope_normalize=two_slope_normalize,
                                         **kwargs)
        
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
    fig.tight_layout()
    
    return fig, ax


if __name__ == "__main__":

    cal_data = load_cell_calibration()

    aero_coeffs = calc_aero_coefficients(
        'app/results',
        cal_data
    )

    lookup_df = parse_lookup_df("app/results/")

    pdf = parse_spl_df(lookup_df, aero_coeffs)

    hdf = parse_harmonic_df(lookup_df, aero_coeffs, harmonics=10)

    print(hdf.columns)

    pis = 2*np.pi/60
    fig, ax = multi_function_plot(pdf, 
                            x_var='angle', 
                            y_var='OASPL',
                            filter_dict={'distance' : 1270, 'speed' : (9500*pis, 10500*pis)},
                            group_by=['propeller'],
                            plot_type='line')

    fig, ax = multi_function_plot(pdf, 
                            x_var='distance', 
                            y_var='OASPLref',
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
                            y_var='OASPLref',
                            filter_dict={ 'propeller' : 'dalprop5045', 'angle' : 90, 'speed' : (10000*pis, 13000*pis)},
                            group_by=['speed'],
                            plot_type='line',
                            #colour_by='speed',
                            log_bin_factor=0.01,
                            log_colourbar=True)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')

    plt.show()
