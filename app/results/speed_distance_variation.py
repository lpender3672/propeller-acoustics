from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from app.results.graphing_tools import (
    multi_function_plot,
    filter_df
)

from app.routines import (
    load_prop_from_file
)

from app.routines_audio import (
    parse_lookup_df,
    parse_harmonic_df,
    parse_spl_df
)

from app.routines_aero import (
    load_cell_calibration,
    calc_aero_coefficients
)

def plot_3d_regression_scatter(df, coeffs):

    speed = df['speed'].values * 2 * np.pi / 60  # rad/s
    distance = df['distance'].values * 1e-3      # m
    SPL = df['SPL'].values

    x = np.log10(speed)
    y = np.log10(distance)
    z = np.log10(SPL)

    beta0, beta1, beta2 = coeffs
    z_fit = beta0 + beta1 * x + beta2 * y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='k', label='Data')

    ax.plot_trisurf(x, y, z_fit, color='blue', alpha=0.2, linewidth=0, antialiased=True)

    ax.set_xlabel('log10(speed) [rad/s]')
    ax.set_ylabel('log10(distance) [m]')
    ax.set_zlabel('log10(SPL)')
    ax.set_title('3D scatter with fitted regression plane')
    plt.legend()
    plt.tight_layout()


def speed_distance_SPL_regression(hdf):

    # proof of concept for the speed-distance regression
    # group by propeller, angle, harmonic
    # run 2D regression on log 
    # update had to normalize the data to avoid
    # better fit to the higher magnitude values

    data_rows = []

    angle_tolerance = 22.5  # degrees
    half_width = angle_tolerance / 2
    angle_centers = np.arange(0, 181, angle_tolerance)
    angle_bin_edges = np.arange(-half_width, 181 + half_width + 1e-6, angle_tolerance)

    hdf['angle_bin'] = pd.cut(hdf['angle'], bins=angle_bin_edges, labels=False) * angle_tolerance

    for (propeller, angle, harmonic), group_df in hdf.groupby(['propeller', 'angle_bin', 'harmonic']):
        if len(group_df) < 3:
            continue  # not enough points

        propf = f'app/props/{propeller}.prop'
        prop = load_prop_from_file(propf)

        speed = group_df['speed'].values * 2 * np.pi / 60  # to rad/s
        distance = group_df['distance'].values * 1e-3 # to m

        logSpeed = np.log10(speed)
        logDistance = np.log10(distance)
        logSPL = group_df['SPL'].values / 20
        logSPL -= 2 * np.log10(prop['rt']) # remove the effect of propeller radius
        # only -2log radius because distance is already in R/rt

        # normalization
        logSpeed_norm = (logSpeed - np.mean(logSpeed)) / np.std(logSpeed)
        logDistance_norm = (logDistance - np.mean(logDistance)) / np.std(logDistance)
        logSPL_norm = (logSPL - np.mean(logSPL)) / np.std(logSPL)

        # normalization factors
        speed_mean, speed_std = np.mean(logSpeed), np.std(logSpeed)
        distance_mean, distance_std = np.mean(logDistance), np.std(logDistance)
        spl_mean, spl_std = np.mean(logSPL), np.std(logSPL)

        X_norm = np.column_stack((np.ones_like(logSpeed_norm), logSpeed_norm, logDistance_norm))
        y_norm = logSPL_norm

        # least squares regression on normalized data
        normalized_coeffs, _, _, _ = np.linalg.lstsq(X_norm, y_norm, rcond=None)
        beta0_norm, beta1_norm, beta2_norm = normalized_coeffs

        # back to original scale coefficients for interpretability
        # logSPL = beta0 + beta1*logSpeed + beta2*logDistance
        beta1 = beta1_norm * (spl_std / speed_std)
        beta2 = beta2_norm * (spl_std / distance_std)
        beta0 = spl_mean - beta1 * speed_mean - beta2 * distance_mean

        # For validation, compute predictions and residuals on original scale
        X = np.column_stack((np.ones_like(logSpeed), logSpeed, logDistance))
        coeffs = np.array([beta0, beta1, beta2])
        
        y_pred = X @ coeffs
        residual = logSPL - y_pred
        residual_std = np.std(residual)
        data_std = np.std(logSPL)
        residual_std_norm = residual_std / data_std if data_std != 0 else np.inf

        data_rows.append({
            'propeller': propeller,
            'angle_bin': angle,
            'harmonic': harmonic,
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'beta0_norm': beta0_norm,
            'beta1_norm': beta1_norm, 
            'beta2_norm': beta2_norm,
            'r_squared': 1 - np.var(residual) / np.var(logSPL),
            'residual_std': residual_std,
            'residual_std_norm': residual_std_norm,
        })

 
    # make a dataframe from the data rows
    coeffs_df = pd.DataFrame(data_rows)
    
    return coeffs_df

def speed_distance_OASPL_regression(sdf):

    data_rows = []

    angle_tolerance = 22.5  # degrees
    half_width = angle_tolerance / 2
    angle_centers = np.arange(0, 181, angle_tolerance)
    angle_bin_edges = np.arange(-half_width, 181 + half_width + 1e-6, angle_tolerance)

    sdf['angle_bin'] = pd.cut(sdf['angle'], bins=angle_bin_edges, labels=False) * angle_tolerance

    for (propeller, angle), group_df in sdf.groupby(['propeller', 'angle_bin']):
        if len(group_df) < 3:
            continue  # not enough points

        propf = f'app/props/{propeller}.prop'
        prop = load_prop_from_file(propf)

        speed = group_df['speed'].values * 2 * np.pi / 60  # to rad/s
        distance = group_df['distance'].values

        logSpeed = np.log10(speed)
        logDistance = np.log10(distance)
        logSPL = group_df['OASPL'].values / 20
        logSPL -= 2 * np.log10(prop['rt']) # remove the effect of the propeller radius
        # only -2log radius because distance is already in R/rt

        # normalization
        logSpeed_norm = (logSpeed - np.mean(logSpeed)) / np.std(logSpeed)
        logDistance_norm = (logDistance - np.mean(logDistance)) / np.std(logDistance)
        logSPL_norm = (logSPL - np.mean(logSPL)) / np.std(logSPL)

        # normalization factors
        speed_mean, speed_std = np.mean(logSpeed), np.std(logSpeed)
        distance_mean, distance_std = np.mean(logDistance), np.std(logDistance)
        spl_mean, spl_std = np.mean(logSPL), np.std(logSPL)

        X_norm = np.column_stack((np.ones_like(logSpeed_norm), logSpeed_norm, logDistance_norm))
        y_norm = logSPL_norm

        # least squares regression on normalized data
        try:
            normalized_coeffs, _, _, _ = np.linalg.lstsq(X_norm, y_norm, rcond=None)
        except np.linalg.LinAlgError:
            print(f"LinAlgError for propeller {propeller}, angle {angle}. Skipping this group.")
            continue

        beta0_norm, beta1_norm, beta2_norm = normalized_coeffs

        # back to original scale coefficients for interpretability
        # logSPL = beta0 + beta1*logSpeed + beta2*logDistance
        beta1 = beta1_norm * (spl_std / speed_std)
        beta2 = beta2_norm * (spl_std / distance_std)
        beta0 = spl_mean - beta1 * speed_mean - beta2 * distance_mean

        # For validation, compute predictions and residuals on original scale
        X = np.column_stack((np.ones_like(logSpeed), logSpeed, logDistance))
        coeffs = np.array([beta0, beta1, beta2])
        
        y_pred = X @ coeffs
        residual = logSPL - y_pred
        residual_std = np.std(residual)
        data_std = np.std(logSPL)
        residual_std_norm = residual_std / data_std if data_std != 0 else np.inf

        data_rows.append({
            'propeller': propeller,
            'angle_bin': angle,
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'beta0_norm': beta0_norm,
            'beta1_norm': beta1_norm, 
            'beta2_norm': beta2_norm,
            'r_squared': 1 - np.var(residual) / np.var(logSPL),
            'residual_std': residual_std,
            'residual_std_norm': residual_std_norm,
        })

 
    # make a dataframe from the data rows
    coeffs_df = pd.DataFrame(data_rows)
    
    return coeffs_df


def plot_best_coeffs_2D(hdf, coeffs_df):
    # objective for speed
    nshow = 3

    speed_obj =  (coeffs_df['residual_std_norm']).abs() + (coeffs_df['beta1'] - 2).abs() #+ (coeffs_df['beta2'] + 1).abs()
    best_speed_rows = coeffs_df.iloc[speed_obj.argsort()[:nshow]]

    # objective for distance 
    distance_obj = (coeffs_df['residual_std_norm']).abs() + (coeffs_df['beta2'] + 1).abs()
    best_distance_rows = coeffs_df.iloc[distance_obj.argsort()[:nshow]]

    print(f'Best speed row: {best_speed_rows}')
    print(f'Best distance row: {best_distance_rows}')

    # Plot the data
    speed_harmonic = best_speed_rows.iloc[0]['harmonic']
    speed_angle = best_speed_rows.iloc[0]['angle_bin']
    speed_propeller = best_speed_rows.iloc[0]['propeller']


    #print(hdf['speed'].unique())
    
    # Plot the data
    fig, ax = multi_function_plot(hdf, 
                            x_var='speed', 
                            y_var='SPL',
                            filter_dict = {
                                'propeller': speed_propeller,
                                'angle_bin': speed_angle,
                                'harmonic': speed_harmonic,
                            },
                            plot_type='scatter',
                            colour_by='distance',
                            log_bin_factor=0.01,
                            log_colourbar=True)
    
    xlo,xhi = ax.get_xlim()
    x_speed = np.linspace(xlo, xhi, 1000)
    y_speed = 20 * np.log10(5e-5 * x_speed ** 2)
    ax.plot(x_speed, y_speed, 'r--', label='Speed Fit', linewidth=1)
    
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.grid(True, which='both')

    distance_harmonic = best_distance_rows.iloc[0]['harmonic']
    distance_angle = best_distance_rows.iloc[0]['angle_bin']
    distance_propeller = best_distance_rows.iloc[0]['propeller']

    #print(filter_df(hdf, filter_dict = {
    #                            'propeller': distance_propeller,
    #                            'angle': distance_angle,
    #                            'harmonic': distance_harmonic
    #                        })['speed'].unique())

    # Plot the data
    fig, ax = multi_function_plot(hdf, 
                            x_var='distance', 
                            y_var='SPL',
                            filter_dict = {
                                'propeller': distance_propeller,
                                'angle_bin': distance_angle,
                                'harmonic': distance_harmonic
                            },
                            plot_type='scatter',
                            colour_by='speed',
                            log_bin_factor=0.01,
                            log_colourbar=True)
    
    xlo,xhi = ax.get_xlim()
    x_distance = np.linspace(xlo, xhi, 1000)
    y_distance = 20 * np.log10(5e6 * x_distance ** -1)
    ax.plot(x_distance, y_distance, 'r--', label='Speed Fit', linewidth=1)
    
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.grid(True, which='both')
    

def plot_best_coeffs_3D(hdf, coeffs_df):

    best_speed_rows = coeffs_df.iloc[(coeffs_df['beta1'] - 2).abs().argsort()[:3]]

    best_distance_rows = coeffs_df.iloc[(coeffs_df['beta2'] + 1).abs().argsort()[:3]]

    print(f'Best speed row: {best_speed_rows}')
    print(f'Best distance row: {best_distance_rows}')

    # Plot the data
    speed_harmonic = best_speed_rows.iloc[0]['harmonic']
    speed_angle = best_speed_rows.iloc[0]['angle']
    speed_propeller = best_speed_rows.iloc[0]['propeller']
    speed_coeffs = [
        best_speed_rows.iloc[0]['beta0'],
        best_speed_rows.iloc[0]['beta1'],
        best_speed_rows.iloc[0]['beta2']
    ]

    speed_df = filter_df(hdf, {
        'propeller': speed_propeller,
        'angle': speed_angle,
        'harmonic': speed_harmonic
    })

    distance_harmonic = best_distance_rows.iloc[0]['harmonic']
    distance_angle = best_distance_rows.iloc[0]['angle']
    distance_propeller = best_distance_rows.iloc[0]['propeller']
    distance_coeffs = [
        best_distance_rows.iloc[0]['beta0'],
        best_distance_rows.iloc[0]['beta1'],
        best_distance_rows.iloc[0]['beta2']
    ]

    dist_df = filter_df(hdf, {
        'propeller': distance_propeller,
        'angle': distance_angle,
        'harmonic': distance_harmonic
    })

    # 3D plot the data

    plot_3d_regression_scatter(
        speed_df,
        speed_coeffs
    )

    plot_3d_regression_scatter(
        dist_df,
        distance_coeffs
    )    

if __name__ == "__main__":
    # Load the data
    lookup_df = parse_lookup_df('app/results')

    cal_data = load_cell_calibration()

    aero_coeffs = calc_aero_coefficients(
        'app/results',
        cal_data
    )

    hdf = parse_harmonic_df(lookup_df, aero_coeffs)

    sdf = parse_spl_df(lookup_df, aero_coeffs)

    hdf_filtered = filter_df(hdf,
                    { 'propeller' : '5045_s15'})
    
    coeff_df = speed_distance_SPL_regression(hdf_filtered)
    plot_best_coeffs_2D(hdf_filtered, coeff_df)

    # closeness metric for beta1 and beta2
    coeff_df['dbeta1'] = np.exp(- np.abs(coeff_df['beta1'] - 2) / 2)
    coeff_df['dbeta2'] = np.exp(- np.abs(coeff_df['beta2'] + 1) / 1)

    """
    fig, ax = multi_function_plot(coeff_df,
                        x_var='angle_bin',
                        y_var='harmonic',
                        filter_dict={
                            'propeller': '5045_s15',
                        },
                        #group_by='harmonic',
                        colour_by='dbeta1',
                        plot_type='scatter',
                        cmap='viridis',
                        colourbar_label='Speed coefficient closeness metric',
                        #marker = lambda df: ['o' if row['beta1'] > 2 else 'x' for _, row in df.iterrows()]
                        )
    
    
    fig, ax = multi_function_plot(coeff_df,
                        x_var='angle_bin',
                        y_var='harmonic',
                        filter_dict={
                            'propeller': '5045_s15',
                        },
                        #group_by='harmonic',
                        colour_by='dbeta2',
                        plot_type='scatter',
                        cmap='viridis',
                        colourbar_label='Distance coefficient closeness metric',
                        #marker = lambda df: ['o' if row['beta2'] > 1 else 'x' for _, row in df.iterrows()]
                        )
    """
    
    fig, ax = multi_function_plot(coeff_df,
                        x_var='harmonic',
                        y_var='beta1',
                        filter_dict={
                            'propeller': '5045_s15',
                        },
                        fig_size=(5,4),
                        group_by='angle_bin',
                        plot_type='scatter',
                        size_by='residual_std',
                        #marker = lambda df: ['o' if row['beta2'] > 1 else 'x' for _, row in df.iterrows()]
                        )
    ax.set_title('')
    ax.set_ylabel(r'$\beta_\Omega$ [-]')
    ax.set_xlabel(r'$m$ [-]')
    ax.legend().set_title('Angle [deg]')
    ax.axhline(2, color='black', linestyle='--', label='Expected')
    fig.savefig('deliverables/final_report/figures/speed_coefficient_for_angles_harmonics.png',
                    dpi = 300)
    
    fig, ax = multi_function_plot(coeff_df,
                        x_var='harmonic',
                        y_var='beta2',
                        filter_dict={
                            'propeller': '5045_s15',
                        },
                        group_by='angle_bin',
                        plot_type='scatter',
                        fig_size=(5,4),
                        size_by='residual_std',
                        #marker = lambda df: ['o' if row['beta2'] > 1 else 'x' for _, row in df.iterrows()]
                        )
    
    ax.set_title('')
    ax.set_ylabel(r'$\beta_r$ [-]')
    ax.set_xlabel(r'$m$ [-]')
    ax.axhline(-1, color='black', linestyle='--', label='Expected')
    ax.legend().set_title('Angle [deg]')
    fig.savefig('deliverables/final_report/figures/distance_coefficient_for_angles_harmonics.png',
                    dpi = 300)
    

    coeff_df2 = speed_distance_OASPL_regression(sdf)

    fig, ax = multi_function_plot(coeff_df2,
                        x_var='angle_bin',
                        y_var='beta1',
                        filter_dict={
                            'propeller': 'dalprop5045',
                        },
                        plot_type='scatter',
                        size_by='residual_std',
                        #marker = lambda df: ['o' if row['beta2'] > 1 else 'x' for _, row in df.iterrows()]
                        )

    fig, ax = multi_function_plot(coeff_df2,
                        x_var='angle_bin',
                        y_var='beta2',
                        filter_dict={
                            'propeller': 'dalprop5045',
                        },
                        plot_type='scatter',
                        size_by='residual_std',
                        #marker = lambda df: ['o' if row['beta2'] > 1 else 'x' for _, row in df.iterrows()]
                        )


    plt.show()


