
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from app.routines_audio import (
    parse_lookup_df,
    parse_spl_df,
    parse_harmonic_df
)
from app.routines import (
    load_prop_from_file
)
from app.results.graphing_tools import (
    multi_function_plot,
)

from app.routines_aero import (
    calc_aero_coefficients,
    load_cell_calibration
)

def fixed_speed_distance_regression(hdf):
    """
    For each group (propeller, angle_bin, harmonic):
      SPL ∝ k · speed^2 · distance^(-1)
    ⇒ log10(SPL) = intercept + 2·log10(speed) + (-1)·log10(distance) + error
    We fix the exponents and solve only for the intercept.
    """
    angle_tolerance = 22.5  # degrees
    half_width = angle_tolerance / 2
    bins = np.arange(-half_width, 181 + half_width + 1e-6, angle_tolerance)
    df = hdf.copy()
    df['angle_bin'] = pd.cut(df['angle'], bins=bins, labels=False) * angle_tolerance

    results = []
    for (propeller, angle_bin, harmonic), group in df.groupby(['propeller', 'angle_bin', 'harmonic']):
        if len(group) < 3:
            continue

        propf = f'app/props/{propeller}.prop'
        prop = load_prop_from_file(propf)

        omega = group['speed'].values * 2 * np.pi / 60   # rad/s
        r     = group['distance'].values # already in R/rt

        # Log10 values
        log_speed    = np.log10(omega)
        log_distance = np.log10(r)
        log_p      = group['SPLref'].values / 20  + np.log10(20e-6)
        log_radius = np.log10(prop['rt'])  # m

        # only -2log radius because distance is already in R/rt
        adjusted = log_p - 2*log_speed + log_distance - 2*log_radius

        # Fit intercept only
        intercept = np.mean(adjusted)
        pred = np.full_like(adjusted, intercept)
        residuals = adjusted - intercept

        var_adj = np.var(adjusted)
        var_res = np.var(residuals)
        r_squared = 1 - var_res/var_adj if var_adj != 0 else np.nan

        std_res   = np.std(residuals, ddof=0)
        std_adj   = np.std(adjusted, ddof=0)
        std_norm  = std_res/std_adj if std_adj != 0 else np.inf

        intercept_db = 20 * intercept # convert back to dB

        results.append({
            'propeller':          propeller,
            'angle_bin':          angle_bin,
            'harmonic':           harmonic,
            'intercept':          intercept_db,
            'r_squared':          r_squared,
            'residual_std':       std_res,
            #'residual_std_norm':  std_norm
        })

    # merge back on to df
    results = pd.DataFrame(results)

    return results

def plot_sound_FOM(hdf, aero_data):

    angle = 135 # degrees
    harmonic = 1
    
    fm_dict = {
        prop: Ct**(3/2) / (np.sqrt(2) * Cq)
        for prop, (Ct, Cq, _, _) in aero_data.items()
    }
    
    hdf['FM'] = hdf['propeller'].map(fm_dict)

    print(hdf.head())

    fig, ax = multi_function_plot(
        hdf,
        'FM',
        'intercept',
        filter_dict={
            'harmonic': harmonic,
            'angle_bin': angle,
        },
        group_by='propeller'
    )

    # set x and y labels
    ax.set_xlabel('FM [-]')
    ax.set_ylabel('$20 \log_{10} G()  $ [dB]')

    # get xlim and ylim
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xcont = np.linspace(xlo, xhi, 200)
    ycont = np.linspace(ylo, yhi, 200)
    xmesh, ymesh = np.meshgrid(xcont, ycont)

    PM = xmesh / (10 ** (ymesh / 20))
    # contour with labels
    CS = ax.contour(xmesh, ymesh, PM, colors='gray')
    ax.clabel(CS, inline=True, fontsize=10, fmt='%.2f')

    return fig, ax


if __name__ == "__main__":
    # Load data
    lookup_df = parse_lookup_df('app/results')

    cal_data = load_cell_calibration()

    aero_coeffs = calc_aero_coefficients(
        'app/results',
        cal_data
    )

    hdf = parse_harmonic_df(lookup_df, aero_coeffs, harmonics=10)

    results = fixed_speed_distance_regression(hdf)

    # plot harmonic
    fig, ax = multi_function_plot(
        results,
        'angle_bin',
        'intercept',
        filter_dict={
            'harmonic': 4,
            'propeller' : [
                'dalprop5045',
                'printed5045',
                '5045_s15',
                '5045_s30',
                '5045_s45'
            ]
        },
        group_by='propeller'
    )

    # harmonic variation for different propellers
    fig, ax = multi_function_plot(
        results,
        'harmonic',
        'intercept',
        filter_dict={
            'propeller' : [
                'dalprop5045',
                'printed5045',
                '5045_s15',
                '5045_s30',
                '5045_s45'
            ],
            'angle_bin': 135
        },
        fig_size=(8, 6),
        group_by='propeller'
    )
    ax.set_title('')
    ax.set_xlabel('Harmonic [-]')
    ax.set_ylabel('$20 \log_{10} G()  $ [dB]')
    fig.savefig('deliverables/final_report/figures/harmonic_sound_sweep_for_135deg.pdf')

    # harmonic variation for different angles
    fig, ax = multi_function_plot(
        results,
        'harmonic',
        'intercept',
        filter_dict={
            'propeller' : 'dalprop5045'
        },
        group_by='angle_bin',
        fig_size=(8, 6)
    )
    ax.legend().set_title('Angle [deg]')
    ax.set_title('')
    ax.set_xlabel('Harmonic [-]')
    ax.set_ylabel('$20 \log_{10} G()  $ [dB]')
    fig.savefig('deliverables/final_report/figures/harmonic_sound_angle_for_dalprop5045.pdf')


    fig, ax = plot_sound_FOM(results, aero_coeffs)
    fig.savefig('deliverables/final_report/figures/prop_sound_FOM.pdf')

    plt.show()
