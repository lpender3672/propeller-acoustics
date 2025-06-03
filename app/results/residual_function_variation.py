
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
    filter_df
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



def fixed_speed_distance_regression_oaspl(sdf):
    """
    For each group (propeller, angle_bin, harmonic):
      SPL ∝ k · speed^2 · distance^(-1)
    ⇒ log10(SPL) = intercept + 2·log10(speed) + (-1)·log10(distance) + error
    We fix the exponents and solve only for the intercept.
    """
    angle_tolerance = 22.5  # degrees
    half_width = angle_tolerance / 2
    bins = np.arange(-half_width, 181 + half_width + 1e-6, angle_tolerance)
    df = sdf.copy()
    df['angle_bin'] = pd.cut(df['angle'], bins=bins, labels=False) * angle_tolerance

    results = []
    for (propeller, angle_bin), group in df.groupby(['propeller', 'angle_bin']):
        if len(group) < 3:
            continue

        propf = f'app/props/{propeller}.prop'
        prop = load_prop_from_file(propf)

        omega = group['speed'].values * 2 * np.pi / 60   # rad/s
        r     = group['distance'].values # already in R/rt

        # Log10 values
        log_speed    = np.log10(omega)
        log_distance = np.log10(r)
        log_p      = group['OASPL'].values / 20  + np.log10(20e-6)
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
            'intercept':          intercept_db,
            'r_squared':          r_squared,
            'residual_std':       std_res,
            'reference' : group['reference'].values[0],
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
        fig_size=(8, 4),
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


def hdf_plots(hdfr, aero_coeffs):

    # plot harmonic
    fig, ax = multi_function_plot(
        hdfr,
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
        hdfr,
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
        group_by='propeller'
    )
    ax.set_title('')
    ax.set_xlabel('Harmonic [-]')
    ax.set_ylabel('$20 \log_{10} G()  $ [dB]')
    fig.savefig('deliverables/final_report/figures/harmonic_sound_sweep_for_135deg.pdf')

    # harmonic variation for different angles
    fig, ax = multi_function_plot(
        hdfr,
        'harmonic',
        'intercept',
        filter_dict={
            'propeller' : 'dalprop5045'
        },
        group_by='angle_bin',
    )
    ax.legend().set_title('Angle [deg]')
    ax.set_title('')
    ax.set_xlabel('Harmonic [-]')
    ax.set_ylabel('$20 \log_{10} G()  $ [dB]')
    fig.savefig('deliverables/final_report/figures/harmonic_sound_angle_for_dalprop5045.pdf')

    fig, ax = plot_sound_FOM(hdfr, aero_coeffs)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('')
    fig.tight_layout()
    #fig.savefig('deliverables/final_report/figures/prop_sound_FOM.pdf')


def plot_oaspl_ref_propellers(sdf):

    print(sdf.columns)
    print(sdf['reference'].values)

    fdf = filter_df(sdf, {'angle_bin':135,
                          'propeller' : [
                '5045_s15',
                '5045_s30',
                '5045_s45',
                '50loop_s50',
                'dalprop4045',
                'dalprop5045',
                'dalprop6045',
                'foxeer_toroidal',
                '50loop_s50',

            ]})
    
    # set fdf propeller row order
    fdf['propeller'] = pd.Categorical(
        fdf['propeller'],
        categories=[
            'dalprop4045',
            'dalprop5045',
            'dalprop6045',
            '5045_s15',
            '5045_s30',
            '5045_s45',
            'foxeer_toroidal',
            '50loop_s50',
        ],
        ordered=True
    )
    fdf = fdf.sort_values('propeller')

    # Create a figure and axis for the CT and CQ plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(fdf['propeller']))
    width = 0.7
    OASPL_bars = ax.bar(x, fdf['intercept'], width,
                        label='$20\log_{10} g$ [dB]',
                        color="#0095ff", alpha=0.7)
    ref_bars = ax.bar(x + width/4, fdf['reference'], width,  fdf['intercept'],
                       label=r'$20 \log_{10} \left(\frac{K_T}{C_T}\right)^{7/2} \left(\frac{C_Q}{K_Q} \right)^4$',
                         color="#ff6600", alpha=0.7)
    #ct_bars = ax.bar(x + width/2, fdf['OASPLref'], width, label='CT', color='#1f77b4')
    
    # plot a vline for the dalprop5045 reference
    dal5045_intercept = fdf.loc[fdf['propeller'] == 'dalprop5045', 'intercept'].values[0]
    ax.hlines(dal5045_intercept, x[0] - width/2, x[-1] + width/2, colors='gray', linestyles='--', label='Reference line',
              linewidth=1.5)

    ax.set_ylabel('Reference interference factor [dB]', fontsize=16)
    ax.set_xticks(x)
    #ax.set_xticklabels(fdf['propeller'], rotation=60)

    from app.results.graphing_tools import (
        place_tick_images
    )
    place_tick_images(
        fig, ax, [f'app/props/images/{p}.png' for p in fdf['propeller']],
        half_props=True
    )
    plt.yticks(fontsize=12)

    ax.legend(loc='upper left', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.set_ylim(5, -70)
    
    fig.tight_layout()
    fig.savefig(
        'deliverables/final_report/figures/OASPLref_bar.pdf',
    )

def plot_fm_oaspl_propellers(sdfr):

    fig, ax = multi_function_plot(
        sdfr,
        'FM',
        'intercept',
        filter_dict={
            'angle_bin': 135,
            'propeller' : [
                '5045_s15',
                '5045_s30',
                '5045_s45',
                '50loop_s50',
                #'dalprop4045',
                'dalprop5045',
                'dalprop5045bnr',
                'dalprop6045',
                'foxeer_toroidal',
                'printed5045',
                'printed5045bnr'
            ]
        },
        fig_size=(6, 4),
        group_by='propeller',
        max_groups=11
    )

    # set x and y labels
    ax.set_xlabel('FM [-]')
    ax.set_ylabel('Interference factor $20\log_{10}g()$ [dB]')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('')
    fig.tight_layout()

    fig.savefig('deliverables/final_report/figures/prop_sound_FOM.pdf')

    return fig, ax

def sdf_plots(sdfr, aero_data):

    
    angle = 135 # degrees
    
    fm_dict = {
        prop: Ct**(3/2) / (np.sqrt(2) * Cq)
        for prop, (Ct, Cq, _, _) in aero_data.items()
    }
    
    sdfr['FM'] = sdfr['propeller'].map(fm_dict)

    print(sdf.head())

    fig, ax = multi_function_plot(
        sdfr,
        'FM',
        'intercept',
        filter_dict={
            'angle_bin': angle,
        },
        fig_size=(8, 5),
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
    sdf = parse_spl_df(lookup_df, aero_coeffs)

    hdfr = fixed_speed_distance_regression(hdf)
    sdfr = fixed_speed_distance_regression_oaspl(sdf)

    hdf_plots(hdfr, aero_coeffs)
    sdf_plots(sdfr, aero_coeffs)

    plot_oaspl_ref_propellers(sdfr)
    plot_fm_oaspl_propellers(sdfr)

    plt.show()
