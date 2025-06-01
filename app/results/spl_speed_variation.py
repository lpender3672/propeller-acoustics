
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


from app.routines_audio import (
    parse_lookup_df,
    parse_spl_df,
    parse_harmonic_df,
)
from app.results.graphing_tools import (
    multi_function_plot,
    filter_df
)

from app.routines_aero import (
    calc_aero_coefficients,
    load_cell_calibration,
    merge_aero_coeffs
)


FREQUENCY = 51200  # Hz
TOTAL_CHANNELS = 7


def plot_rmsndp_speed(sdf):

    fig, ax = multi_function_plot(
        sdf,
        x_var="speed",
        y_var="OASPL",
        filter_dict={
            "propeller": "dalprop5045",
            "angle": 180,
        },
        group_by='distance',
        plot_type="scatter",
    )

    xlo, xhi = ax.get_xlim()
    xcont = np.linspace(xlo, xhi, 1000)
    ax.plot(xcont, -10 + 2 * 20 * np.log10(xcont), "k--", label="$x^2$")

    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.grid(True, which='both')

    return fig, ax

def plot_hrmsndp_speed(hdf):

    fig, ax = multi_function_plot(
        hdf,
        x_var="speed",
        y_var="SPL",
        filter_dict={
            "propeller": ["dalprop5045", "dalprop5045_nonlinear_test", "dalprop5045_nonlinear_test2"],
            "angle": 180,
            "harmonic": 1,
        },
        group_by='distance',
        plot_type="scatter",
    )

    xlo, xhi = ax.get_xlim()
    xcont = np.linspace(xlo, xhi, 1000)
    ycont = 20 * np.log10(1e-2 * xcont**2)
    ax.plot(xcont, ycont, "k--", label="$x^2$")

    ax.legend(loc="upper left").set_title("Distance [-]")
    ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.grid(True, which='both')

    return fig, ax


def plot_oaspl_ref_propellers(sdf):

    print(sdf.columns)
    print(sdf['reference'].values)

    fdf = filter_df(sdf, {'angle':135,
                          'distance' : (19, 21),
                          'speed' : (15000 * np.pi / 30, 17000* np.pi / 30)
                          })
    # agg
    fdf = fdf.groupby('propeller').agg({
        'OASPL' : 'first',
        'reference' : 'first'
    }).reset_index()
    
    # Create a figure and axis for the CT and CQ plot
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(fdf['propeller']))
    width = 0.35
    OASPL_bars = ax.bar(x, fdf['OASPL'], width, label='OASPL', color="#0095ff", alpha=0.7)
    ref_bars = ax.bar(x + width/4, fdf['reference'], width,  fdf['OASPL'],
                       label=r'$20 \log_{10} \left( \frac{K_Q}{K_T^2} \frac{C_T^2}{C_Q}  \right)$',
                         color="#ff6600", alpha=0.7)
    #ct_bars = ax.bar(x + width/2, fdf['OASPLref'], width, label='CT', color='#1f77b4')

    ax.set_xlabel('Propeller')
    ax.set_ylabel('$OASPL_{ref}$ [dB]')
    ax.set_xticks(x)
    ax.set_xticklabels(fdf['propeller'], rotation=45)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(60, 85)
    
    fig.tight_layout()
    fig.savefig(
        'deliverables/final_report/figures/OASPLref_bar.pdf',
    )


if __name__ == "__main__":

    aero_coeffs = calc_aero_coefficients(
        'app/results',
        load_cell_calibration()
    )

    ldf = parse_lookup_df('app/results/')

    sdf = parse_spl_df(ldf, aero_coeffs)
    hdf = parse_harmonic_df(ldf, aero_coeffs)

    fig, ax = plot_rmsndp_speed(sdf)
    ax.set_title("")
    ax.legend().set_title("Distance [-]")
    fig.savefig(
        'deliverables/final_report/figures/spl_speed_variation.pdf',
    )

    asdf = merge_aero_coeffs(sdf, aero_coeffs)

    plot_hrmsndp_speed(hdf)

    plot_oaspl_ref_propellers(asdf)
    plt.show()

