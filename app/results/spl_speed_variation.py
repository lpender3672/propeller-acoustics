
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


from app.routines_audio import (
    parse_lookup_df,
    parse_spl_df,
    parse_harmonic_df
)
from app.results.graphing_tools import (
    multi_function_plot,
)

from app.routines_aero import (
    calc_aero_coefficients,
    load_cell_calibration
)


FREQUENCY = 51200  # Hz
TOTAL_CHANNELS = 7


def plot_rmsndp_speed(sdf):

    fig, ax = multi_function_plot(
        sdf,
        x_var="speed",
        y_var="OASPL",
        filter_dict={
            "propeller": ["dalprop5045", "dalprop5045_nonlinear_test", "dalprop5045_nonlinear_test2"],
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
    ax.plot(xcont, 1e-2 * xcont**2, "k--", label="$x^2$")

    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which='both')


if __name__ == "__main__":

    aero_coeffs = calc_aero_coefficients(
        'app/results',
        load_cell_calibration()
    )

    ldf = parse_lookup_df('app/results/')

    sdf = parse_spl_df(ldf, aero_coeffs)
    hdf = parse_harmonic_df(ldf, aero_coeffs)

    plot_rmsndp_speed(sdf)
    plot_hrmsndp_speed(hdf)

    plt.show()

