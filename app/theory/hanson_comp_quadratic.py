

# this file will assume quadratic chordwise distributions
# and perform the static swirl analysis for lift and drag.

import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib import pyplot as plt

from scipy.integrate import (
    simpson, trapezoid
)
from scipy.optimize import minimize
from scipy.special import jv as besselj

from app.routines import (
    sample_airfoil,
    AppVars,
    load_prop_from_file,
    load_oper_from_file,
    load_foil,
    run_xfoil
)

from app.routines_audio import (
    parse_lookup_df,
    parse_harmonic_df
)
from app.routines_aero import (
    calc_aero_coefficients,
    load_cell_calibration
)

from app.results.graphing_tools import (
    multi_function_plot,
    filter_df
)

from app.results.residual_function_variation import (
    fixed_speed_distance_regression
)

from app.bem import static_bem_swirl

def Psi(kx, X, fX):
    kx = kx.reshape(-1, 1)
    f = fX * np.exp(1j * kx * X)
    ans = simpson(f, x=X, axis=1)
    return ans

def hanson_noise(prop, oper, airfoil_data, bem_res, r_rt, theta, ms = np.arange(1, 10)):
    # this form of hansons noise is dimensionless
    # similar to g

    _, zf = sample_airfoil(airfoil_data, 2 * prop["nx"])
    tf = zf[: prop["nx"]] - zf[prop["nx"]:]
    tb = np.max(tf) * prop["c"]

    Cd = bem_res["Cd"]
    Cl = bem_res["Cl"]
    dx = prop["r0"] * np.sin(prop["sweep"])
    phi = prop["twist"] - bem_res["alpha"]
    FCA = dx * np.sin(phi)
    MCA = dx * np.cos(phi)
    b = prop['c']
    B = prop['B']
    z = prop["r0_rt"]
    r = r_rt * prop["rt"] # so we need the r term for an exponential

    harmonic_noise = np.zeros(len(ms), dtype=complex)

    _, xc = np.meshgrid(z, prop["xc"], indexing="ij")

    quadratic = xc[:, 0][:, np.newaxis]**2 - xc**2
    quadratic /= simpson(quadratic, xc, axis=1)[:, np.newaxis]

    _, thickness = np.meshgrid(b, tf, indexing="ij")
    thickness /= simpson(thickness, xc, axis=1)[:, np.newaxis]

    for i, m in enumerate(ms):

        kx = m * B * b
        ky = m * B * oper['Omega'] * b * np.cos(theta) / (z * oper['c0'])
        phi0 = ky * MCA / b
        phis = kx * FCA / b

        # large term top of p5
        term1 = -(
            oper["rho"]
            * prop["B"]
            * np.exp(1j * m * prop["B"] * (r / oper["c0"] - np.pi / 2))
        ) / (4 * np.pi)

        bess = besselj(
            m * prop["B"],
            m * prop["B"] * prop["r0_rt"] * oper["Omega"] * np.sin(theta) / oper['c0'],
        )
        term2 = z ** 2 * np.exp(1j * (phi0 + phis)) * bess

        terms1and2 = term1 * term2

        psiVKx = Psi(kx, xc, thickness)
        psiLKx = Psi(kx, xc, quadratic)
        psiDKx = Psi(kx, xc, quadratic)

        I1 = terms1and2 * kx**2 * tb * psiVKx
        I2 = terms1and2 * 1j * kx * Cd / 2 * psiDKx
        I3 = terms1and2 * -1j * ky * Cl / 2 * psiLKx

        harmonic_noise[i] = (
            trapezoid(I1, z) +
            trapezoid(I2, z) +
            trapezoid(I3, z) )
    
    # already non-dimensional
    SPL = 10 * np.log10(harmonic_noise * np.conj(harmonic_noise))
    return SPL - 20 * np.log10(20e-6)


def parse_tdf(hdf):
    angle_tolerance = 22.5
    half_width = angle_tolerance / 2
    bins = np.arange(-half_width, 181 + half_width + 1e-6, angle_tolerance)
    df = hdf.copy()
    df['angle_bin'] = pd.cut(df['angle'], bins=bins, labels=False) * angle_tolerance
    
    av = AppVars()
    av.oper = load_oper_from_file("app/app_vars.json")
    av.airfoil_data = load_foil("app/foils/naca4412.surf")
    av.xfoil_data = run_xfoil(av.airfoil_data)
    
    dict_array = []
    # i wanted to group over just propeller and angle_bin, but
    # the distance is needed for the hanson noise calculation
    # so we will aggregate over distance, selecting the maximum
    bem_results = {}  # Cache BEM results by propeller
    for propeller in df['propeller'].unique():
        propf = f"app/props/{propeller}.prop"
        av.prop = load_prop_from_file(propf)
        av = static_bem_swirl(av)
        bem_results[propeller] = {
            'CT': av.res['CT'],
            'CQ': av.res['CQ'], 
            'FM': av.res['FM'],
            'prop': av.prop.copy(),
            'res': av.res.copy()
        }

    for (propeller, angle_bin, distance), group in df.groupby(['propeller', 'angle_bin', 'distance']):
        
        bem_data = bem_results[propeller]

        ms = group['harmonic'].unique()
        angle = angle_bin * np.pi / 180
        harmonic_noise = hanson_noise(
            bem_data['prop'],
            av.oper,
            av.airfoil_data, 
            bem_data['res'],
            distance, angle, ms=ms
        )

        #for i, m in enumerate(ms):  # use ms, not av.prop['ms']
        for i,m in enumerate(ms):
            dict_array.append({
                'propeller': propeller,
                'angle_bin': angle_bin,
                'distance' : distance,
                'harmonic': m,
                'SPLhanson': harmonic_noise[i],
                'CTbem': bem_data['CT'],
                'CQbem': bem_data['CQ'],
                'FMbem': bem_data['FM']
            })

        
    tdf = pd.DataFrame(dict_array)
    # TODO need a way to aggregate theoretical distance

    return tdf

def merge_aero_coeffs(df, aero_coeffs):
    # merge the aero coefficients into the dataframe

    #aero_coeffs is a dict
    # with key 'propeller' : [CQ, CT]

    aero_coeffs_df = pd.DataFrame.from_dict(aero_coeffs, orient='index', columns=['CT', 'CQ'])
    aero_coeffs_df.index.name = 'propeller'
    aero_coeffs_df.reset_index(inplace=True)

    aero_coeffs_df['FM'] = (aero_coeffs_df['CT'] ** (3/2)) / (np.sqrt(2) * aero_coeffs_df['CQ'])

    # merge
    rdf = df.merge(aero_coeffs_df, on='propeller', how='left')
    return rdf

if __name__ == "__main__":

    aero_coeffs = calc_aero_coefficients(
        'app/results',
        load_cell_calibration()
    )
    
    ldf = parse_lookup_df('app/results')

    hdf = parse_harmonic_df(ldf, aero_coeffs)
    rdf = fixed_speed_distance_regression(hdf)
    tdf = parse_tdf(hdf)

    #'propeller', 'speed', 'angle', 'distance', 'harmonic', 'RMS', 'SPL',
    #'SPLref', 'CT', 'CQ', 'FM', 'angle_bin', 'SPLhanson', 'CTbem', 'CQbem',
    #'FMbem'

    # merge tdf and rdf
    rdf = rdf.merge(tdf, on=['propeller', 'angle_bin', 'harmonic'], how='left')
    rdf = merge_aero_coeffs(rdf, aero_coeffs)

    fdf = filter_df(rdf, {'harmonic': 1,
                          'angle_bin':135,
                          'distance' : (19, 21)})
    
    fdf.plot.bar(
        x='propeller', y=['FMbem', 'FM'], 
        title='Hanson Noise vs SPL for Harmonic 1 at 135 degrees'
    )

    fdf.plot.bar(
        x='propeller', y=['SPLhanson', 'intercept'], 
        title='Hanson Noise vs SPL for Harmonic 1 at 135 degrees'
    )

    fdf = filter_df(rdf, {'propeller': 'dalprop5045',
                          'angle_bin': 135,
                          'distance' : (19, 21)})
    fdf.plot.line(
        x='harmonic', y=['SPLhanson', 'intercept'], 
        title='Hanson Noise vs SPL for Dalprop 5045 at 135 degrees'
    )

    fdf = filter_df(rdf, {'propeller': 'dalprop5045',
                          'harmonic': 4,
                          'distance' : (19, 21)})
    fdf.plot.line(
        x='angle_bin', y=['SPLhanson', 'intercept'], 
        title='Hanson Noise vs SPL for Dalprop 5045 at 1st harmonic'
    )
    
    plt.show()








