

# this file will assume quadratic chordwise distributions
# and perform the static swirl analysis for lift and drag.

import matplotlib as mpl
import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt
from PyQt6.QtWidgets import QApplication

from scipy.integrate import (
    cumulative_trapezoid, simpson, trapezoid
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

from app.bem import static_bem_swirl

def Psi(kx, X, fX):
    kx = kx.reshape(-1, 1)
    f = fX * np.exp(1j * kx * X)
    ans = simpson(f, x=X, axis=1)
    return ans

def hanson_noise(prop, oper, airfoil_data, bem_res, r, theta, ms = np.arange(1, 10)):

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
            * (prop["rt"] * oper['Omega'])**2
            * prop["B"]
            * np.exp(1j * m * prop["B"] * (r / oper["c0"] - np.pi / 2))
        ) / (4 * np.pi * (r / prop['rt']) + 1e-10)

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
    
    REF = 20e-6
    SPL = 10 * np.log10(harmonic_noise * np.conj(harmonic_noise) / REF**2)
    return SPL


def parse_tdf(hdf):
    angle_tolerance = 22.5
    half_width = angle_tolerance / 2
    bins = np.arange(-half_width, 181 + half_width + 1e-6, angle_tolerance)
    df = hdf.copy()
    df['angle_bin'] = hdf.cut(df['angle'], bins=bins, labels=False) * angle_tolerance
    
    # Aggregate experimental data first
    hdf_agg = df.groupby(['propeller', 'angle_bin', 'distance', 'harmonic']).agg({
        'SPL': 'mean',  # or whatever aggregation you want
        # add other columns you need
    }).reset_index()
    
    av = AppVars()
    av.oper = load_oper_from_file("app/app_vars.json")
    av.airfoil_data = load_foil("app/foils/naca0018.surf")
    av.xfoil_data = run_xfoil(av.airfoil_data)
    
    dict_array = []
    for (propeller, angle_bin, distance), group in hdf_agg.groupby(['propeller', 'angle_bin', 'distance']):
        propf = f"app/props/{propeller}.prop"
        av.prop = load_prop_from_file(propf)
        av = static_bem_swirl(av)
        ms = group['harmonic'].unique()
        angle = angle_bin * np.pi / 180
        harmonic_noise = hanson_noise(
            av.prop, av.oper, av.airfoil_data, av.res,
            distance, angle, ms=ms
        )
        for i, m in enumerate(ms):  # use ms, not av.prop['ms']
            dict_array.append({
                'propeller': propeller,
                'angle_bin': angle_bin,
                'distance': distance,
                'harmonic': m,
                'SPLhanson': harmonic_noise[i]
            })
        


if __name__ == "__main__":
    
    ldf = parse_lookup_df("app/data/lookup.csv")
    hdf = parse_harmonic_df("app/data/harmonic.csv")
    tdf = parse_tdf(hdf)

    

