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

from app.hanson import (
    hanson_secondary_variables,
    hanson
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

    thickness = tf[np.newaxis, :] * np.ones((prop['nr'], 1))
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
    SPL = 20 * np.log10( np.abs(harmonic_noise) )
    return SPL


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

    # aero_coeffs is a dict
    # with key 'propeller' : [CQ, CT, std_CT, std_CQ]

    aero_coeffs_df = pd.DataFrame.from_dict(aero_coeffs, orient='index', columns=['CT', 'CQ', 'std_CT', 'std_CQ'])
    aero_coeffs_df.index.name = 'propeller'
    aero_coeffs_df.reset_index(inplace=True)

    aero_coeffs_df['FM'] = (aero_coeffs_df['CT'] ** (3/2)) / (np.sqrt(2) * aero_coeffs_df['CQ'])
    
    # For FM = CT^(3/2) / (sqrt(2) * CQ)
    # standard error propagation
    # dFM/dCT = 3/2 * CT^(1/2) / (sqrt(2) * CQ)
    # dFM/dCQ = -CT^(3/2) / (sqrt(2) * CQ^2)
    
    dFM_dCT = 3/2 * aero_coeffs_df['CT']**(1/2) / (np.sqrt(2) * aero_coeffs_df['CQ'])
    dFM_dCQ = -aero_coeffs_df['CT']**(3/2) / (np.sqrt(2) * aero_coeffs_df['CQ']**2)
    
    var_FM = (dFM_dCT**2 * aero_coeffs_df['std_CT']**2) + (dFM_dCQ**2 * aero_coeffs_df['std_CQ']**2)
    aero_coeffs_df['std_FM'] = np.sqrt(var_FM)

    # merge
    rdf = df.merge(aero_coeffs_df, on='propeller', how='left')
    return rdf


def useless_plots(rdf):

    fdf = filter_df(rdf, {'harmonic': 1,
                          'angle_bin':135,
                          'distance' : (19, 21)})
    
    # Create a figure and axis for the CT and CQ plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(fdf['propeller']))
    width = 0.35
    ct_bars = ax.bar(x - width/2, fdf['CT'], width, label='CT', color='#1f77b4')
    ax.errorbar(x - width/2, fdf['CT'], yerr=fdf['std_CT'], fmt='none', ecolor='black', capsize=5)
    cq_bars = ax.bar(x + width/2, fdf['CQ'], width, label='CQ', color='#ff7f0e')
    ax.errorbar(x + width/2, fdf['CQ'], yerr=fdf['std_CQ'], fmt='none', ecolor='black', capsize=5)
    
    ax.set_xlabel('Propeller')
    ax.set_ylabel('Coefficient [-]')
    ax.set_xticks(x)
    ax.set_xticklabels(fdf['propeller'], rotation=45)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    fig.savefig(
        'deliverables/final_report/figures/CT_CQ_bar.pdf',
    )

    # Create a figure and axis for the FM plot with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    fm_bars = ax.bar(x, fdf['FM'], width=0.5, color='#2ca02c')
    ax.errorbar(x, fdf['FM'], yerr=fdf['std_FM'], fmt='none', ecolor='black', capsize=5)
    
    ax.set_xlabel('Propeller')
    ax.set_ylabel('Figure of Merit [-]')
    ax.set_xticks(x)
    ax.set_xticklabels(fdf['propeller'], rotation=45)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    fig.savefig(
        'deliverables/final_report/figures/FM_bar.pdf',
    )
    
    fdf.plot.bar(
        x='propeller', y=['SPLhanson', 'intercept'], 
        title='Hanson Noise vs SPL for Harmonic 1 at 135 degrees'
    )

    fdf = filter_df(rdf, {'propeller': 'dalprop5045',
                          'angle_bin': 112.5,
                          'distance' : (10, 21)})
    fdf.plot.line(
        x='harmonic', y=['SPLhanson', 'intercept'], 
        title='Hanson Noise vs SPL for Dalprop 5045 at 135 degrees'
    )

    fdf = filter_df(rdf, {'propeller': 'dalprop5045',
                          'harmonic': 4,
                          'distance' : (10, 21)})
    fdf.plot.line(
        x='angle_bin', y=['SPLhanson', 'intercept'], 
        title='Hanson Noise vs SPL for Dalprop 5045 at 1st harmonic'
    )

def useful_plots(rdf):

    fig, ax = multi_function_plot(
        rdf.copy(),
        'harmonic',
        'SPLhanson',
        filter_dict={
            'propeller' : [
                'printed5045',
                '5045_s15',
                '5045_s30',
                '5045_s45'
            ],
            'angle_bin': 135
        },
        group_by='propeller'
    )


    fig, ax = multi_function_plot(
        rdf,
        x_var='FMbem',
        y_var='FM',
        group_by=['propeller'],
        title='Hanson Noise vs SPL for Harmonics',
        plot_type='scatter'
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig, ax = multi_function_plot(
        rdf,
        x_var='SPLhanson',
        y_var='intercept',
        group_by=['propeller'],
        title='Hanson Noise vs SPL for Harmonics',
        plot_type='scatter'
    )

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig, ax = multi_function_plot(
        rdf,
        'harmonic',
        'SPLhanson',
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
    ax.set_ylabel('$20 \log_{10} g()$ [dB]')
    fig.savefig('deliverables/final_report/figures/harmonic_hanson_sweep_for_135deg.pdf')

    fig, ax = multi_function_plot(
        rdf,
        'harmonic',
        'SPLhanson',
        filter_dict={
            'propeller' : 'dalprop5045'
        },
        group_by='angle_bin',
        fig_size=(8, 6),
    )
    ax.legend().set_title('Angle [deg]')
    ax.set_title('')
    ax.set_xlabel('Harmonic [-]')
    ax.set_ylabel('$20 \log_{10} g()  $ [dB]')
    ax.set_ylim(-160, -20)
    fig.savefig('deliverables/final_report/figures/harmonic_hanson_angle_for_dalprop5045.pdf')


def hanson_speed_dependence():
    
    
    av = AppVars()
    av.oper = load_oper_from_file("app/app_vars.json")
    av.airfoil_data = load_foil("app/foils/naca4412.surf")
    av.xfoil_data = run_xfoil(av.airfoil_data)

    propf = "app/props/printed5045.prop"
    av.prop = load_prop_from_file(propf)
    av = static_bem_swirl(av)

    fig,ax = plt.subplots()

    nspeeds = 1000
    speeds = np.logspace(1.5, 4.0, nspeeds)
    ms = [1, 6]

    sounds = np.zeros((nspeeds, len(ms)))

    for i,om in enumerate(speeds):
        av.oper['Omega'] = om

        SPL = hanson_noise( 
            av.prop,
            av.oper,
            av.airfoil_data, 
            av.res,
            20, 3 * np.pi / 4, ms=ms )

        #obs = {}
        #obs["r"] = [20 * av.prop["rt"]]
        #obs["theta"] = [3 * np.pi / 4]
        #oper, prop, obs = hanson_secondary_variables(av, obs)
        #PVm, PDm, PLm = hanson(oper, prop, obs, np.arange(1, 10))
        #SPL = np.log10(np.abs(PVm + PDm + PLm) / 20e-6)
        #print(SPL.shape)
        
        sounds[i, :] = SPL #+ 20 * np.log10(om**2)

    # plot the results
    for i, m in enumerate(ms):
        ax.plot(speeds, sounds[:, i], label=f'Harmonic {m}')

    ax.set_xscale('log')
    xlo, xhi = ax.get_xlim()
    xcont = np.logspace(np.log10(xlo), np.log10(xhi), 1000)
    ax.plot(xcont, - 20 * np.log10(xcont), 'k--', label='$-20 \log_{10}(\Omega)$')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.set_ylim(-120, -20)
    ax.set_xlabel('$\Omega$ [rad/s]')
    ax.set_ylabel('$\log_{10} g()$ [dB]')

    fig.savefig('deliverables/final_report/figures/hanson_speed_dependence.pdf', bbox_inches='tight')


def main():

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

    print(rdf)

    print(aero_coeffs)

    print(rdf.head())
    print(rdf.columns)

    useless_plots(rdf)
    useful_plots(rdf)

    rdf_filt = filter_df(rdf, {'harmonic': 1, 'angle_bin': 135, 'distance': (19, 21)})
    rdf_filt = rdf_filt.drop(columns=['distance', 'r_squared', 'residual_std', 'harmonic'])

    print(rdf_filt)

     

if __name__ == "__main__":

    main()
    #hanson_speed_dependence()
    plt.tight_layout()
    plt.show()







