import numpy as np
from scipy.special import jv as besselj
from scipy.integrate import trapz, simps, cumtrapz
from scipy.optimize import minimize

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from bem import betz_off_design, guaranteed_convergence_BEM
from hanson import (
    hanson, 
    calc_noise_components,
    hanson_secondary_variables
)

from routines import (
    load_oper_from_file,
    load_prop_from_file,
    interpolate_clcd,
    correct_clcd_sweep,
    AppVars,
    XFOIL_INSTALLED,
    run_xfoil
)

def Psi(kx, X, fX):
    f = fX * np.exp(1j * kx * X)
    ans = simps(f, x=X, axis=0)
    return ans

def radial_bessel(oper: dict, prop: dict):

    theta = 3 * np.pi / 4 # np.arange(0, np.pi, 0.01)

    fig, ax = plt.subplots()

    dopfac_o = 1 - oper['Mfl'] * np.cos(theta)
    for m in np.arange(1, 3):
        bess = besselj(m * prop['B'], m * prop['B'] * prop['r0_rt'] * oper['Mt'] * np.sin(theta) / dopfac_o)
        ax.plot(prop['r0_rt'], bess, label=f'm={m}')

    ax.legend()
    plt.show()

def get_radial_magnitudes(oper: dict, prop: dict, m: int):
    # secondary variables required

    theta = 3 * np.pi / 4 # np.arange(0, np.pi, 0.01)

    dopfac_o = 1 - oper['Mfl'] * np.cos(theta)
    r = 10 * prop['rt']
    y = r * np.sin(theta)
    omegaDop_o = oper['Omega'] / dopfac_o

    _, xc = np.meshgrid(prop['r0_rt'], prop['xc'])

    fac = 2 * m * prop['B'] / (oper['Mr'] * dopfac_o)
    yofac = oper['Mr'] ** 2 * np.cos(theta) - oper['Mx']

    kx = fac * prop['Bd'] * oper['Mt']
    ky = fac * yofac * prop['Bd'] / prop['r0_rt']
    phi0 = fac * yofac * prop['FA'] / (prop['r0_rt'] * 2 * prop['rt'])
    phis = fac * oper['Mt'] * prop['MCA'] / (2 * prop['rt'])

    # large term top of p5
    term1 = - (oper['rho'] * oper['c0']**2 * prop['B'] * np.sin(theta) * np.exp(1j * m * prop['B'] * (omegaDop_o * r / oper['c0'] - np.pi / 2))) / ((8 * np.pi * y / (2 * prop['rt']) * dopfac_o))

    bess = besselj(m * prop['B'], m * prop['B'] * prop['r0_rt'] * oper['Mt'] * np.sin(theta) / dopfac_o)
    
    term2 = oper['Mr']**2 * np.exp(1j * (phi0 + phis)) * bess
    terms1and2 = term1 * term2

    psiVKx = Psi(kx, xc, prop['HX'])
    psiLKx = Psi(kx, xc, prop['dCl_dxc'])
    psiDKx = Psi(kx, xc, prop['dCd_dxc'])

    I1 = terms1and2 * kx**2 * prop['tb'] * psiVKx
    I2 = terms1and2 * 1j * kx * prop['Cd_r'] / 2 * psiDKx
    I3 = terms1and2 * -1j * ky * prop['Cl_r'] / 2 * psiLKx

    PVm = cumtrapz(I1, prop['r0_rt'], initial=0)
    PDm = cumtrapz(I2, prop['r0_rt'], initial=0)
    PLm = cumtrapz(I3, prop['r0_rt'], initial=0)

    return PVm, PLm, PDm

def plot_harmonic_components(oper: dict, prop: dict, ms = np.arange(1,5), ax=None, hatching=None, label=None, w = 0.5, dx = 0):

    totals = np.zeros(ms.shape)

    PVms = np.zeros(ms.shape)
    PLms = np.zeros(ms.shape)
    PDms = np.zeros(ms.shape)
    for i,m in enumerate(ms):
        PVm, PLm, PDm = get_radial_magnitudes(oper, prop, m)
        PVms[i] = np.log10(np.abs(PVm[-1]))
        PLms[i] = np.log10(np.abs(PLm[-1]))
        PDms[i] = np.log10(np.abs(PDm[-1]))

    if ax is None:
        fig, ax = plt.subplots()

    #colors = ["#666666", "#999999", "#CCCCCC"]
    colors = ['c', 'm', 'y']
    alphas = [0.4, 0.3, 0.4]
    mpl.rcParams['hatch.linewidth'] = 3.0  # previous svg hatch linewidth

    cumulative_heights = np.zeros(ms.shape)
    for i,P in enumerate([PVms, PLms, PDms]):
        bar = ax.bar(ms + dx, P, w, bottom=cumulative_heights, color=colors[i], label=label, hatch=hatching)
        cumulative_heights += P
        for rect in bar:
            rect.set_edgecolor(colors[i])
            clr = rect.get_facecolor()
            rect.set_edgecolor((clr[0], clr[1], clr[2], alphas[i] + 0.2))
            rect.set_facecolor((clr[0], clr[1], clr[2], alphas[i]))

    for x, total in zip(ms + dx, cumulative_heights):
        valignment = 'top' if total < 0 else 'bottom'
        y_position = total + 0.05 if total >= 0 else total - 0.05
        ax.text(
            x, y_position, f"{total:.2f}",  # Add a slight offset above the bar
            ha="center", va=valignment, fontsize=10
        )
        ax.hlines(total, x - w/2, x + w/2, color='k', linestyle='-', linewidth=1)

    return ax


def optimise_lift_magnitude(av, m, ax=None, plot=True, colour=None):

    avc = av.copy()

    def objective_function(x):
        avc.prop['sweep'] = x[0] * avc.prop['r0_rt'] + x[1] * avc.prop['r0_rt']**2
        range_penalty = np.max(avc.prop['sweep']) - np.min(avc.prop['sweep'])
        oper, prop, _ = hanson_secondary_variables(avc)
        _, PLm, _ = get_radial_magnitudes(oper, prop, m)
        nois = np.log10(np.abs(PLm[-1]))
        return nois
    
    # get initial magnitude and sweep
    x0 = [0.01, 0.01]
    res = minimize(objective_function, x0 ,method='Nelder-Mead', options={'disp': True})
    print(res)

    x = res.x
    av.prop['sweep'] = x[0] * av.prop['r0_rt'] + x[1] * av.prop['r0_rt']**2

    if not plot:
        return
    
    oper, prop, _ = hanson_secondary_variables(avc)
    ax = radial_locus(oper, prop, ax, m=m, colour=colour)
    ax[0,0].set_title('Thickness')
    ax[0,1].set_title('Lift')
    ax[1,0].set_title('Drag')
    ax[1,1].set_title('Total')

    sfig, sax = plt.subplots()
    sax.plot(av.prop['r0_rt'], av.prop['sweep'] * 180 / np.pi)

def optimise_lift_harmonic_ratio(av, m1, m2, plot=True):
    """maximise the ratio of two harmonics of lift at specific observer location
    """
    avc = av.copy()

    def objective_function(x):
        avc.prop['sweep'] = x[0] * avc.prop['r0_rt'] + x[1] * avc.prop['r0_rt']**2
        oper, prop, _ = hanson_secondary_variables(avc)
        range_penalty = np.max(prop['sweep']) - np.min(prop['sweep'])
        
        _, PLm1, _ = get_radial_magnitudes(oper, prop, m1)
        _, PLm2, _ = get_radial_magnitudes(oper, prop, m2)
        dnois = np.log10(np.abs(PLm1[-1])) - np.log10(np.abs(PLm2[-1]))
        return dnois

    # get initial magnitude and sweep
    x0 = [0.01, 0.01]
    res = minimize(objective_function, x0 ,method='Nelder-Mead', options={'disp': True})
    print(res)

    x = res.x
    av.prop['sweep'] = x[0] * av.prop['r0_rt'] + x[1] * av.prop['r0_rt']**2

    if not plot:
        return
    oper, prop, _ = hanson_secondary_variables(avc)
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    B = prop['B']
    ax = radial_locus(oper, prop, ax=ax, m=m1, label=f'mB={m1 * B} (Minimised)', colour='b')
    ax = radial_locus(oper, prop, ax=ax, m=m2, label=f'mB={m2 * B} (Maximised)', colour='r')
    
    ax[0,0].set_title('Thickness')
    ax[0,1].set_title('Lift')
    ax[1,0].set_title('Drag')
    ax[1,1].set_title('Total')

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)  # Extra space for the legend

    fig.savefig('deliverables/tms/figures/optimised_ratio.png', dpi=300)

    fig, ax = plt.subplots()
    ax.plot(prop['r0_rt'], prop['sweep'] * 180 / np.pi)


def radial_locus(oper: dict, prop: dict, ax=None, colour=None, label=None, m=1):

    PVm, PLm, PDm = get_radial_magnitudes(oper, prop, m)

    if ax is None:
        fig,ax = plt.subplots(2, 2)
    
    ax[0,0].quiver(PVm.real[:-1], PVm.imag[:-1], np.diff(PVm.real), np.diff(PVm.imag), angles='xy', scale_units='xy', scale=1, label=label, color=colour)
    ax[0,1].quiver(PLm.real[:-1], PLm.imag[:-1], np.diff(PLm.real), np.diff(PLm.imag), angles='xy', scale_units='xy', scale=1, color=colour)
    ax[1,0].quiver(PDm.real[:-1], PDm.imag[:-1], np.diff(PDm.real), np.diff(PDm.imag), angles='xy', scale_units='xy', scale=1, color=colour)
    total = PVm + PLm + PDm
    ax[1,1].quiver(total.real[:-1], total.imag[:-1], np.diff(total.real), np.diff(total.imag), angles='xy', scale_units='xy', scale=1, color=colour)

    for axi in ax.flatten():
        axi.grid(True)

    return ax


def chord_locus(oper: dict, prop: dict, m, ax = None):

    theta = 3 * np.pi / 4 # np.arange(0, np.pi, 0.01)

    dopfac_o = 1 - oper['Mfl'] * np.cos(theta)
    r = 10 * prop['rt']
    y = r * np.sin(theta)

    _, xc = np.meshgrid(prop['r0_rt'], prop['xc'])

    fac = 2 * m * prop['B'] / (oper['Mr'] * dopfac_o)
    yofac = oper['Mr'] ** 2 * np.cos(theta) - oper['Mx']

    kx = fac * prop['Bd'] * oper['Mt']
    ky = fac * yofac * prop['Bd'] / prop['r0_rt']

    dpsiVKx = prop['HX'] * np.exp(1j * kx * xc)
    dpsiLKx = prop['dCl_dxc'] * np.exp(1j * kx * xc)
    dpsiDKx = prop['dCd_dxc'] * np.exp(1j * kx * xc)

    I1plt = cumtrapz(dpsiVKx, xc, axis=0, initial=0)
    I2plt = cumtrapz(dpsiLKx, xc, axis=0, initial=0)
    I3plt = cumtrapz(dpsiDKx, xc, axis=0, initial=0)
    total = I1plt + I2plt + I3plt

    if ax is None:
        fig,ax = plt.subplots(2, 2)

    viridis = plt.get_cmap('viridis')
    step = 1
    clrs = viridis(prop['r0_rt'][::step])

    for j in range(prop['nr']//step):

        ax[0,0].quiver(I1plt[::step,j].real[:-1], I1plt[::step,j].imag[:-1], np.diff(I1plt[::step,j].real), np.diff(I1plt[::step,j].imag), angles='xy', scale_units='xy', scale=1, color=clrs[j])
        ax[0,1].quiver(I2plt[::step,j].real[:-1], I2plt[::step,j].imag[:-1], np.diff(I2plt[::step,j].real), np.diff(I2plt[::step,j].imag), angles='xy', scale_units='xy', scale=1, color=clrs[j])
        ax[1,0].quiver(I3plt[::step,j].real[:-1], I3plt[::step,j].imag[:-1], np.diff(I3plt[::step,j].real), np.diff(I3plt[::step,j].imag), angles='xy', scale_units='xy', scale=1, color=clrs[j])
        ax[1,1].quiver(total[::step,j].real[:-1], total[::step,j].imag[:-1], np.diff(total[::step,j].real), np.diff(total[::step,j].imag), angles='xy', scale_units='xy', scale=1, color=clrs[j])

    return ax
    

def chord_locus_alpha(av : AppVars, alpha, m, ax = None):

    fig,ax = plt.subplots(2, 2)
    theta = 3 * np.pi / 4 # np.arange(0, np.pi, 0.01)

    viridis = plt.get_cmap('viridis')
    step = 1
    alpha_norm = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
    clrs = viridis(alpha_norm)

    for i,a in enumerate(alpha):
        av.res['alpha'][0] = a * np.pi / 180
        oper, prop, _ = hanson_secondary_variables(av)

        dopfac_o = 1 - oper['Mfl'] * np.cos(theta)
        r = 10 * prop['rt']
        _, xc = np.meshgrid(prop['r0_rt'], prop['xc'])

        fac = 2 * m * prop['B'] / (oper['Mr'] * dopfac_o)
        kx = fac * prop['Bd'] * oper['Mt']

        dpsiVKx = prop['HX'] * np.exp(1j * kx * xc)
        dpsiLKx = prop['dCl_dxc'] * np.exp(1j * kx * xc)
        dpsiDKx = prop['dCd_dxc'] * np.exp(1j * kx * xc)

        I1plt = cumtrapz(dpsiVKx, xc, axis=0, initial=0)
        I2plt = cumtrapz(dpsiLKx, xc, axis=0, initial=0)
        I3plt = cumtrapz(dpsiDKx, xc, axis=0, initial=0)
        total = I1plt + I2plt + I3plt

        ax[0,0].quiver(I1plt[:,0].real[:-1], I1plt[::step,0].imag[:-1], np.diff(I1plt[::step,0].real), np.diff(I1plt[::step,0].imag), angles='xy', scale_units='xy', scale=1, color=clrs[i])
        ax[0,1].quiver(I2plt[:,0].real[:-1], I2plt[::step,0].imag[:-1], np.diff(I2plt[::step,0].real), np.diff(I2plt[::step,0].imag), angles='xy', scale_units='xy', scale=1, color=clrs[i])
        ax[1,0].quiver(I3plt[:,0].real[:-1], I3plt[::step,0].imag[:-1], np.diff(I3plt[::step,0].real), np.diff(I3plt[::step,0].imag), angles='xy', scale_units='xy', scale=1, color=clrs[i])
        ax[1,1].quiver(total[:,0].real[:-1], total[::step,0].imag[:-1], np.diff(total[::step,0].real), np.diff(total[::step,0].imag), angles='xy', scale_units='xy', scale=1, color=clrs[i])

    return ax

def plot_hansen(oper: dict, prop: dict):
    # requires secondary variables

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    obs = {
        'r': 10 * prop['rt'] * np.ones((100)),
        'theta': np.linspace(0, np.pi, 100)
    }

    out = hanson(oper, prop, obs, np.arange(1, 5))
    V, L, D = calc_noise_components(out, oper['pref'])

    ax.plot(obs['theta'], V)
    ax.plot(obs['theta'], L)
    ax.plot(obs['theta'], D)

    ax.set_ylim(-90, 100)
    ax.set_theta_zero_location("N")
    ax.set_xlim(0, np.pi)

def hanson_sweep(av):

    prop = av.prop
    oper = av.oper

    fig, ax = plt.subplots()

    obs = {
        'r': [10 * prop['rt']],
        'theta': [3 * np.pi / 4]
    }
    max_sweep = np.pi/2
    nsweeps = 100
    sweep = np.linspace(-max_sweep, max_sweep, nsweeps)
    SPL = np.zeros((nsweeps, 4))
    for i in range(nsweeps):
        prop['sweep'] = np.linspace(0, sweep[i], prop['nr'])
        
        oper, prop, _ = hanson_secondary_variables(av)

        out = hanson(oper, prop, obs, np.arange(1, 5))
        SPL[i] = calc_noise_components(out, oper['pref'])

    #ax.plot(sweep, SPL[1], label=['Thickness', 'Lift', 'Drag'])
    ax.plot(sweep, SPL[:,1], label='Lift')
    ax.set_xlabel('Tip Sweep [rad]')
    ax.set_ylabel('SPL [dB]')

def radial_locus_sweep(av):
    prop = av.prop
    oper = av.oper

    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    ax[0,0].set_title('Thickness')
    ax[0,1].set_title('Lift')
    ax[1,0].set_title('Drag')
    ax[1,1].set_title('Total')

    # set equal aspect ratio
    for axi in ax.flatten():
        axi.set_aspect('equal', 'box')

    max_sweep = np.pi/4
    nsweeps = 8
    sweep = np.linspace(0, max_sweep, nsweeps)
    SPL = np.zeros((nsweeps, 3))
    for i in range(nsweeps):
        prop['sweep'] = np.linspace(0, sweep[i], prop['nr'])
        oper, prop, _ = hanson_secondary_variables(av)

        colour = cm.jet(i/nsweeps)
        ax = radial_locus(oper, prop, ax=ax, colour=colour, label=f'$\psi= {sweep[i]*180/np.pi:.2f}$')

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)  # Extra space for the legend

def operating_range(av, ):

    Js = np.linspace(-0.1, 0.2, 100)
    CPs = np.zeros(Js.shape)
    CTs = np.zeros(Js.shape)
    FMs = np.zeros(Js.shape)

    for i,J in enumerate(Js):
        av.oper['V'] = J * av.oper['Omega'] * av.prop['rt']
        av = betz_off_design(av)

        if not av.res['converged']:
            CPs[i] = np.nan
            CTs[i] = np.nan
            FMs[i] = np.nan
            continue

        ivlds = av.res['invalids']
        if (len(ivlds) > 0 and ivlds[-1] > av.prop['nr'] // 2):
            CPs[i] = np.nan
            CTs[i] = np.nan
            FMs[i] = np.nan
            continue

        CPs[i] = av.res['CP']
        CTs[i] = av.res['CT']
        FMs[i] = av.res['FM']

    fig, ax = plt.subplots()
    ax.plot(Js[CTs > 0], CPs[CTs > 0], 'o-', label='CP')
    ax.plot(Js[CTs > 0], CTs[CTs > 0], 'o-', label='CT')

    ax.legend()
    ax.grid()

    fig, ax = plt.subplots()
    ax.plot(Js[CTs > 0],  (FMs)[CTs > 0], label='FM')

    ax.legend()
    ax.grid()

    plt.show()

def plot_directivity(oper, prop, ms=1, obsr_rt=10, thetamin=0, thetamax=180, ax=None, label=None):

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.legend()

    obs = {
        'r': obsr_rt * prop['rt'] * np.ones((100)),
        'theta': np.linspace(thetamin, thetamax, 100) * np.pi / 180
    }
    ms = np.array(ms)

    out = hanson(oper, prop, obs, ms)
    V, L, D, total = calc_noise_components(out, oper['pref'])

    ax.plot(obs['theta'], L, label=label)

    ax.set_xlabel('Observer Angle [deg]')
    ax.set_ylabel('SPL [dB]')
    

    return ax


def plot_optimised_harmonics(av):
    total_width = 0.8
    n = 5
    width = total_width / n
    hatchings = [r'/', r'.. ', r'*', r"xx", r"//"]
    ms = np.arange(1, n)
    fig, bar_ax = plt.subplots(figsize=(12, 6))
    dfig, dax = plt.subplots(subplot_kw={'projection': 'polar'})
    sfig, sweep_ax = plt.subplots()

    opt_labels = ['Unswept', 'min $P_{1B}$', 'min $P_{2B}$', 'min $P_{3B}$', 'min $P_{4B}$']
    profile_colours = cm.viridis(np.linspace(0, 1, n))

    for i in range(n):
        dx = i * width - total_width / 2 + width / 2
        oper, prop, _ = hanson_secondary_variables(av)
        sweep_deg_zero = (prop['sweep'] - prop['sweep'][0]) * 180 / np.pi
        sweep_ax.plot(prop['r0_rt'], sweep_deg_zero, label=opt_labels[i], color=profile_colours[i])
        bar_ax = plot_harmonic_components(oper, prop, ms=ms, ax=bar_ax, hatching=hatchings[i], w=width, dx=dx)
        dax = plot_directivity(oper, prop, ms=ms, ax=dax, obsr_rt=10, thetamin=0, thetamax=180, label=opt_labels[i])
        #optimise_lift_harmonic_ratio(oper, prop, 2, 1)
        optimise_lift_magnitude(av, i+1, plot=False)


    bar_ax.grid()
    # create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hatchpatches = [Patch(facecolor='white', edgecolor='black', hatch=h) for h in hatchings]
    component_colours = ['c', 'm', 'y']
    component_patches = [Line2D([0], [0], color=c, linewidth=5) for c in component_colours]

    bar_ax.legend(hatchpatches + component_patches,  opt_labels + ['Thickness', 'Lift', 'Drag'], loc='lower left')
    bar_ax.set_xlabel('m')
    bar_ax.set_xticks(ms)
    bar_ax.set_ylabel('dB')
    bar_ax.set_title('Harmonic Noise Components')

    fig.tight_layout()
    fig.savefig('deliverables/tms/figures/optimised_harmonics.png', dpi=300)

    dax.set_ylim(50, 180)
    dax.set_xlim(0, np.pi)
    dax.set_theta_zero_location("N")
    # ensure legend is on the right
    dfig.legend(loc='center right', ncol=1, frameon=False)
    dfig.tight_layout()
    dfig.subplots_adjust(left=0, right=0.8)
    dfig.savefig('deliverables/tms/figures/optimised_directivity.png', dpi=300)

    sweep_ax.legend()
    sweep_ax.set_xlabel('$r_0/r_t$')
    sweep_ax.set_ylabel('Sweep [deg]')
    sweep_ax.set_title('Optimised Sweep')
    sweep_ax.grid()

    sfig.tight_layout()
    sfig.savefig('deliverables/tms/figures/optimised_harmonic_profiles.png', dpi=300)



def main():
    
    prop = load_prop_from_file('app/props/constant_chord.prop')
    oper = load_oper_from_file('app/app_vars.json')
    airfoil_data = np.loadtxt(prop['foil_path'])
    
    av = AppVars()
    av.oper = oper
    av.prop = prop

    airfoil_data = run_xfoil(airfoil_data)
    av.airfoil_data = airfoil_data

    av = betz_off_design(av)

    if not av.res['converged']:
        print("BEM failed")
        return
    
    oper, prop, _ = hanson_secondary_variables(av)

    #plot_optimised_harmonics(av)

    #operating_range(av)
    #optimise_lift_harmonic_ratio(av, 2, 1)
    fig, axes = plt.subplots(2, 2)
    m = 2
    axes = radial_locus(oper, prop, axes, m=m, colour='b')
    optimise_lift_magnitude(av, m, axes, colour='r')
    fig.tight_layout()
    
    chord_locus_alpha(av, np.arange(1,10), 1)
    hanson_sweep(av)

    plt.show()


if __name__ == "__main__":
    main()