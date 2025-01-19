import numpy as np
from scipy.special import jv as besselj
from scipy.integrate import trapz, simps, cumtrapz
from scipy.optimize import minimize

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from betz import betz_off_design, guaranteed_convergence_BEM
from hanson import hanson, calc_noise_components

from routines import (
    load_oper_from_file,
    load_prop_from_file,
    AppVars,
    XFOIL_INSTALLED,
    run_xfoil
)

def Psi(kx, X, fX):
    f = fX * np.exp(1j * kx * X)
    ans = simps(f, x=X, axis=0)
    return ans

def radial_bessel(oper: dict, prop: dict):

    oper['Min'] = 0;                  #% inflow Mach number [-]
    oper['Mfl'] = oper['V']/oper['c0'];                     #% flight Mach number [-] 
    oper['Mht'] = np.sqrt((oper['Omega']*2*np.pi*prop['rt'])**2 + oper['V']**2)/oper['c0'];                 #% Helical tip Mach number [-]

    oper['Mx'] = oper['Mfl'] + oper['Min']                       # Effective Mach number [-]
    oper['Mt'] = np.sqrt(oper['Mht']**2 - oper['Mx']**2)     # Tip mach number [-]
    oper['Mr'] = np.sqrt(oper['Mx']**2 + (oper['Mt']*prop['r0_rt'])**2) #  blade section Mach number [-]

    theta = 3 * np.pi / 4 # np.arange(0, np.pi, 0.01)

    fig, ax = plt.subplots()

    dopfac_o = 1 - oper['Mfl'] * np.cos(theta)
    for m in np.arange(1, 3):
        bess = besselj(m * prop['B'], m * prop['B'] * prop['r0_rt'] * oper['Mt'] * np.sin(theta) / dopfac_o)
        ax.plot(prop['r0_rt'], bess, label=f'm={m}')

    ax.legend()
    plt.show()

def get_radial_magnitudes(oper: dict, prop: dict, m: int):
    # calc dopfac for observer
    oper['Min'] = 0;                  #% inflow Mach number [-]
    oper['Mfl'] = oper['V']/oper['c0'];                     #% flight Mach number [-] 
    oper['Mht'] = np.sqrt((oper['Omega']*2*np.pi*prop['rt'])**2 + oper['V']**2)/oper['c0'];                 #% Helical tip Mach number [-]

    oper['Mx'] = oper['Mfl'] + oper['Min']                       # Effective Mach number [-]
    oper['Mt'] = np.sqrt(oper['Mht']**2 - oper['Mx']**2)     # Tip mach number [-]
    oper['Mr'] = np.sqrt(oper['Mx']**2 + (oper['Mt']*prop['r0_rt'])**2) #  blade section Mach number [-]

    dx = prop['r0_rt'] * prop['rt'] * np.sin( prop['sweep'] )
    phi = np.arcsin(oper['Mx'] / oper['Mr'])
    #phi = prop['twist'] - res['alpha']
    prop['FA'] = dx * np.sin(phi)
    prop['MCA'] = dx * np.cos(phi)

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
    colors = ['r', 'g', 'b']
    mpl.rcParams['hatch.linewidth'] = 3.0  # previous svg hatch linewidth

    cumulative_heights = np.zeros(ms.shape)
    for i,P in enumerate([PVms, PLms, PDms]):
        bar = ax.bar(ms + dx, P, w, bottom=cumulative_heights, color=colors[i], label=label, hatch=hatching)
        cumulative_heights += P
        for rect in bar:
            rect.set_edgecolor(colors[i])
            clr = rect.get_facecolor()
            rect.set_facecolor((clr[0], clr[1], clr[2], 0.5))

    for x, total in zip(ms + dx, cumulative_heights):
        valignment = 'top' if total < 0 else 'bottom'
        y_position = total + 0.05 if total >= 0 else total - 0.05
        ax.text(
            x, y_position, f"{total:.2f}",  # Add a slight offset above the bar
            ha="center", va=valignment, fontsize=10
        )

    return ax


def optimise_lift_magnitude(oper: dict, prop: dict, m, plot=True):


    def objective_function(x):
        prop['sweep'] = x[1] * prop['r0_rt'] + x[2] * prop['r0_rt']**2
        range_penalty = np.max(prop['sweep']) - np.min(prop['sweep'])
        
        _, PLm, _ = get_radial_magnitudes(oper, prop, m)
        nois = np.log10(np.abs(PLm[-1]))
        return nois
    
    # get initial magnitude and sweep
    x0 = [0.01, 0.01, 0.01]
    res = minimize(objective_function, x0 ,method='Nelder-Mead', options={'disp': True})
    print(res)

    if not plot:
        return
    
    ax = radial_locus(oper, prop)
    ax[0,0].set_title('Thickness')
    ax[0,1].set_title('Lift')
    ax[1,0].set_title('Drag')
    ax[1,1].set_title('Total')

    fig, ax = plt.subplots()
    x = res.x
    prop['sweep'] = x[0] + x[1] * prop['r0_rt'] + x[2] * prop['r0_rt']**2
    ax.plot(prop['r0_rt'], prop['sweep'] * 180 / np.pi)

def optimise_lift_harmonic_ratio(oper: dict, prop: dict, m1, m2):
    """maximise the ratio of two harmonics of lift at specific observer location
    """

    def objective_function(x):
        prop['sweep'] = x[0] + x[1] * prop['r0_rt'] + x[2] * prop['r0_rt']**2
        range_penalty = np.max(prop['sweep']) - np.min(prop['sweep'])
        
        _, PLm1, _ = get_radial_magnitudes(oper, prop, m1)
        _, PLm2, _ = get_radial_magnitudes(oper, prop, m2)
        dnois = np.log10(np.abs(PLm1[-1])) - np.log10(np.abs(PLm2[-1]))
        return dnois

    # get initial magnitude and sweep
    x0 = [0.01, 0.01, 0.01]
    res = minimize(objective_function, x0 ,method='Nelder-Mead', options={'disp': True})
    print(res)

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
    x = res.x
    prop['sweep'] = x[0] + x[1] * prop['r0_rt'] + x[2] * prop['r0_rt']**2
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


def chord_locus(oper: dict, prop: dict):

# calc dopfac for observer
    oper['Min'] = 0;                  #% inflow Mach number [-]
    oper['Mfl'] = oper['V']/oper['c0'];                     #% flight Mach number [-] 
    oper['Mht'] = np.sqrt((oper['Omega']*2*np.pi*prop['rt'])**2 + oper['V']**2)/oper['c0'];                 #% Helical tip Mach number [-]

    oper['Mx'] = oper['Mfl'] + oper['Min']                       # Effective Mach number [-]
    oper['Mt'] = np.sqrt(oper['Mht']**2 - oper['Mx']**2)     # Tip mach number [-]
    oper['Mr'] = np.sqrt(oper['Mx']**2 + (oper['Mt']*prop['r0_rt'])**2) #  blade section Mach number [-]

    theta = 3 * np.pi / 4 # np.arange(0, np.pi, 0.01)

    dopfac_o = 1 - oper['Mfl'] * np.cos(theta)
    r = 10 * prop['rt']
    y = r * np.sin(theta)

    _, xc = np.meshgrid(prop['r0_rt'], prop['xc'])

    for i, m in enumerate(np.arange(1, 2)):
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

        viridis = plt.get_cmap('viridis')
        clrs = viridis(prop['r0_rt'])

        fig, ax = plt.subplots()
        for j in range(prop['nr']):
            ax.plot(I1plt[:,j].real, I1plt[:,j].imag, color = clrs[j])
        ax.legend()
        
        fig, ax = plt.subplots()
        for j in range(prop['nr']):
            ax.plot(I2plt[:,j].real, I2plt[:,j].imag, color = clrs[j])
        ax.legend()
        
        fig, ax = plt.subplots()
        for j in range(prop['nr']):
            ax.plot(I3plt[:,j].real, I3plt[:,j].imag, color = clrs[j])
        ax.legend()

def plot_hansen(oper: dict, prop: dict):

    oper['Min'] = 0;                  #% inflow Mach number [-]
    oper['Mfl'] = oper['V']/oper['c0'];                     #% flight Mach number [-] 
    oper['Mht'] = np.sqrt((oper['Omega']*2*np.pi*prop['rt'])**2 + oper['V']**2)/oper['c0'];                 #% Helical tip Mach number [-]

    oper['Mx'] = oper['Mfl'] + oper['Min']                       # Effective Mach number [-]
    oper['Mt'] = np.sqrt(oper['Mht']**2 - oper['Mx']**2)     # Tip mach number [-]
    oper['Mr'] = np.sqrt(oper['Mx']**2 + (oper['Mt']*prop['r0_rt'])**2) #  blade section Mach number [-]

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

def hanson_sweep(oper: dict, prop: dict):

    oper['Min'] = 0;                  #% inflow Mach number [-]
    oper['Mfl'] = oper['V']/oper['c0'];                     #% flight Mach number [-] 
    oper['Mht'] = np.sqrt((oper['Omega']*2*np.pi*prop['rt'])**2 + oper['V']**2)/oper['c0'];                 #% Helical tip Mach number [-]

    oper['Mx'] = oper['Mfl'] + oper['Min']                       # Effective Mach number [-]
    oper['Mt'] = np.sqrt(oper['Mht']**2 - oper['Mx']**2)     # Tip mach number [-]
    oper['Mr'] = np.sqrt(oper['Mx']**2 + (oper['Mt']*prop['r0_rt'])**2) #  blade section Mach number [-]

    fig, ax = plt.subplots()

    obs = {
        'r': [10 * prop['rt']],
        'theta': [3 * np.pi / 4]
    }
    max_sweep = np.pi/2
    nsweeps = 100
    sweep = np.linspace(-max_sweep, max_sweep, nsweeps)
    SPL = np.zeros((nsweeps, 3))
    for i in range(nsweeps):
        prop['sweep'] = np.linspace(0, sweep[i], prop['nr'])
        dx = prop['r0_rt'] * prop['rt'] * np.sin( prop['sweep'] )
        phi = np.arcsin(oper['Mx'] / oper['Mr'])
        #phi = prop['twist'] - res['alpha']
        alpha = prop['twist'] - phi
        prop['FA'] = dx * np.sin(phi)
        prop['MCA'] = dx * np.cos(phi)
        prop['Cl_r'] = (2 * np.pi * alpha) #* np.cos(prop['sweep'])**2
        prop['Cd_r'] = (0.0087 - 0.021 * alpha + 0.400 * alpha ** 2) #* np.cos(prop['sweep'])**2

        out = hanson(oper, prop, obs, np.arange(1, 5))
        SPL[i] = calc_noise_components(out, oper['pref'])

    #ax.plot(sweep, SPL[1], label=['Thickness', 'Lift', 'Drag'])
    ax.plot(sweep, SPL[:,1], label='Lift')
    ax.set_xlabel('Tip Sweep [rad]')
    ax.set_ylabel('SPL [dB]')

def radial_locus_sweep(oper: dict, prop: dict):

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


def plot_optimised_harmonics(prop, oper):
    total_width = 0.8
    n = 5
    width = total_width / n
    hatchings = [r'/', r'.. ', r'*', r"xx", r"//"]
    ms = np.arange(1, n)
    fig, bar_ax = plt.subplots(figsize=(12, 6))

    sfig, sweep_ax = plt.subplots()
    opt_labels = ['Unswept', 'min $P_{1B}$', 'min $P_{2B}$', 'min $P_{3B}$', 'min $P_{4B}$']
    profile_colours = cm.viridis(np.linspace(0, 1, n))

    for i in range(n):
        dx = i * width - total_width / 2 + width / 2
        sweep_deg_zero = (prop['sweep'] - prop['sweep'][0]) * 180 / np.pi
        sweep_ax.plot(prop['r0_rt'], sweep_deg_zero, label=opt_labels[i], color=profile_colours[i])
        bar_ax = plot_harmonic_components(oper, prop, ms=ms, ax=bar_ax, hatching=hatchings[i], w=width, dx=dx)
        #optimise_lift_harmonic_ratio(oper, prop, 2, 1)
        optimise_lift_magnitude(oper, prop, i+1, plot=False)


    bar_ax.grid()
    # create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hatchpatches = [Patch(facecolor='white', edgecolor='black', hatch=h) for h in hatchings]
    component_colours = ['r', 'g', 'b']
    component_patches = [Line2D([0], [0], color=c, linewidth=5) for c in component_colours]

    bar_ax.legend(hatchpatches + component_patches,  opt_labels + ['Thickness', 'Lift', 'Drag'], loc='lower left')
    bar_ax.set_xlabel('m')
    bar_ax.set_xticks(ms)
    bar_ax.set_ylabel('dB')
    bar_ax.set_title('Harmonic Noise Components')

    fig.tight_layout()
    fig.savefig('deliverables/tms/figures/optimised_harmonics.png', dpi=300)

    sweep_ax.legend()
    sweep_ax.set_xlabel('$r_0/r_t$')
    sweep_ax.set_ylabel('Sweep [deg]')
    sweep_ax.set_title('Optimised Sweep')
    sweep_ax.grid()

    sfig.tight_layout()
    sfig.savefig('deliverables/tms/figures/optimised_harmonic_profiles.png', dpi=300)



def main():
    oper = {
        'V':  0.01,
        'c0': 343,
        'Omega': 20000 * 2 * np.pi / 60,
        'rho': 1.225,
        'pref': 2e-5,
        'nu': 1.48e-5
    }
    prop = {
        'rt': 0.05,
        'rh': 0.01,
        'B': 2,
        'nr' : 30,
        'nx' : 50,
        'c75' : 0.01
    }
    prop['r0_rt'] = np.linspace(prop['rh'], prop['rt'], prop['nr']) / prop['rt']
    prop['c'] = prop['c75'] * np.ones(prop['nr'])
    prop['xc'] = np.linspace(-0.5, 0.5, prop['nx'])
    prop['twist'] = 5 + 5 * np.arctan(1 / prop['r0_rt'])
    prop['twist'] *= np.pi / 180
    #prop['sweep'] = np.zeros(prop['nr'])
    prop['sweep'] = np.linspace(0, 0.01, prop['nr'])

    airfoil_data = np.loadtxt('app/foils/naca2412.surf')
    prop['Bd'] = prop['c'] / (2 * prop['rt'])
    n = airfoil_data.shape[0]
    xf = np.interp(np.linspace(0,1, 2*prop['nx']), np.linspace(0,1, n), airfoil_data[:, 0])
    yf = np.interp(np.linspace(0,1, 2*prop['nx']), np.linspace(0,1, n), airfoil_data[:, 1])
    tf = yf[:prop['nx']] - yf[prop['nx']:]
    _, hf = np.meshgrid(prop['c'], tf)
    prop['HX'] = hf

    _, xc = np.meshgrid(prop['r0_rt'], prop['xc'])

    prop['HX'] /= simps(prop['HX'], xc, axis=0) # normalize by area under curve

    prop['tb'] = np.max(tf) * prop['c']
    
    av = AppVars()
    av.oper = oper
    av.prop = prop

    airfoil_data = run_xfoil(airfoil_data)
    av.airfoil_data = airfoil_data

    av = betz_off_design(av)

    if not av.res['converged']:
        print("BEM failed")
        return

    res = av.res

    alpha = av.res['alpha']
    sweep = prop['sweep']
    alphas = av.airfoil_data[:, 2]
    Cl0 = av.airfoil_data[:, 3]
    Cd0 = av.airfoil_data[:, 4]
    Cl_valid = ~np.isnan(Cl0)
    Cd_valid = ~np.isnan(Cd0)

    Cl = np.interp(alpha * 180/np.pi, alphas[Cl_valid], Cl0[Cl_valid]) * np.cos(sweep) ** 2
    Cd = np.interp(alpha * 180/np.pi, alphas[Cd_valid], Cd0[Cd_valid]) * np.cos(sweep) ** 2

    prop['Cl_r'] = Cl
    prop['Cd_r'] = Cd

    prop['dCl_dxc'] = 0.5 * np.ones((prop['nx'], prop['nr']))
    prop['dCd_dxc'] = 0.01 * np.ones((prop['nx'], prop['nr']))

    prop['dCl_dxc'] /= simps(prop['dCl_dxc'], xc, axis=0) # normalize by area under curve
    prop['dCd_dxc'] /= simps(prop['dCd_dxc'], xc, axis=0) # normalize by area under curve

    #radial_locus(oper, prop)
    #radial_locus_sweep(oper, prop)


    #plot_optimised_harmonics(prop, oper)

    #operating_range(av)
    optimise_lift_harmonic_ratio(oper, prop, 2, 1)
    #
    #chord_locus(oper, prop)
    #hanson_sweep(oper, prop, res)

    plt.show()


if __name__ == "__main__":
    main()