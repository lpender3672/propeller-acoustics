import numpy as np
from scipy.special import jv as besselj
from scipy.integrate import trapz, simps, cumtrapz

from matplotlib import pyplot as plt
from matplotlib import cm
from betz import bem
from hanson import hanson, calc_noise_components

class AppVars():
    def __init__(self):
        self.oper = {

        }

        self.prop = {

        }

        self.dist = {

        }

        self.res = {
            
        }

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

def radial_locus(oper: dict, prop: dict):

    # calc dopfac for observer
    oper['Min'] = 0;                  #% inflow Mach number [-]
    oper['Mfl'] = oper['V']/oper['c0'];                     #% flight Mach number [-] 
    oper['Mht'] = np.sqrt((oper['Omega']*2*np.pi*prop['rt'])**2 + oper['V']**2)/oper['c0'];                 #% Helical tip Mach number [-]

    oper['Mx'] = oper['Mfl'] + oper['Min']                       # Effective Mach number [-]
    oper['Mt'] = np.sqrt(oper['Mht']**2 - oper['Mx']**2)     # Tip mach number [-]
    oper['Mr'] = np.sqrt(oper['Mx']**2 + (oper['Mt']*prop['r0_rt'])**2) #  blade section Mach number [-]

    dx = prop['r0_rt'] * prop['rt'] * np.sin( prop['sweep'] )
    phi = np.arcsin(oper['Mx'] / oper['Mr'])
    prop['FA'] = dx * np.sin(phi)
    prop['MCA'] = dx * np.cos(phi)

    theta = 3 * np.pi / 4 # np.arange(0, np.pi, 0.01)

    fig, ax = plt.subplots()

    dopfac_o = 1 - oper['Mfl'] * np.cos(theta)
    r = 10 * prop['rt']
    y = r * np.sin(theta)
    omegaDop_o = oper['Omega'] / dopfac_o

    Nrs = prop['nr']
    Nms = 5
    PVm = np.zeros((Nrs), dtype=complex)
    PDm = np.zeros((Nrs), dtype=complex)
    PLm = np.zeros((Nrs), dtype=complex)

    for i, m in enumerate(np.arange(1, Nms + 1)):
        fac = 2 * m * prop['B'] / (oper['Mr'] * dopfac_o)
        yofac = oper['Mr'] ** 2 * np.cos(theta) - oper['Mx']

        kx = fac * prop['Bd'] * oper['Mt']
        ky = fac * yofac * prop['Bd'] / prop['r0_rt']
        phi0 = fac * yofac * prop['FA'] / (prop['r0_rt'] * 2 * prop['rt'])
        phis = fac * oper['Mt'] * prop['MCA'] / (2 * prop['rt'])

        # large term top of p5
        term1 = - (oper['rho'] * oper['c0']**2 * prop['B'] * np.sin(theta) * np.exp(1j * m * prop['B'] * (omegaDop_o * r / oper['c0'] - np.pi / 2))) / ((8 * np.pi * y / (2 * prop['rt']) * dopfac_o) + 1e-10)

        bess = besselj(m * prop['B'], m * prop['B'] * prop['r0_rt'] * oper['Mt'] * np.sin(theta) / dopfac_o)
        

        term2 = oper['Mr']**2 * np.exp(1j * (phi0 + phis)) * bess

        terms1and2 = term1 * term2

        psiVKx = Psi(kx, xc, prop['HX'])
        psiLKx = Psi(kx, xc, prop['dCl_dxc'])
        psiDKx = Psi(kx, xc, prop['dCd_dxc'])

        I1 = terms1and2 * kx**2 * prop['tb'] * psiVKx
        I2 = terms1and2 * 1j * kx * prop['Cd_r'] / 2 * psiDKx
        I3 = terms1and2 * -1j * ky * prop['Cl_r'] / 2 * psiLKx

        I1plt = cumtrapz(I1, prop['r0_rt'], initial=0)
        I2plt = cumtrapz(I2, prop['r0_rt'], initial=0)
        I3plt = cumtrapz(I3, prop['r0_rt'], initial=0)

        PVm += I1plt
        PDm += I2plt
        PLm += I3plt


    viridis = cm.get_cmap('viridis')
    clrs = viridis(prop['r0_rt'])

    ax.plot(PVm.real, PVm.imag, label=f'Thickness, m={m}')
    ax.plot(PLm.real, PLm.imag, label=f'Lift m={m}')
    ax.plot(PDm.real, PDm.imag, label=f'Drag m={m}')

    ax.grid()
    ax.legend()


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


if __name__ == "__main__":
    oper = {
        'V':  50.0,
        'c0': 343,
        'Omega': 10000 * 2 * np.pi / 60,
        'rho': 1.225,
        'pref': 2e-5
    }
    prop = {
        'rt': 0.01,
        'rh': 0.001,
        'B': 2,
        'nr' : 20,
        'nx' : 50
    }
    prop['r0_rt'] = np.linspace(prop['rh'], prop['rt'], prop['nr']) / prop['rt']
    prop['c'] = 0.1 * prop['rt'] * np.ones(prop['nr'])
    prop['xc'] = np.linspace(-0.5, 0.5, prop['nx'])
    prop['twist'] = 0.1 + 0.01 / prop['r0_rt']
    #prop['sweep'] = np.zeros(prop['nr'])
    prop['sweep'] = np.linspace(0, 1.2, prop['nr'])

    airfoil_data = np.loadtxt('app/foils/naca0012.surf')
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
    
    prop['sweep'] = np.linspace(0, 1.5, prop['nr']) ** 2

    av = AppVars()
    av.oper = oper
    av.prop = prop
    av.airfoil_data = airfoil_data

    bem(av)
    res = av.res
    alpha = res['alpha']
    alpha = prop['twist'] - np.arctan( oper['V'] / (oper['Omega'] * prop['r0_rt'] * prop['rt']) )

    prop['Cl_r'] = (2 * np.pi * alpha) #* np.cos(prop['sweep'])**2
    prop['Cd_r'] = (0.0087 - 0.021 * alpha + 0.400 * alpha ** 2) #* np.cos(prop['sweep'])**2

    prop['dCl_dxc'] = 0.5 * np.ones((prop['nx'], prop['nr']))
    prop['dCd_dxc'] = 0.01 * np.ones((prop['nx'], prop['nr']))

    prop['dCl_dxc'] /= simps(prop['dCl_dxc'], xc, axis=0) # normalize by area under curve
    prop['dCd_dxc'] /= simps(prop['dCd_dxc'], xc, axis=0) # normalize by area under curve

    radial_locus(oper, prop)
    #chord_locus(oper, prop)
    #hanson_sweep(oper, prop)

    plt.show()
