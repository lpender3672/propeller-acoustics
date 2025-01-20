import numpy as np
from scipy.special import jv as besselj
from scipy.integrate import trapz, simps, cumtrapz
from scipy.interpolate import RectBivariateSpline, griddata

from matplotlib import pyplot as plt

from scipy.io import loadmat

from routines import (
    calc_chordwise_loading,
    _separate
)

XFOIL_INSTALLED = True
try: 
    from xfoil import XFoil
    from xfoil.model import Airfoil
except ModuleNotFoundError:
    XFOIL_INSTALLED = False
    print("Warning Xfoil not installed - hanson.py")
    

def Psi(kx, X, fX):
    f = fX * np.exp(1j * kx * X)
    ans = simps(f, x=X, axis=0)
    return ans


def uncrap(data):
    x, y = np.indices(data.shape)

    valid_mask = ~np.isnan(data)
    valid_points = np.array([x[valid_mask], y[valid_mask]]).T
    valid_values = data[valid_mask]
    filled_data = griddata(valid_points, valid_values, (x, y), method='linear')
    return filled_data

def hanson(oper: dict, prop: dict, obs: dict, ms: np.ndarray, obsmove : bool = False) -> np.ndarray:
    """
    Hansons Helicoidal Surface Theory for Harmonic Noise of Propellers in the Far Field

    oper: dict - Gas properties and operating conditions
    prop: dict - Propeller geometry
    obs: dict - Observer locations
    ms: np.ndarray - Harmonic numbers to calculate noise for
    """

    if obsmove:
        x = obs['r'] * np.sin(obs['theta'])
        y = obs['r'] * np.cos(obs['theta'])

        beta = np.arccos(oper['Mfl'])
        S0 = np.sqrt(x**2 + beta **2 * y**2)
        yr = 1 / beta**2 * (y + oper['Mfl'] * S0)
        rr = np.sqrt(yr**2 + obs['r']**2)
        thetar = np.arctan2(yr, obs['r'])

        obs_rs = rr
        obs_thetas = thetar
    else:
        obs_rs = obs['r']
        obs_thetas = obs['theta']

    dopfac = 1 - oper['Mfl'] * np.cos(obs_thetas)
    omegaDop = oper['Omega'] / dopfac

    Nobs = len(obs['r'])
    Nms = len(ms)

    PVm = np.zeros((Nobs, Nms), dtype=complex)
    PDm = np.zeros((Nobs, Nms), dtype=complex)
    PLm = np.zeros((Nobs, Nms), dtype=complex)

    _, xc = np.meshgrid(prop['r0_rt'], prop['xc'])

    if prop['HX'].shape != xc.shape:
        pass
        #prop['HX'] = prop['HX'].reshape(prop['xc'].shape[0], 1)
    
    for o in range(Nobs):

        # calc dopfac for observer
        theta = obs_thetas[o]
        r = obs_rs[o]
        y = r * np.sin(theta)
        dopfac_o = dopfac[o]
        omegaDop_o = omegaDop[o]

        for i, m in enumerate(ms):
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

            PVm[o, i] = trapz( I1, prop['r0_rt'])
            PDm[o, i] = trapz( I2, prop['r0_rt'])
            PLm[o, i] = trapz( I3, prop['r0_rt'])

    pref = oper['pref']

    Vm = 20*np.log10(2*np.abs(PVm)/pref)
    Lm = 20*np.log10(2*np.abs(PLm)/pref)
    Dm = 20*np.log10(2*np.abs(PDm)/pref)

    total = 20*np.log10(np.sqrt(np.sum((2*abs(Vm + Dm + Lm))**2, axis=1))/pref)

    # polar plot of theta vs total
    #plt.figure()
    #plt.polar(obs['theta'], Vm[:,0], label='thickness')
    #plt.polar(obs['theta'], Dm[:,0], label='drag loading')
    #plt.polar(obs['theta'], Lm[:,0], label='lift loading')
    #plt.gca().set_theta_zero_location('N')
    #plt.legend()
    #plt.show()

    out = np.array([PVm, PDm, PLm], dtype=complex)

    return out


def radial_noise_contributions(oper: dict, prop: dict, obs: dict, ms: np.ndarray, obsmove : bool = False) -> np.ndarray:
    """
    Hansons Helicoidal Surface Theory for Harmonic Noise of Propellers in the Far Field

    oper: dict - Gas properties and operating conditions
    prop: dict - Propeller geometry
    obs: dict - Observer locations
    ms: np.ndarray - Harmonic numbers to calculate noise for
    """

    _, xc = np.meshgrid(prop['r0_rt'], prop['xc'])


    theta = obs['theta']
    r = obs['r']
    y = r * np.sin(theta)
    dopfac_o = 1 - oper['Mfl'] * np.cos(theta)
    omegaDop_o = oper['Omega'] / dopfac_o

    I1plt_sum = np.zeros((prop['r0_rt'].shape), dtype=complex)
    I2plt_sum = np.zeros((prop['r0_rt'].shape), dtype=complex)
    I3plt_sum = np.zeros((prop['r0_rt'].shape), dtype=complex)

    for i, m in enumerate(ms):
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

        I1plt_sum += cumtrapz(I1, prop['r0_rt'], initial=0)
        I2plt_sum += cumtrapz(I2, prop['r0_rt'], initial=0)
        I3plt_sum += cumtrapz(I3, prop['r0_rt'], initial=0)

    out = np.array([I1plt_sum, I2plt_sum, I3plt_sum], dtype=complex)

    return out

def calc_noise_components(arr, pref):

    Vm = arr[0]
    Dm = arr[1]
    Lm = arr[2]

    V = 20*np.log10(np.sqrt(np.sum(2*Vm*np.conj(Vm), axis=1))/pref)
    L = 20*np.log10(np.sqrt(np.sum(2*Lm*np.conj(Lm), axis=1))/pref)
    D = 20*np.log10(np.sqrt(np.sum(2*Dm*np.conj(Dm), axis=1))/pref)

    csum = Vm + Dm + Lm
    total = 20*np.log10(np.sqrt(np.sum(2*csum*np.conj(csum), axis=1))/pref)

    return V, L, D, total


def hanson_secondary_variables(av, compact_chord = False):

    oper = av.oper.copy()
    prop = av.prop.copy()
    res = av.res.copy()

    ## Observer
    obs = {}
    Nobs = 100
    obs['r'] = oper['r_obs'] * prop['rt'] * np.ones((Nobs))
    obs['theta'] = np.linspace(0, np.pi, Nobs)

    ## Spanwise variables
    prop['Bd'] = prop['c'] / (2 * prop['rt'])
    n = av.airfoil_data.shape[0]
    yf = np.interp(np.linspace(0,1, 2 * prop['nx']), np.linspace(0,1, n), av.airfoil_data[:, 1])
    tf = yf[:prop['nx']] - yf[prop['nx']:]
    prop['tb'] = np.max(tf) * prop['c']
    prop['dz'] = np.diff(prop['r0'])

    oper['Min'] = 0;                  #% inflow Mach number [-]
    oper['Mfl'] = oper['V']/oper['c0'];                     #% flight Mach number [-] 
    oper['Mht'] = np.sqrt((oper['Omega']*2*np.pi*prop['rt'])**2 + oper['V']**2)/oper['c0'];                 #% Helical tip Mach number [-]

    oper['Mx'] = oper['Mfl'] + oper['Min']                       # Effective Mach number [-]
    oper['Mt'] = np.sqrt(oper['Mht']**2 - oper['Mx']**2)     # Tip mach number [-]
    oper['Mr'] = np.sqrt(oper['Mx']**2 + (oper['Mt']*prop['r0_rt'])**2) #  blade section Mach number [-]
    oper['beta'] = np.sqrt(1-oper['Mfl']**2)

    prop['Cl_r'] = res['Cl']
    prop['Cd_r'] = res['Cd']

    ## Sweep
    # This requires a component of FA 'face alignment' to prevent axial skewing
    dx = prop['r0'] * np.sin( prop['sweep'] )
    phi = prop['twist'] - res['alpha']
    prop['FA'] = dx * np.sin(phi)
    prop['MCA'] = dx * np.cos(phi)

    if compact_chord:
        prop['HX'] = np.ones((prop['nx'], prop['nr']))
        prop['dCl_dxc'] = np.ones((prop['nx'], prop['nr']))
        prop['dCd_dxc'] = np.ones((prop['nx'], prop['nr']))
    else:

        _, prop['HX'] = np.meshgrid(prop['c'], tf)
        _, xc = np.meshgrid(prop['r0_rt'], prop['xc'])

        cl_x_alpha, cd_x_alpha = calc_chordwise_loading(av.airfoil_data)
        xcdata, _ = _separate(av.airfoil_data[:, 0])
        xcdata = xcdata[:-1] - 0.5 # center the data
        interp_clx = RectBivariateSpline(xcdata, av.airfoil_data[:, 2], cl_x_alpha)
        interp_cdx = RectBivariateSpline(xcdata, av.airfoil_data[:, 2], cd_x_alpha)
        # some reason RectBivariateSpline requires eval data to be in increasing order
        argsort_resalp = np.argsort(res['alpha'])
        prop['dCl_dxc'] = np.zeros((prop['nx'], prop['nr']))
        prop['dCl_dxc'][:, argsort_resalp] = interp_clx(
            prop['xc'], res['alpha'][argsort_resalp] * 180/np.pi
            )
        prop['dCd_dxc'] = np.zeros((prop['nx'], prop['nr']))
        prop['dCd_dxc'][:, argsort_resalp] = interp_cdx(
            prop['xc'], res['alpha'][argsort_resalp] * 180/np.pi
            )

    # ensure that the integrals of the chordwise loading are equal to 1
    prop['HX'] /= simps(prop['HX'], xc, axis=0)
    prop['dCl_dxc'] /= simps(prop['dCl_dxc'], xc, axis=0)
    prop['dCd_dxc'] /= simps(prop['dCd_dxc'], xc, axis=0)

    return oper, prop, obs


def hanson_av(av):

    oper, prop, obs = hanson_secondary_variables(av)

    ms = np.arange(1, 5)
    out = hanson(oper, prop, obs, ms, False)

    V, L, D, total = calc_noise_components(out, av.oper['pref'])
    peak_observer = {
        'r': av.oper['r_obs'] * av.prop['rt'],
        'theta': av.oper['theta_obs']
    }
    vector_contributions = radial_noise_contributions(oper, prop, peak_observer, ms, False)

    return obs['theta'], V, L, D, peak_observer['theta'], vector_contributions

def validate():

    matf = loadmat('app/validation/exact.mat')['prop'][0][0]
    # Jatinder Goyal
    # https://gitlab.tudelft.nl/jatindergoyal/hanson-model-helicoidal-theory

    oper = {}
    prop = {}

    prop['B'] = 3
    prop['rt'] = 0.2032
    prop['rh'] = 0.21*0.2032
    prop['nr'] = 100 # radial sections
    prop['nx'] = 50 # chordwise elements per section

    prop['dCl_dxc'] = matf[23]
    prop['dCd_dxc'] = matf[24]
    prop['Cl_r'] =  matf[17]
    prop['Cd_r'] =  matf[18]
    prop['r0_rt'] = matf[10]
    prop['HX'] =  matf[22]
    prop['r0'] = prop['r0_rt'] * prop['rt']
    prop['xc'] = matf[11]
    prop['Bd'] = matf[13]
    prop['c'] = np.max(prop['Bd']) * 2 * prop['rt']
    prop['tb'] = matf[14]

    oper['rho'] = 1.225
    oper['c0'] = 343
    oper['gamma'] = 1.4

    V = 0
    Omega = 100

    oper['Min'] = 0;                  #% inflow Mach number [-]
    oper['Mfl'] = V/oper['c0'];                     #% flight Mach number [-] 
    oper['Mht'] = np.sqrt((Omega*2*np.pi*prop['rt'])**2 + V**2)/oper['c0'];                 #% Helical tip Mach number [-]

    oper['Mx'] = oper['Mfl'] + oper['Min']                       # Effective Mach number [-]
    oper['Mt'] = np.sqrt(oper['Mht']**2 - oper['Mx']**2)     # Tip mach number [-]
    oper['Omega'] = oper['Mt']*oper['c0'] / prop['rt']                # Rotation speed [rad/s]
    oper['Mr'] = np.sqrt(oper['Mx']**2 + (oper['Mt']*prop['r0_rt'])**2) #  blade section Mach number [-]
    oper['beta'] = np.sqrt(1-oper['Mfl']**2)

    prop['sweep'] = np.linspace(0, np.pi/2, prop['nr'])
    dx = prop['r0'] * np.sin( prop['sweep'] )
    phi = np.arcsin(oper['Mx'] / oper['Mr'])
    prop['FA'] = dx * np.sin(phi)
    prop['MCA'] = dx * np.cos(phi)

    prop['twist'] = np.linspace(0, 0.05, prop['nr'])

    obs = {}
    nobs = 180
    obs['r'] = 10 * np.ones(nobs) * prop['rt']
    obs['theta'] = np.pi / 180 * np.arange(1, 179, (179 - 1) / nobs)
    #obs['theta'] = np.array([0.017453292519943])

    oper['pref'] = 2e-5

    ms = np.arange(1, 5)

    out = hanson(oper, prop, obs, ms)

    V, D, L, _ = calc_noise_components(out, oper['pref'])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.plot(obs['theta'], V, label='thickness')
    ax.plot(obs['theta'], D, label='drag loading')
    ax.plot(obs['theta'], L, label='lift loading')
    ax.set_ylim(0, 90)
    # set theta range
    ax.set_theta_zero_location("N")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    validate()
