import numpy as np
from scipy.special import jv as besselj
from scipy.integrate import trapz, simps
from scipy.interpolate import interp2d

def Psi(kx, X, fX):
    f = fX * np.exp(1j * kx * X)
    return simps(f, X, 1)


def hanson(av: dict, prop: dict, obs: dict, ms: np.ndarray) -> dict:
    """
    Hansons Helicoidal Surface Theory for Harmonic Noise of Propellers in the Far Field

    av: dict - Gas properties and operating conditions
    prop: dict - Propeller geometry
    obs: dict - Observer locations
    ms: np.ndarray - Harmonic numbers to calculate noise for
    """

    y = obs['r'] * np.sin(obs['theta'])

    dopfac = 1 - av['Mfl'] * np.cos(obs['theta'])
    omegaDop = av['omega'] / dopfac

    Nobs = len(obs['r'])
    Nms = len(ms)
    out = {
        "PVM" : np.zeros((Nobs, Nms)),
        "PDM" : np.zeros((Nobs, Nms)),
        "PLM" : np.zeros((Nobs, Nms))
    }

    for o in range(Nobs):

        # calc dopfac for observer
        theta = obs['theta'][o]
        r = obs['r'][o]

        for m in ms:
            fac = 2 * m * prop['B'] / (av['Mr'] * dopfac)
            yofac = av['Mr'] ** 2 * np.cos(theta)

            kx = fac * prop['Bd'] * av['Mt']
            ky = fac * yofac * prop['Bd'] / prop['r0_rt']
            phi0 = fac * yofac * prop['FA'] / (prop['r0_rt'] * 2 * prop['rt'])
            phis = fac * av['Mt'] * prop['MCA'] / (2 * prop['rt'])

            # large term top of p5
            term1 = - (av['rho'] * av['c0']**2 * prop['B'] * np.sin(theta) * np.exp(1j * m * prop['B'] * (omegaDop * r / av['c0'] - np.pi / 2))) / (8 * np.pi * y / (2 * prop['rt']) * dopfac)

            bess = besselj(m * prop['B'], m * prop['B'] * prop['r0_rt'] * av['Mt'] * np.sin(theta) / dopfac)

            term2 = av['Mr']**2 * np.exp(1j * (phi0 + phis)) * bess

            terms1and2 = term1 * term2

            psiVKx = Psi(kx, prop['HX'])
            psiLKx = Psi(kx, prop['xc'], prop['Cl'])
            psiDKx = Psi(kx, prop['xc'], prop['Cd'])

            out['PVM'][o, m] = trapz( prop['r0_rt'], terms1and2 * kx**2 * prop['tb'] * psiVKx)
            out['PDM'][o, m] = trapz( prop['r0_rt'], terms1and2 * 1j * kx * prop['Cd'] / 2 * psiDKx)
            out['PLM'][o, m] = trapz( prop['r0_rt'], terms1and2 * -1j * ky * prop['Cl'] / 2 * psiLKx)

    return out


av = {}
prop = {}

av['rho'] = 1.225
av['c0'] = 343
av['gamma'] = 1.4

prop['B'] = 2
prop['rt'] = 0.05
prop['rh'] = 0.01
prop['nr'] = 100 # radial sections
prop['nx'] = 100 # chordwise elements per section


xc_pts = np.linspace(-0.5,0.5,prop['nx']+1)
rarr_pts = np.linspace(prop['rh'], prop['rt'], prop['nr']+1)
prop['xc']  = (xc_pts[1:] + xc_pts[:-1]) / 2
prop['r0_rt'] = (rarr_pts[1:] + rarr_pts[:-1]) / 2
prop['dz'] = np.diff(rarr_pts)

av['Min'] = 0;                  #% inflow Mach number [-]
av['Mfl'] = 54.9567/av['c0'];                     #% flight Mach number [-] 
av['Mht'] = np.sqrt((84.5792*2*np.pi*prop['rt'])**2 + 54.9567**2)/av['c0'];                 #% Helical tip Mach number [-]

av['Mx'] = av['Mfl'] + av['Min']                       # Effective Mach number [-]
av['Mt'] = np.sqrt(av['Mht']**2 - av['Mx']**2)     # Tip mach number [-]
av['omega'] = av['Mt']*av['c0'] / prop['rt']                # Rotation speed [rad/s]
av['Mr'] = np.sqrt(av['Mx']**2 + (av['Mt']*prop['rarr'])**2) #  blade section Mach number [-]
av['beta'] = np.sqrt(1-av['Mfl']**2)

obs = {}
nobs = 100
obs['r'] = 100 * np.ones(nobs)
obs['theta'] = np.pi / 180 * np.linspace(0, 180, nobs)

thick_data = np.loadtxt('cfd/data/t_xc.csv', delimiter=',')
chord_rR = thick_data[0,1:]
chord_xc = thick_data[1:,0]
chord_t = thick_data[1:,1:]

HXinterp = interp2d(chord_rR,chord_xc,chord_t, kind='cubic');
prop['HX'] = HXinterp(prop['rarr'],prop['xc'])

ms = np.arange(1, 10)

hanson(av, prop, obs, ms)