import numpy as np
from scipy.special import jv as besselj
from scipy.integrate import trapz, simps
from scipy.interpolate import RectBivariateSpline, griddata

from matplotlib import pyplot as plt

def Psi(kx, X, fX):
    mkx, mX = np.meshgrid(kx, X)
    f = fX * np.exp(1j * mkx * mX)
    ans = simps(f, x=X, axis=0)
    return ans


def uncrap(data):
    x, y = np.indices(data.shape)

    valid_mask = ~np.isnan(data)
    valid_points = np.array([x[valid_mask], y[valid_mask]]).T
    valid_values = data[valid_mask]
    filled_data = griddata(valid_points, valid_values, (x, y), method='linear')
    return filled_data

def hanson(oper: dict, prop: dict, obs: dict, ms: np.ndarray) -> dict:
    """
    Hansons Helicoidal Surface Theory for Harmonic Noise of Propellers in the Far Field

    oper: dict - Gas properties and operating conditions
    prop: dict - Propeller geometry
    obs: dict - Observer locations
    ms: np.ndarray - Harmonic numbers to calculate noise for
    """

    dopfac = 1 - oper['Mfl'] * np.cos(obs['theta'])
    omegaDop = oper['Omega'] / dopfac

    Nobs = len(obs['r'])
    Nms = len(ms)

    PVm = np.zeros((Nobs, Nms))
    PDm = np.zeros((Nobs, Nms))
    PLm = np.zeros((Nobs, Nms))
    
    for o in range(Nobs):

        # calc dopfac for observer
        theta = obs['theta'][o]
        r = obs['r'][o]
        y = r * np.sin(theta)
        dopfac_o = dopfac[o]
        omegaDop_o = omegaDop[o]

        for i, m in enumerate(ms):
            fac = 2 * m * prop['B'] / (oper['Mr'] * dopfac_o)
            yofac = oper['Mr'] ** 2 * np.cos(theta)

            kx = fac * prop['Bd'] * oper['Mt']
            ky = fac * yofac * prop['Bd'] / prop['r0_rt']
            phi0 = fac * yofac * prop['FA'] / (prop['r0_rt'] * 2 * prop['rt'])
            phis = fac * oper['Mt'] * prop['MCA'] / (2 * prop['rt'])

            # large term top of p5
            term1 = - (oper['rho'] * oper['c0']**2 * prop['B'] * np.sin(theta) * np.exp(1j * m * prop['B'] * (omegaDop_o * r / oper['c0'] - np.pi / 2))) / (8 * np.pi * y / (2 * prop['rt']) * dopfac_o)

            bess = besselj(m * prop['B'], m * prop['B'] * prop['r0_rt'] * oper['Mt'] * np.sin(theta) / dopfac_o)
            

            term2 = oper['Mr']**2 * np.exp(1j * (phi0 + phis)) * bess

            terms1and2 = term1 * term2

            psiVKx = Psi(kx, prop['xc'], prop['HX'].reshape(-1,1))
            psiLKx = Psi(kx, prop['xc'], prop['dCl_dxc'])
            psiDKx = Psi(kx, prop['xc'], prop['dCl_dxc'])

            I1 = terms1and2 * kx**2 * prop['tb'] * psiVKx
            I2 = terms1and2 * 1j * kx * prop['Cd_r'] / 2 * psiDKx
            I3 = terms1and2 * -1j * ky * prop['Cl_r'] / 2 * psiLKx

            PVm[o, i] = trapz( prop['r0_rt'], I1)
            PDm[o, i] = trapz( prop['r0_rt'], I2)
            PLm[o, i] = trapz( prop['r0_rt'], I3)

    pref = oper['pref']

    Vm = 20*np.log10(2*np.abs(PVm)/pref)
    Lm = 20*np.log10(2*np.abs(PLm)/pref)
    Dm = 20*np.log10(2*np.abs(PDm)/pref)

    total = 20*np.log10(np.sqrt(np.sum((2*abs(Vm + Dm + Lm))**2, axis=1))/pref)

    # polar plot of theta vs total
    plt.figure()
    plt.polar(obs['theta'], Vm[:,0], label='thickness')
    plt.polar(obs['theta'], Dm[:,0], label='drag loading')
    plt.polar(obs['theta'], Lm[:,0], label='lift loading')
    plt.gca().set_theta_zero_location('N')
    plt.legend()
    plt.show()

    out = np.hstack((Vm, Dm, Lm)).T

    return out

def hanson_av(avs, obs):

    oper = avs.oper.copy()
    prop = avs.prop.copy()

    Nobs = len(obs['r'])

    prop['Bd'] = prop['c'] / (2 * prop['rt'])
    n = avs.airfoil_data.shape[0]
    xf = np.interp(np.linspace(0,1, 2*prop['nx']), np.linspace(0,1, n), avs.airfoil_data[:, 0])
    yf = np.interp(np.linspace(0,1, 2 * prop['nx']), np.linspace(0,1, n), avs.airfoil_data[:, 1])
    prop['HX'] =  yf[:prop['nx']] - yf[prop['nx']:]
    prop['xc']  = (xf[:prop['nx']] + xf[prop['nx']:][::-1]) / 2 - 0.5

    prop['dz'] = np.diff(prop['r0'])

    prop['FA'] = 0
    prop['MCA'] = 0

    oper['Min'] = 0;                  #% inflow Mach number [-]
    oper['Mfl'] = oper['V']/oper['c0'];                     #% flight Mach number [-] 
    oper['Mht'] = np.sqrt((prop['Omega']*2*np.pi*prop['rt'])**2 + oper['V']**2)/oper['c0'];                 #% Helical tip Mach number [-]

    oper['Mx'] = oper['Mfl'] + oper['Min']                       # Effective Mach number [-]
    oper['Mt'] = np.sqrt(oper['Mht']**2 - oper['Mx']**2)     # Tip mach number [-]
    oper['Mr'] = np.sqrt(oper['Mx']**2 + (oper['Mt']*prop['r0_rt'])**2) #  blade section Mach number [-]
    oper['beta'] = np.sqrt(1-oper['Mfl']**2)

    # need
    # prop['Cl_r']
    # prop['Cd_r']
    # prop['dCl_dxc']
    # prop['dCd_dxc']


def main():
    oper = {}
    prop = {}

    oper['rho'] = 1.225
    oper['c0'] = 343
    oper['gamma'] = 1.4

    prop['B'] = 3
    prop['rt'] = 0.2032
    prop['rh'] = 0.21*0.2032
    prop['nr'] = 100 # radial sections
    prop['nx'] = 50 # chordwise elements per section

    chord_data = np.array([
        0.080153,
        0.078986,
        0.07782,
        0.07665,
        0.075486,
        0.074388,
        0.073457,
        0.072795,
        0.072553,
        0.072788,
        0.07342,
        0.074252,
        0.075086,
        0.075726,
        0.076125,
        0.076408,
        0.076688,
        0.076673,
        0.075735,
        0.073268,
        0.068986,
        0.062789,
        0.054702,
        0.045605,
        0.034028,
    ])
    prop['c'] = np.interp(np.linspace(0,1,prop['nr']), np.linspace(0,1,len(chord_data)), chord_data)

    prop['Bd'] = prop['c'] / (2 * prop['rt'])

    airfoil_data = np.loadtxt('app/foils/naca0012.surf')

    n = airfoil_data.shape[0]
    xf = np.interp(np.linspace(0,1, 2*prop['nx']), np.linspace(0,1, n), airfoil_data[:, 0])
    yf = np.interp(np.linspace(0,1, 2 * prop['nx']), np.linspace(0,1, n), airfoil_data[:, 1])
    prop['HX'] =  yf[:prop['nx']] - yf[prop['nx']:]

    rarr_pts = np.linspace(prop['rh'], prop['rt'], prop['nr']+1)
    prop['xc']  = (xf[:prop['nx']] + xf[prop['nx']:][::-1]) / 2 - 0.5

    #plt.plot(prop['xc'], prop['HX'])
    #plt.show()

    prop['r0'] = (rarr_pts[1:] + rarr_pts[:-1]) / 2
    prop['r0_rt'] = prop['r0'] / prop['rt']
    prop['dz'] = np.diff(rarr_pts)
    
    oper['Min'] = 0;                  #% inflow Mach number [-]
    oper['Mfl'] = 54.9567/oper['c0'];                     #% flight Mach number [-] 
    oper['Mht'] = np.sqrt((84.5792*2*np.pi*prop['rt'])**2 + 54.9567**2)/oper['c0'];                 #% Helical tip Mach number [-]

    oper['Mx'] = oper['Mfl'] + oper['Min']                       # Effective Mach number [-]
    oper['Mt'] = np.sqrt(oper['Mht']**2 - oper['Mx']**2)     # Tip mach number [-]
    oper['omega'] = oper['Mt']*oper['c0'] / prop['rt']                # Rotation speed [rad/s]
    oper['Mr'] = np.sqrt(oper['Mx']**2 + (oper['Mt']*prop['r0_rt'])**2) #  blade section Mach number [-]
    oper['beta'] = np.sqrt(1-oper['Mfl']**2)

    prop['FA'] = 0
    prop['MCA'] = 0

    lift_data = np.loadtxt('app/validation/BEM_Cl_rR_J1.60_idx71.csv', delimiter=',')
    drag_data = np.loadtxt('app/validation/BEM_Cd_rR_J1.60_idx71.csv', delimiter=',')
    # clean NaNs
    lift_data = lift_data[~np.isnan(lift_data).any(axis=1)]
    drag_data = drag_data[~np.isnan(drag_data).any(axis=1)]

    prop['Cl_r'] = np.interp(prop['r0_rt'], lift_data[:, 0], lift_data[:, 1])
    prop['Cd_r'] = np.interp(prop['r0_rt'], drag_data[:, 0], drag_data[:, 1])

    lift_data2d = np.loadtxt('app/validation/BEM_dCl_dxc_J1.60_idx71.csv', delimiter=',')
    drag_data2d = np.loadtxt('app/validation/BEM_dCd_dxc_J1.60_idx71.csv', delimiter=',')

    # replace NaNs with zeros
    lift_data2d = uncrap(lift_data2d)
    drag_data2d = uncrap(drag_data2d)

    # prop.fDX            = interp2(prop.chord.dCd.rR, prop.chord.dCd.xc,prop.chord.dCd.data,prop.r_mu,prop.xc,'spline');
    dCl_dxc = RectBivariateSpline(lift_data2d[1:,0], lift_data2d[0,1:], lift_data2d[1:,1:])
    dCd_dxc = RectBivariateSpline(drag_data2d[1:,0], drag_data2d[0,1:], drag_data2d[1:,1:])
    r0_rt, xc = np.meshgrid(prop['r0_rt'], prop['xc'])
    prop['dCl_dxc'] = dCl_dxc.ev(r0_rt, xc)
    prop['dCd_dxc'] = dCd_dxc.ev(r0_rt, xc)

    print(prop['dCl_dxc'].shape)

    obs = {}
    nobs = 360
    obs['r'] = 10 * np.ones(nobs) * prop['rt']
    obs['theta'] = np.pi / 180 * np.linspace(1, 179, nobs)
    #obs['theta'] = np.array([0.017453292519943])

    prop['tb'] = np.max(prop['HX']) * prop['c']

    ms = np.arange(1, 5)
    oper['pref'] = 2e-3

    out = hanson(oper, prop, obs, ms)

    tsteps = 300
    T = 2 * np.pi / (oper['omega'] * prop['B'])
    t = np.linspace(-T, T, tsteps)

    Ptotal = np.sum(out, axis=1)
    p = np.zeros((nobs, tsteps))







if __name__ == "__main__":
    main()
