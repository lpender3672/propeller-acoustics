
import numpy as np

from matplotlib import pyplot as plt
from routines import (
    XFOIL_INSTALLED,
    foil_data,
    run_xfoil,
    AppVars,
    load_oper_from_file,
    load_prop_from_file
)
from scipy.optimize import root, brentq

def betz_design(av):

    #print(av.prop)
    #print(av.oper)

    R = av.prop['rt']
    B = av.prop['B']
    Omega = av.oper['Omega']
    nu = av.oper['nu']
    ro = av.oper['rho']

    Nsect = av.prop['nr']

    # Select an initial estimate for zeta
    dzeta = 100
    xi = av.prop['r0_rt']

    alpha = 5 * np.pi / 180
    Cl = 0.5
    Cd = 0.1

    target_thrust_N = 1 # N
    target_power_W = 50 # W

    V = av.oper['V'] # m/s
    if V == 0:
        print("V is zero, unanble to find betz optimal solution")
        return False
    T_c = 2 * target_thrust_N / (ro * V**2 * np.pi * R**2)
    P_c = 2 * target_power_W / (ro * V**3 * np.pi * R**2)
    y = xi * R * Omega / V
    lamda = V / (Omega * R) # advance ratio

    zeta = np.sqrt(2 * T_c)

    print(f'Advance ratio: {lamda}')

    while np.abs(dzeta/zeta) > 1e-3:

        # Design of Optimum Propellers
        # Charles N. Adkins*
        # Falls Church, Virginia 22042
        # Douglas Aircraft Company, Long Beach, California 90846

        # 2 calculate F and phi
        phi_t = np.arctan(lamda * (1 + zeta / 2))
        # ensure phi_t is not too small
        phi_t = np.clip(phi_t, 1e-3, np.pi/2)

        f = B/2 * (1 - xi) / np.sin(phi_t)
        F = 2 / np.pi * np.arccos(np.exp(-f))
        phi = np.arctan2( np.tan(phi_t), xi)

        # 3 calculate Wc and Re_c
        G = F * np.cos(phi) * np.sin(phi)
        Wc = 4 * np.pi * lamda * G * V * R * zeta / (Cl * B)
        Re_c = Wc / nu

        #print(np.max(Re_c) - np.min(Re_c))

        # 4,5 determine epsilon and Cl
        # TODO: find best airfoil at each chord Reynolds number
        alpha = alpha
        Cl = Cl
        Cd = Cd
        epsilon = Cd / Cl

        # 6 calculate a, a', W
        a = zeta / 2 * np.cos(phi)**2 * (1 - epsilon * np.tan(phi))
        a_prime = zeta / (2 * y) * np.cos(phi) * np.sin(phi) * (1 + epsilon / np.tan(phi))
        W = V * (1 + a) / np.sin(phi)
        # 7 recompute step 3 for chord and blade twist
        Wc = 4 * np.pi * lamda * G * V * R * zeta / (Cl * B)
        c = Wc / W
        beta = alpha + phi

        #print(np.max(phi))

        # 8 calculate derivatives and integrate wrt xi
        I1_prime = 4 * xi * G * (1 - epsilon * np.tan(phi)) 
        I2_prime = lamda * I1_prime / (2 * xi) * (1 + epsilon / np.tan(phi)) * np.cos(phi) * np.sin(phi)
        J1_prime = 4 * xi * G * (1 + epsilon / np.tan(phi))
        J2_prime = J1_prime / 2 * (1 - epsilon * np.tan(phi)) * np.cos(phi) ** 2

        I1 = np.trapz(I1_prime, xi)
        I2 = np.trapz(I2_prime, xi)
        J1 = np.trapz(J1_prime, xi)
        J2 = np.trapz(J2_prime, xi)

        # 9 determine new zeta
        I1_over_2I2 = I1 / (2 * I2)
        J1_over_2J2 = J1 / (2 * J2)

        # Thrust specified
        #new_zeta = I1_over_2I2 - (I1_over_2I2**2 - T_c / I2)**(1/2)
        #P_c = J1 * zeta + J2 * zeta**2

        # Power specified
        new_zeta = - J1_over_2J2 + (J1_over_2J2**2 + P_c / J2)**(1/2)
        T_c = I1 * zeta + I2 * zeta**2

        if np.isnan(new_zeta):
            print("unable to reach target input power/thrust")
            return False

        dzeta = new_zeta - zeta
        zeta = new_zeta

    else:
        print("Betz calculation converged")
        # success
        av.prop['c'] = c
        av.prop['twist'] = beta

        av.dist['CTL_c_type'] = 'custom'
        av.dist['CTL_twist_type'] = 'custom'
        
        return True
    
def betz_off_design(av):

    R = av.prop['rt']
    B = av.prop['B']
    Omega = av.oper['Omega']
    nu = av.oper['nu']
    ro = av.oper['rho']
    xi = av.prop['r0_rt']
    c = av.prop['c']
    sweep = av.prop['sweep']
    beta = av.prop['twist']

    V = av.oper['V'] # m/s

    y = xi * R * Omega / V
    lamda = V / (Omega * R * np.cos(sweep)) # advance ratio

    # An initial estimate for phi can be obtained from Eq. (8) by setting zeta = 0
    phi = np.arctan((1 + 0) * lamda / xi)
    dphi = np.inf

    sigma = B * c / (2 * np.pi * xi * R)

    loss_model = 'Prantl'

    if XFOIL_INSTALLED:
        alphas = av.airfoil_data[:, 2]
        Cl0 = av.airfoil_data[:, 3]
        Cd0 = av.airfoil_data[:, 4]
        Cl_valid = ~np.isnan(Cl0)
        Cd_valid = ~np.isnan(Cd0)

    iters = 0

    while (iters < 100):
        if (np.max(np.abs(dphi / phi)) < 1e-3):
            break

        # Analysis of Arbitrary Designs
        # Charles N. Adkins*

        alpha = beta - phi
        # airfoil coefficients are known from the section data and alpha

        #Cl = Cl0 + alpha * 2 * np.pi
        #Cd = Cd0
        if XFOIL_INSTALLED:
            # mark indicies where Cl and Cd are outside airfoil data
            invalids = np.where((alpha * 180/np.pi < alphas.min()) | (alpha * 180/np.pi > alphas.max()))[0]
            Cl = np.interp(alpha * 180/np.pi, alphas[Cl_valid], Cl0[Cl_valid]) * np.cos(sweep) ** 2
            Cd = np.interp(alpha * 180/np.pi, alphas[Cd_valid], Cd0[Cd_valid]) * np.cos(sweep) ** 2
        else:
            Cl = 2 * np.pi * alpha
            Cd = 0.0087 - 0.021 * alpha + 0.400 * alpha ** 2

        # better to wait and see if its out of bounds
        # phi = np.clip(phi, beta - alphas.min() * np.pi / 180, beta - alphas.max() * np.pi / 180)

        Cx = Cl * np.cos(phi) - Cd * np.sin(phi)
        Cz = Cl * np.sin(phi) + Cd * np.cos(phi)
        K = Cz / (4 * np.sin(phi)**2)
        K_prime = Cx / (4 * np.sin(phi) * np.cos(phi))
        phi_t = np.arctan( xi * np.tan(phi))
        if loss_model == 'Prantl':
            F = 2 / np.pi * np.arccos(np.exp( - B/2 * (1/xi - 1) / np.sin(phi_t))) # Prandtl tip loss factor
        elif loss_model == 'Viterna':
            pass
        else:
            F = 1 # no tip loss factor

        a = sigma * K * ( F - sigma * K)
        a_prime = sigma * K_prime * ( F + sigma * K_prime)
        # Viterna and Janetzke clip a and a_prime to 0.7
        a = np.clip(a, -0.7, 0.7)
        a_prime = np.clip(a_prime, -0.7, 0.7)

        Ucorr = Omega * xi * R * np.cos(sweep)
        # the Reynolds number is determined from the known chord and W
        W = np.sqrt((V * (1 + a)) ** 2 + (Ucorr * (1 - a_prime)) ** 2)
        Re_c = W * c / nu
        #print(Re_c)
        new_phi = np.arctan(V * (1 + a) / (Ucorr * (1 - a_prime)))
        dphi = new_phi - phi
        phi = new_phi
        iters += 1

    else:
        # never broke out of loop
        av.res['converged'] = False
        return av
    
    av.res['converged'] = True
    av.res['alpha'] = alpha
    av.res['Cl'] = Cl
    av.res['Cd'] = Cd
    av.res['invalids'] = invalids
    # set interesting values
    
    # not sure if this is right because of a typo in the paper, specifically exponent of 3/2
    #CT_prime = (np.pi ** 3 / 4) * sigma * Cz * xi * F**(3/2) / ((F + sigma * K_prime) * np.cos(phi))**2
    #CP_prime = CT_prime * np.pi * xi * Cx / Cz * np.cos(sweep)
    Wsq = (V * (1 + a)) ** 2 + (Ucorr * (1 - a_prime)) ** 2
    T_prime = 1 / 2 * B * Wsq * c * (F * Cl * np.cos(phi) - Cd * np.sin(phi))
    Q_prime = 1 / 2 * B * Wsq * c * (F * Cl * np.sin(phi) + Cd * np.cos(phi)) * xi * R * np.cos(sweep)
    P_prime = Omega * Q_prime

    A = np.pi * R**2
    CT_prime = T_prime / (1/2 * A * (R * Omega) ** 2)
    CP_prime = P_prime / (1/2 * A * (R * Omega) ** 3)
    
    #print(Cl)

    CT = np.trapz(CT_prime, xi * R)
    CP = np.trapz(CP_prime, xi * R)

    FM = np.sign(CT) * np.abs(CT) ** (2/3) / (np.sqrt(2) * np.abs(CP))

    av.res['CT'] = CT
    av.res['CP'] = CP
    av.res['FM'] = FM
    av.res['dCP'] = CP_prime
    av.res['dCT'] = CT_prime

    #print(f"CP: {CP}, CT: {CT}, FM: {FM}")

    return av

def guaranteed_convergence_BEM(av):

    # A simple solution method for the blade element momentum equations with guaranteed convergence
    # S. Andrew Ning

    epsilon = 1e-6
    max_iters = 100

    if XFOIL_INSTALLED:
        falphas = av.airfoil_data[:, 2]
        Cl0 = av.airfoil_data[:, 3]
        Cd0 = av.airfoil_data[:, 4]
        Cl_valid = ~np.isnan(Cl0)
        Cd_valid = ~np.isnan(Cd0)

    av.res['converged'] = False

    twist = av.prop['twist']
    sweep = av.prop['sweep']
    B = av.prop['B']
    R = av.prop['rt']
    Omega = av.oper['Omega']
    V = -av.oper['V']
    lamda_r = V / (Omega * av.prop['r0_rt'] * R)
    sigmap = B * av.prop['c'] / (2 * np.pi * av.prop['r0_rt'] * R)

    def prantl_tiploss(phi, i):
        return 2 / np.pi * np.arccos(np.exp(-av.prop['B'] / 2 * (1 - av.prop['r0_rt'][i]) / np.sin(phi)))
    
    def cxcz(phi, i):
        alpha = twist[i] - phi
        Cl = np.interp(alpha * 180/np.pi, falphas[Cl_valid], Cl0[Cl_valid]) * np.cos(sweep[i]) ** 2
        Cd = np.interp(alpha * 180/np.pi, falphas[Cd_valid], Cd0[Cd_valid]) * np.cos(sweep[i]) ** 2
        Cx = Cl * np.cos(phi) - Cd * np.sin(phi)
        Cz = Cl * np.sin(phi) + Cd * np.cos(phi)
        return Cx, Cz
    
    def kappa(phi, i):
        F = prantl_tiploss(phi, i)
        _, cz = cxcz(phi, i)
        return sigmap[i] * cz / ( 4 * F * np.sin(phi) ** 2)
    def kappa_prime(phi, i):
        F = prantl_tiploss(phi, i)
        cx, _ = cxcz(phi, i)
        return sigmap[i] * cx / (4 * F * np.sin(phi) * np.cos(phi))


    def a(phi, i):
        k = kappa(phi, i)
        if (k < 2/3):
            return k / (1 + k)
        
        F = prantl_tiploss(phi, i)
        gamma1 = 2 * F * k -(10/9 - F)
        gamma2 = 2 * F * k - F * (4/3 - F)
        gamma3 = 2 * F * k - (25/9 - 2 * F)
        return (( gamma1 - np.sqrt(gamma2)) / gamma3)
    
    def a_prime(phi, i):
        kp = kappa_prime(phi, i)
        ap = kp / (1 - kp)
        #ap = np.clip(ap, -0.7, 0.7)
        return ap

    A = np.pi * av.prop['rt']**2

    CT_prime = np.zeros(av.prop['nr'])
    CP_prime = np.zeros(av.prop['nr'])
    FM_prime = np.zeros(av.prop['nr'])
    Cls = np.zeros(av.prop['nr'])
    Cds = np.zeros(av.prop['nr'])
    alphas = np.zeros(av.prop['nr'])

    for i in range(av.prop['nr']):
        def f(phi):
            return np.sin(phi) / (1 - a(phi, i)) - np.cos(phi) * (1 - kappa_prime(phi, i)) / lamda_r[i]
    
        def fpb(phi):
            return np.sin(phi) * (1 - kappa(phi, i)) - np.cos(phi) * (1 - kappa_prime(phi, i)) / lamda_r[i]
    
        try:
            if f(np.pi/2) > 0:
                phistar = brentq(f, epsilon, np.pi / 2, maxiter=max_iters)
            elif fpb(-np.pi/4) < 0 and fpb(epsilon) > 0:
                phistar = brentq(fpb, -np.pi/4, epsilon, maxiter=max_iters)
            else:
                phistar = brentq(f, np.pi / 2, np.pi, maxiter=max_iters)
        except ValueError:
            av.res['converged'] = False
            return av

        a_val = a(phistar, i)
        ap_val = a_prime(phistar, i)
        c = av.prop['c'][i]

        #print(a_val, ap_val)

        alpha = twist[i] - phistar

        Cl = np.interp(alpha * 180/np.pi, falphas[Cl_valid], Cl0[Cl_valid]) * np.cos(sweep[i]) ** 2
        Cd = np.interp(alpha * 180/np.pi, falphas[Cd_valid], Cd0[Cd_valid]) * np.cos(sweep[i]) ** 2
        Wsq = (V * (1 + a_val)) ** 2 + (Omega * av.prop['r0_rt'][i] * R * (1 - ap_val)) ** 2
        T_prime = 1 / 2 * B * Wsq * c * (Cl * np.cos(phistar) - Cd * np.sin(phistar))
        Q_prime = 1 / 2 * B * Wsq * c * (Cl * np.sin(phistar) + Cd * np.cos(phistar)) * av.prop['r0_rt'][i] * av.prop['rt'] * np.cos(sweep[i])
        P_prime = Omega * Q_prime

        CT_prime[i] = T_prime / (1/2 * A * (R * Omega) ** 2)
        CP_prime[i] = P_prime / (1/2 * A * (R * Omega) ** 3)
        Cls[i] = Cl
        Cds[i] = Cd
        alphas[i] = alpha

        FM_prime[i] = 0.5 * np.sign(CT_prime[i]) * np.abs(CT_prime[i]) ** (2/3) / (np.sqrt(2) * np.abs(CP_prime[i]))

    
    CT = np.trapz(CT_prime, av.prop['r0_rt'] * av.prop['rt'])
    CP = np.trapz(CP_prime, av.prop['r0_rt'] * av.prop['rt'])

    FM = np.sign(CT) * np.abs(CT) ** (2/3) / (np.sqrt(2) * np.abs(CP))

    av.res['converged'] = True

    av.res['CT'] = CT
    av.res['CP'] = CP
    av.res['FM'] = FM

    av.res['dCP'] = CP_prime
    av.res['dCT'] = CT_prime
    av.res['dFM'] = FM_prime
    av.res['Cl'] = Cls
    av.res['Cd'] = Cds
    av.res['alpha'] = alphas

    av.res['invalids'] = np.zeros(av.prop['nr'])

    #print(f"CT: {CT}, CP: {CP}")

    return av



def operating_range(av, Js):

    CPs = np.zeros(Js.shape)
    CTs = np.zeros(Js.shape)
    FMs = np.zeros(Js.shape)

    avcopy = av.copy()

    for i,J in enumerate(Js):
        avcopy.oper['V'] = J * avcopy.oper['Omega'] * avcopy.prop['rt']
        avcopy = guaranteed_convergence_BEM(avcopy)

        if not avcopy.res['converged']:
            CPs[i] = np.nan
            CTs[i] = np.nan
            FMs[i] = np.nan
            continue

        ivlds = avcopy.res['invalids']
        if (len(ivlds) > 0 and ivlds[-1] > avcopy.prop['nr'] // 2):
            CPs[i] = np.nan
            CTs[i] = np.nan
            FMs[i] = np.nan
            continue

        CPs[i] = avcopy.res['CP']
        CTs[i] = avcopy.res['CT']
        FMs[i] = avcopy.res['FM']

    return Js[CTs > 0], CPs[CTs > 0], CTs[CTs > 0], FMs[CTs > 0]

def main():

    av = AppVars()

    av.oper = load_oper_from_file('app/app_vars.json')
    av.prop = load_prop_from_file('app/props/constant_chord.prop')

    av.airfoil_data = np.loadtxt(av.prop['foil_path'])
    av.airfoil_data = run_xfoil(av.airfoil_data)

    av.oper['V'] = 0.01
    av = guaranteed_convergence_BEM(av)

    fig, ax = plt.subplots()
    ax.plot(av.prop['r0_rt'], av.res['dFM'])

    Js = np.linspace(0.1, 1, 100)
    #Js, CPs, CTs, FMs = operating_range(av, Js)

    fig, ax = plt.subplots()

    #ax.plot(Js, FMs, label='FM')
    ax.legend()

    plt.show()

if __name__ == '__main__':

    main()