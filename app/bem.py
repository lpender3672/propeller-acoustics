
import numpy as np

from matplotlib import pyplot as plt
from routines import (
    XFOIL_INSTALLED,
    run_xfoil,
    load_foil,
    AppVars,
    load_oper_from_file,
    load_prop_from_file,
    interpolate_clcd,
    correct_clcd_sweep
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
        Cl, Cd = interpolate_clcd(av.airfoil_data, alpha, 5e5)

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
    invalids = np.where((alpha * 180/np.pi < - 30))[0]
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


def firstbracket(f, xmin, xmax, n, backwardsearch=False):

    xvec = np.linspace(xmin, xmax, n)
    if backwardsearch:  # Start from xmax and work backwards
        xvec = xvec[::-1]

    fprev = f(xvec[0])
    for i in range(1, n):
        fnext = f(xvec[i])
        if fprev * fnext < 0:  # Bracket found
            if backwardsearch:
                return True, xvec[i], xvec[i - 1]
            else:
                return True, xvec[i - 1], xvec[i]
        fprev = fnext

    return False, 0.0, 0.0

def determine_quadrants(epsilon_everywhere, Vx, Vy, theta):
    epsilon = 1e-6

    if epsilon_everywhere:
        q1 = [epsilon, np.pi / 2 - epsilon]
        q2 = [-np.pi / 2 + epsilon, -epsilon]
        q3 = [np.pi / 2 + epsilon, np.pi - epsilon]
        q4 = [-np.pi + epsilon, -np.pi / 2 - epsilon]
    else:
        q1 = [epsilon, np.pi / 2]
        q2 = [-np.pi / 2, -epsilon]
        q3 = [np.pi / 2, np.pi - epsilon]
        q4 = [-np.pi + epsilon, -np.pi / 2]

    if np.isclose(Vx, 0.0, atol=1e-6) and np.isclose(Vy, 0.0, atol=1e-6):
        return None  # Outputs() placeholder, needs user-defined class

    elif np.isclose(Vx, 0.0, atol=1e-6):
        startfrom90 = False  # Start bracket at 0 degrees

        if Vy > 0 and theta > 0:
            order = (q1, q2)
        elif Vy > 0 and theta < 0:
            order = (q2, q1)
        elif Vy < 0 and theta > 0:
            order = (q3, q4)
        else:  # Vy < 0 and theta < 0
            order = (q4, q3)

    elif np.isclose(Vy, 0.0, atol=1e-6):
        startfrom90 = True  # Start bracket search from 90 degrees

        if Vx > 0 and abs(theta) < np.pi / 2:
            order = (q1, q3)
        elif Vx < 0 and abs(theta) < np.pi / 2:
            order = (q2, q4)
        elif Vx > 0 and abs(theta) > np.pi / 2:
            order = (q3, q1)
        else:  # Vx < 0 and abs(theta) > np.pi / 2
            order = (q4, q2)

    else:  # Normal case
        startfrom90 = False

        if Vx > 0 and Vy > 0:
            order = (q1, q2, q3, q4)
        elif Vx < 0 and Vy > 0:
            order = (q2, q1, q4, q3)
        elif Vx > 0 and Vy < 0:
            order = (q3, q4, q1, q2)
        else:  # Vx < 0 and Vy < 0
            order = (q4, q3, q2, q1)

    return startfrom90, order

def main_function(residual, firstbracket, order, npts, forcebackwardsearch, implicitad_option):

    pass

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
    V = av.oper['V']
    lamda_r = V / (Omega * av.prop['r0_rt'] * R)
    sigmap = B * av.prop['c'] / (2 * np.pi * av.prop['r0_rt'] * R)

    def prantl_tiploss(r, Rhub, Rtip, phi, B):
        asphi = abs(np.sin(phi))
        factortip = B/2.0*(Rtip/r - 1)/asphi
        F = 2.0/np.pi*np.arccos(np.exp(-factortip))
        return F
    
    
    def residual_and_result(phi, i, result):
        # Unpack inputs
        B = av.prop['B']
        r = av.prop['r0_rt'][i] * av.prop['rt']
        chord = av.prop['c'][i]
        theta = av.prop['twist'][i]
        Rhub = av.prop['rh']
        Rtip = av.prop['rt']
        Vx = av.oper['V']
        Vy = av.oper['Omega'] * r
        rho = av.oper['rho']
        mu = rho * av.oper['nu']
        pitch = 0
        asound = av.oper['c0']

        # Constants
        sigma_p = B * chord / (2.0 * np.pi * r)
        sphi = np.sin(phi)
        cphi = np.cos(phi)

        # Angle of attack
        alpha = (theta + pitch) - phi

        # Reynolds/Mach number
        W0 = np.sqrt(Vx**2 + Vy**2)  # Ignoring induction
        Re = rho * W0 * chord / mu
        Mach = W0 / asound

        # Airfoil cl/cd

        cl, cd = interpolate_clcd(av.airfoil_data, alpha, Re)
        cl, cd = correct_clcd_sweep(cl, cd, sweep[i])

        # Resolve into normal and tangential forces
        cn = cl * cphi - cd * sphi
        ct = cl * sphi + cd * cphi

        # Hub/tip loss

        F = prantl_tiploss(r, Rhub, Rtip, phi, B)

        # Section parameters
        k = cn * sigma_p / (4.0 * F * sphi**2)
        kp = ct * sigma_p / (4.0 * F * sphi * cphi)

        # Solve for induced velocities
        if np.isclose(Vx, 0.0, atol=1e-6):
            u = np.sign(phi) * kp * cn / ct * Vy
            v = 0.0
            a = 0.0
            ap = 0.0
            R = np.sign(phi) - k

        elif np.isclose(Vy, 0.0, atol=1e-6):
            u = 0.0
            v = k * ct / cn * abs(Vx)
            a = 0.0
            ap = 0.0
            R = np.sign(Vx) + kp
        else:
            if phi < 0:
                k *= -1
            if np.isclose(k, 1.0, atol=1e-6):
                return 1.0, result
            if k >= -2.0 / 3:  # Momentum region
                a = k / (1 - k)
            else:  # Empirical region
                g1 = 2 * k + 1.0 / 9
                g2 = -2 * k - 1.0 / 3
                g3 = -2 * k - 7.0 / 9
                a = (g1 + np.sqrt(g2)) / g3
            u = a * Vx
            # Tangential induction
            if Vx < 0:
                kp *= -1
            if np.isclose(kp, -1.0, atol=1e-6):
                return 1.0
            ap = kp / (1 + kp)
            v = ap * Vy
            # Residual function
            R = np.sin(phi) / (1 + a) - Vx / Vy * np.cos(phi) / (1 - ap)
        # Loads
        # Hub/tip loss correction
        if np.isclose(Vx, 0.0, atol=1e-6):
            G = np.sqrt(F)
        elif np.isclose(Vy, 0.0, atol=1e-6):
            G = F
        else:
            G = (-1.0 + np.sqrt(1.0 + 4 * a * (1.0 + a) * F)) / (2 * a)
        u *= G
        v *= G

        W = np.sqrt((Vx + u)**2 + (Vy - v)**2)
        Np = cn * 0.5 * rho * W**2 * chord
        Tp = ct * 0.5 * rho * W**2 * chord

        result['alpha'][i] = alpha
        #dT = B * Np
        #dQ = B * r * Tp
        A = np.pi * Rtip**2
        #dCT = dT / (rho * A * (Omega * Rtip)**2)
        #dCP = dQ / (rho * A * (Omega * Rtip)**3)

        if k >= -2.0 / 3:  # Momentum region
            dCT = 4 * a * (1 - a) * F
        else:  # Empirical region
            dCT = (50/9 - 4*F)*a**2 - (40/9 - 4*F)*a + 8/9
        
        dCP = (1-a) * dCT + 0.5 * sigma_p * cd * (r/R)**3

        Vtip = av.oper['Omega'] * Rtip
        result['dCT'][i] = B * Np / (0.5 * rho * Vtip**2 * A)
        result['dCP'][i] = B * Tp * r * sweep[i] * av.oper['Omega'] / (0.5 * rho * Vtip**3 * A)
        result['Cl'][i] = cl
        result['Cd'][i] = cd

        #print(i, R, alpha)

        return R, result
    
    
    Vy = Omega * R
    theta = av.prop['twist'][-1]
    startfrom90, order = determine_quadrants(True, 0, Vy, theta)

    res = {}
    res['dCT'] = np.zeros(av.prop['nr'])
    res['dCP'] = np.zeros(av.prop['nr'])
    res['dFM'] = np.zeros(av.prop['nr'])
    res['Cl'] = np.zeros(av.prop['nr'])
    res['Cd'] = np.zeros(av.prop['nr'])
    res['alpha'] = np.zeros(av.prop['nr'])

    success = False
    for i in range(av.prop['nr']):
        for j in range(len(order)):  # Quadrant orders. In most cases, it should find root in the first quadrant searched.
            phimin, phimax = order[j]

            backwardsearch = False
            if not startfrom90:
                if phimin == -np.pi / 2 or phimax == -np.pi / 2:  # q2 or q4
                    backwardsearch = True
            else:
                if phimax == np.pi / 2:  # q1
                    backwardsearch = True

            # Find bracket
            success, phiL, phiU = firstbracket(lambda phi: residual_and_result(phi, i, res)[0], phimin, phimax, 10, backwardsearch)

            def solve(res):
                phistar = brentq(lambda phi: residual_and_result(phi, i, res)[0], phiL, phiU, xtol=1e-9)
                return phistar

            # Once bracket is found, solve root-finding problem and compute loads
            if success:
                phistar = solve(res)

                _, res = residual_and_result(phistar, i, res)


    res['converged'] = True
    res['invalids'] = np.where(
        (res['alpha'] * 180/np.pi < falphas.min()) | 
        (res['alpha'] * 180/np.pi > falphas.max())
        )[0]
    
    res['CT'] = np.trapz(res['dCT'], av.prop['r0_rt'])
    res['CP'] = np.trapz(res['dCP'], av.prop['r0_rt'])
    res['FM'] = np.sign(res['CT']) * np.abs(res['CT']) ** (2/3) / (np.sqrt(2) * np.abs(res['CP']))
    
    av.res = res
    return av  # Return None if no bracket is found


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

    av.airfoil_data = load_foil(av.prop['foil_path'])
    av.airfoil_data = run_xfoil(av.airfoil_data)

    av.oper['V'] = 10
    av = guaranteed_convergence_BEM(av)

    if not av.res['converged']:
        print("BEM did not converge")
        return
    fig, ax = plt.subplots()
    ax.plot(av.prop['r0_rt'], av.res['alpha'])

    Js = np.linspace(-0.1, 0.1, 20)
    Js, CPs, CTs, FMs = operating_range(av, Js)

    fig, ax = plt.subplots()

    ax.plot(Js, CTs, label='CT')
    ax.legend()

    ax.set_ylim(0, 20)

    plt.show()

if __name__ == '__main__':

    main()