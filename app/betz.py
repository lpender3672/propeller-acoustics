
import numpy as np

from xfoil import XFoil
from xfoil.model import Airfoil

from matplotlib import pyplot as plt


def foil_data(airfoil_data, alpha, Re):
    
    xf = XFoil()
    xf.airfoil = Airfoil(
        airfoil_data[:,0],
        airfoil_data[:,1]
        )
    xf.Re = Re
    xf.M = 0.0
    xf.max_iter = 100
    xf.verbose = False

    if isinstance(alpha, float):
        cls = np.zeros(1)
        cds = np.zeros(1)
        alpha = [alpha]
    else:
        cls = np.zeros(len(alpha))
        cds = np.zeros(len(alpha))

    for i, a in enumerate(alpha):
        out = xf.a(a)
        print(out)
        cl, cd, _, _ = out
        cls[i] = cl
        cds[i] = cd

    return cls, cds


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
        phi = np.arctan( np.tan(phi_t) / xi)

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

    beta = av.prop['twist']

    R = av.prop['rt']
    B = av.prop['B']
    Omega = av.oper['Omega']
    nu = av.oper['nu']
    ro = av.oper['rho']
    xi = av.prop['r0_rt']
    c = av.prop['c']

    V = av.oper['V'] # m/s

    y = xi * R * Omega / V
    lamda = V / (Omega * R) # advance ratio

    # An initial estimate for phi can be obtained from Eq. (8) by setting zeta = 0
    phi = np.arctan((1 + 0) * lamda / xi)
    dphi = np.inf

    alphas = np.arange(-10, 20, 2.5)

    Cl0, Cd0 = foil_data(
        av.airfoil_data, alphas, 1e6
    )
    Cl_valid = ~np.isnan(Cl0)
    Cd_valid = ~np.isnan(Cd0)
    print(alphas, Cl0, Cd0)
    iters = 0

    while (np.abs(dphi / phi).all() > 1e-3 and
           iters < 1000):

        # Analysis of Arbitrary Designs
        # Charles N. Adkins*

        alpha = beta - phi
        # airfoil coefficients are known from the section data and alpha

        #Cl = Cl0 + alpha * 2 * np.pi
        #Cd = Cd0
        Cl = np.interp(alpha * 180/np.pi, alphas[Cl_valid], Cl0[Cl_valid])
        Cd = np.interp(alpha * 180/np.pi, alphas[Cd_valid], Cd0[Cd_valid])

        # better to wait and see if its out of bounds
        # phi = np.clip(phi, beta - alphas.min() * np.pi / 180, beta - alphas.max() * np.pi / 180)

        Cx = Cl * np.cos(phi) - Cd * np.sin(phi)
        Cz = Cl * np.sin(phi) + Cd * np.cos(phi)
        K = Cz / (4 * np.sin(phi)**2)
        K_prime = Cx / (4 * np.sin(phi) * np.cos(phi))
        sigma = B * c / (2 * np.pi * xi * R)
        phi_t = np.arctan( xi * np.tan(phi))
        F = 2 / np.pi * np.arccos(np.exp( - B/2 * (1 - xi) / np.sin(phi_t))) # Prandtl tip loss factor
        a = sigma * K * ( F - sigma * K)
        a_prime = sigma * K_prime * ( F + sigma * K_prime)
        # Viterna and Janetzke clip a and a_prime to 0.7
        a = np.clip(a, -0.7, 0.7)
        a_prime = np.clip(a_prime, -0.7, 0.7)

        # the Reynolds number is determined from the known chord and W
        W = V * (1 + a) / np.sin(phi)
        Re_c = W * c / nu
        #print(Re_c)

        new_phi = np.arctan(V * (1 + a) / (Omega * R * (1 - a_prime)))
        dphi = new_phi - phi
        phi = new_phi
        iters += 1

    # set interesting values
    
    CT_prime = (np.pi ** 3 / 4) * sigma * Cz * xi * F / ((F + sigma * K_prime) * np.cos(phi))**2
    CP_prime = CT_prime * np.pi * xi * Cx / Cz

    print(np.cos(phi))

    CT = np.trapz(CT_prime, R * xi)
    CP = np.trapz(CP_prime, R * xi)

    FM = np.abs(CT) ** (2/3) / np.abs(CP)

    print(f"CP: {CP}, CT: {CT}, FM: {FM}")

    return True



def main():

    xzf = np.loadtxt('app/foils/naca0012.surf')
    alpha = np.arange(-10,10,2)
    Cl,Cd = foil_data(
        xzf, alpha , 1e6
    )
    fig, ax = plt.subplots()
    ax.plot(alpha,Cl)
    ax.set_xlabel('Cl')
    ax.set_ylabel('Cd')

    plt.show()



if __name__ == '__main__':

    main()