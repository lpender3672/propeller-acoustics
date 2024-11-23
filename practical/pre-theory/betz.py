

# Design of Optimum Propellers
# Charles N. Adkins*
# Falls Church, Virginia 22042
# Douglas Aircraft Company, Long Beach, California 90846

import numpy as np
from matplotlib import pyplot as plt

# read data
xf,zf = np.loadtxt('practical/pre-theory/clarkY.surf', unpack=True, skiprows=2)
# split to upper and lower surfaces
idx = np.where(zf == 0)[0][1]
xf_upper = xf[:idx]
zf_upper = zf[:idx]
xf_lower = xf[idx:]
zf_lower = zf[idx:]
xf = np.concatenate((xf_upper, xf_lower[::-1]))
zf = np.concatenate((zf_upper, zf_lower[::-1]))

# read data
alpha_foil, Cl_foil, Cd_foil, _, _, _, _ = np.loadtxt('practical/pre-theory/clarkY.dat', unpack=True, skiprows=11)

V = 80 # m/s

R = 5/2 * 25.4e-3
B = 3
Omega = 10000 * 2 * np.pi / 60 # rad/s
lamda = V / (Omega * R) # advance ratio
nu = 1.48e-5
ro = 1.225

Nsect = 50

# Select an initial estimate for zeta
zeta = 0.0001
dzeta = 100
xi = np.linspace(0.2, 1, Nsect)
y = xi * R * Omega / V

idx = np.argmax(Cl_foil / Cd_foil)
alpha = alpha_foil[idx]
Cl = Cl_foil[idx]
Cd = Cd_foil[idx]


target_thrust_N = 4 # N
T_c = 2 * target_thrust_N / (ro * V**2 * np.pi * R**2)

V = np.sqrt(2 * target_thrust_N / (ro * np.pi * R**2))

while np.abs(dzeta/zeta) > 1e-3:
    phi_t = np.arctan(lamda * (1 + zeta / 2))
    f = B/2 * (1 - zeta) / np.sin(phi_t)
    F = 2 / np.pi * np.arccos(np.exp(-f))

    phi = np.arctan( np.tan(phi_t) / xi)

    G = F * np.cos(phi) * np.sin(phi)
    Wc = 4 * np.pi * lamda * G * V * R * zeta / (Cl * B)
    Re_c = Wc / nu

    print(np.max(Re_c) - np.min(Re_c))

    # TODO: find best airfoil at each chord Reynolds number
    alpha = alpha
    Cl = Cl
    Cd = Cd
    epsilon = Cd / Cl

    a = zeta / 2 * np.cos(phi)**2 * (1 - epsilon * np.tan(phi))
    a_prime = zeta / (2 * y) * np.cos(phi) * np.sin(phi) * (1 + epsilon / np.tan(phi))
    W = V * (1 + a) / np.sin(phi)
    # recompute step 3 for chord
    Wc = 4 * np.pi * lamda * G * V * R * zeta / (Cl * B)
    c = Wc / W

    beta = alpha + phi

    I1_prime = 4 * xi * G * (1 - epsilon * np.tan(phi)) 
    I2_prime = lamda * I1_prime / (2 * xi) * (1 + epsilon / np.tan(phi)) * np.cos(phi) * np.sin(phi)
    J1_prime = 4 * xi * G * (1 + epsilon / np.tan(phi))
    J2_prime = J1_prime / 2 * (1 - epsilon * np.tan(phi)) * np.cos(phi) ** 2

    # integrate wrt xi
    I1 = np.trapz(I1_prime, xi)
    I2 = np.trapz(I2_prime, xi)
    J1 = np.trapz(J1_prime, xi)
    J2 = np.trapz(J2_prime, xi)

    I1_over_2I2 = I1 / (2 * I2)
    new_zeta = I1_over_2I2 - (I1_over_2I2**2 - T_c / I2)**(1/2)
    
    if np.isnan(new_zeta):
        print("unable to reach target thrust")
        break

    P_c = J1 * zeta + J2 * zeta**2

    dzeta = new_zeta - zeta
    zeta = new_zeta

# cross section plot

for i in range(0, Nsect, 10):
    bi = -np.pi/2 + beta[i] - 2 * np.pi
    ci = 1e3 * c[i]
    r = 1e3 * R * xi[i]

    xof = ci * (xf - 0.25)
    zof = ci * (zf - 0.01)
    # rotate
    x = xof * np.cos(bi) - zof * np.sin(bi)
    z = xof * np.sin(bi) + zof * np.cos(bi)

    bideg = bi * 180 / np.pi
    plt.plot(x, z, label=f'r = {r:.2f}, beta={bideg:.2f}')

plt.axis('equal')

plt.legend()

plt.show()