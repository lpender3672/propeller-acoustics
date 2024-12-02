

# Design of Optimum Propellers
# Charles N. Adkins*
# Falls Church, Virginia 22042
# Douglas Aircraft Company, Long Beach, California 90846

import numpy as np
from matplotlib import pyplot as plt

from geometry import save_shape_data, plot3D

# read data
xf,zf = np.loadtxt('cfd/clarkY.surf', unpack=True, skiprows=2)
# split to upper and lower surfaces
idx = np.where(zf == 0)[0][1]
xf_upper = xf[:idx]
zf_upper = zf[:idx]
xf_lower = xf[idx:]
zf_lower = zf[idx:]
xf = np.concatenate((xf_upper, xf_lower[::-1]))
zf = np.concatenate((zf_upper, zf_lower[::-1]))

# read data
alpha_foil, Cl_foil, Cd_foil, _, _, _, _ = np.loadtxt('cfd/clarkY.dat', unpack=True, skiprows=11)


R = 5/2 * 25.4e-3
B = 3
Omega = 10000 * 2 * np.pi / 60 # rad/s
nu = 1.48e-5
ro = 1.225

Nsect = 50

# Select an initial estimate for zeta
zeta = 0.0001
dzeta = 100
xi = np.linspace(0.05, 1, Nsect)

idx = np.argmax(Cl_foil / Cd_foil)
alpha = alpha_foil[idx]
Cl = Cl_foil[idx]
Cd = Cd_foil[idx]

alpha *= np.pi / 180

target_thrust_N = 0.1 # N

V = 10 # m/s
T_c = 2 * target_thrust_N / (ro * V**2 * np.pi * R**2)
y = xi * R * Omega / V
lamda = V / (Omega * R) # advance ratio


while np.abs(dzeta/zeta) > 1e-3:
    phi_t = np.arctan(lamda * (1 + zeta / 2))
    f = B/2 * (1 - xi) / np.sin(phi_t)
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

else:

    # cross section plot

    nf = xf.shape[0]
    X = np.zeros((nf, Nsect))
    Y = np.zeros((nf, Nsect))
    Z = np.zeros((nf, Nsect))
    
    for i in range(0, Nsect):
        bi = -beta[i]
        ci = 1e3 * c[i]

        xof = ci * (xf - 0.25)
        zof = ci * (zf - 0.01)
        # rotate
        X[:,i] = xof * np.cos(bi) - zof * np.sin(bi)
        Z[:,i] = xof * np.sin(bi) + zof * np.cos(bi)

        bideg = beta[i] * 180 / np.pi
        Y[:,i] = 1e3 * R * xi[i] * np.ones(nf)
        #np.savetxt(f'practical/designs/clarkY/section_{i}.sldcrv', np.column_stack((X[:,i], Z[:,i], Y[:,i])))

    fig, ax = plot3D(X, Y, Z, B)
    save_shape_data(c, beta, R * xi, xf, zf, "practical/designs/clarkY.stl")

    # set limits
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)

    plt.show()