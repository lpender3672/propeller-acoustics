import numpy as np
import scipy.special as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



global b, d, B

Omega = 2*np.pi*1000
B = 3 # number of blades
R = 2.5 * 25.1e-3
b = 0.01 # width of blade
d = 0.1 # separation between two blades

Ft = 10 # N
T = 0.1 # Nm

T = 298.15
P = 101325
rho = 1.225
gamma = 1.4
R = 287.05

c = np.sqrt(gamma * R * T)
M = Omega * R / c

omega_1 = B * Omega
k_1 = omega_1 / c

r, theta = np.meshgrid(
    np.linspace(0, 100, 100),
    np.linspace(0, np.pi, 100)
    )
phi = 0
t = 0

# compute a_m from solving integral
a_m = 0.5 * R # for now
Fd = T / a_m 
M = a_m * Omega / c

grid_shape = r.shape

def alpha(n):
    if n == 1:
        return 1
    else:
        return np.sin(n * b * np.pi / d) / (n * b * np.pi / d)

# beta represents DTFT coefficients of f_z(psi)
# delta represents DTFT coefficients of f_(phi)

num_harmonics = 5
num_sources = 10

beta = np.zeros(num_sources)
delta = np.zeros(num_sources)


harmonics = np.zeros((num_harmonics, *grid_shape))
for n in range(1, num_harmonics + 1):
    sum_source_1 = np.zeros((num_sources, *grid_shape))
    sum_source_2 = np.zeros((num_sources, *grid_shape))
    for l in range(0, num_sources):
        sum_source_1[l] = ((beta[l] * Ft * np.cos(theta) + delta[l] * Fd * (n*B - l) / (n*B*M)) * sp.spherical_jn(n * B - l, n * B * M * np.sin(theta)) *
                      np.sin( n * k_1 * (r - c*t) + (n * B - l)*(phi - np.pi / 2)))

        sum_source_2[l] = ((beta[l] * Ft * np.cos(theta) + delta[l] * Fd * (n*B + l) / (n*B*M)) * sp.spherical_jn(n * B + l, n * B * M * np.sin(theta)) *
                        np.sin( n * k_1 * (r - c*t) + (n * B + l)*(phi + np.pi / 2)))

    harmonics[n-1] = 2 * n * k_1 * alpha(n) * (np.sum(sum_source_1, axis=0) + np.sum(sum_source_2, axis=0))

    print(f"Harmonic {n} done")

p = 1 / (4 * np.pi * r) * np.sum(harmonics, axis=0)


print(p)
# plot p in 2d first
# not 3d
fix,ax = plt.subplots(1,1,subplot_kw={'projection':'polar'})

ax.pcolormesh(theta, r, p, shading='gouraud')

plt.show()

