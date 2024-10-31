import numpy as np
import matplotlib.pyplot as plt

# code to produce cross-sections of the far-field intensity of a propeller model



r = np.linspace(1, 5, 100)
theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)

# 2D r,theta cross-section
R, Theta = np.meshgrid(r, theta)

n = 2 # number of blades
omega = 10e3 * 2*np.pi / 60 # rad/s
F = 8 # N (thrust)
a = 0.1 # m (propeller tip radius)
alpha = 5 * np.pi / 180 # rad (angle of attack)

# secondary variables
k = n * omega / (2*np.pi)
# far-field intensity
I = a**(2*n) * k ** (2*n+2) * F**2 / (2 ** (2*n+6) * np.pi **2 * R ** 2 * np.math.factorial(n)**2) * (alpha * n / (k * a) - np.cos(Theta))**2 * np.sin(Theta) ** (2 * n)

# plot I in polar coordinates

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# make pcolourmesh

c = ax.pcolormesh(Theta, R, np.log10(I), cmap='viridis')
# set colour limits
c.set_clim(-1, 10)

fig.colorbar(c, ax=ax, label='Intensity')


plt.show()


