# perform monte carlo simulation
# of load cell uncertainty measurements.

import numpy as np


thetas = np.linspace(-np.pi/2, np.pi/2, 100)

for theta in thetas:

    F = 10 + np.random.normal(0, 0.01)
    M = 0  + np.random.normal(0, 0.01)

    # calculate uM and uF
    uF = np.std(F) / np.mean(F)
    uM = np.std(M) / np.mean(M)

    uR2 = np.sqrt(
        (uF * np.cos(theta)) ** 2 - (uM * np.sin(theta)) ** 2
    )
    uR1 = np.sqrt(
        - (uF * np.sin(theta)) ** 2 + (uM * np.cos(theta)) ** 2
    )
    