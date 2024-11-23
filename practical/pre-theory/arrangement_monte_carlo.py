# perform monte carlo simulation
# of load cell uncertainty measurements.

import numpy as np
from matplotlib import pyplot as plt

thetas = np.linspace(-np.pi/2, np.pi/2, 100)

cal_F_const = 10 #N
cal_M_const = 10 #Nm

d = 0.05 #m
N = 10000

# vals of calibration curves
a = 100
b = 100

delta_Mod_s = np.zeros(thetas.shape[0])
delta_F_s = np.zeros(thetas.shape[0])

i = 0


for theta in thetas:

    # calibration values
    low_F = np.random.normal(0, 0.01, N)
    low_Mod = np.random.normal(0, 0.01, N) / d
    cal_F = np.random.normal(cal_F_const, cal_F_const * 0.01, N)
    cal_Mod = np.random.normal(cal_M_const, cal_M_const * 0.01, N) / d


    R = np.array([
        [np.sin(theta), np.cos(theta)],
        [np.cos(theta), -np.sin(theta)]
    ])
    if np.isclose(np.linalg.det(R), 0).any():
        continue

    rhs_cal = np.array([cal_F, cal_Mod]).reshape(N, 2)
    rhs_low = np.array([low_F, low_Mod]).reshape(N, 2)

    R = np.tile(R, (N, 1, 1))

    R1_cal, R2_cal = np.linalg.solve(R, rhs_cal).T
    R1_low, R2_low = np.linalg.solve(R, rhs_low).T

    # v = a * R + b

    # calc a and b
    vlow = np.random.normal(0, 0.01, N)
    v1high = np.random.normal(a * R1_cal + b, 0.01, N)
    v2high = np.random.normal(a * R2_cal + b, 0.01, N)

    # calc a1 and b1 by solving the system of equations
    A = np.zeros((N, 2, 2))
    B = np.zeros((N, 2))

    A[:, 0, 0] = R1_low
    A[:, 0, 1] = 1
    A[:, 1, 0] = R1_cal
    A[:, 1, 1] = 1

    B[:, 0] = vlow
    B[:, 1] = v1high

    a1, b1 = np.linalg.solve(A, B).T

    # calc a2 and b2 by solving the system of equations

    A = np.zeros((N, 2, 2))
    B = np.zeros((N, 2))

    A[:, 0, 0] = R2_low
    A[:, 0, 1] = 1
    A[:, 1, 0] = R2_cal
    A[:, 1, 1] = 1

    B[:, 0] = vlow
    B[:, 1] = v2high

    a2, b2 = np.linalg.solve(A, B).T

    # calculate delta Mod_s and delta F_s

    R1_final = a1 * R1_cal + b1
    R2_final = a2 * R2_cal + b2

    # calc mod and F from theta
    F = R1_final * np.sin(theta) + R2_final * np.cos(theta)
    Mod = R1_final * np.cos(theta) - R2_final * np.sin(theta)


    delta_Mod_s[i] = np.std(Mod) / np.mean(Mod)
    delta_F_s[i] = np.std(F) / np.mean(F)

    i += 1


plt.plot(thetas, delta_Mod_s)
plt.plot(thetas, delta_F_s)

plt.show()