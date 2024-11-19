# perform monte carlo simulation
# of load cell uncertainty measurements.

import numpy as np
from matplotlib import pyplot as plt

thetas = np.linspace(-np.pi/2, np.pi/2, 100)

cal_F = 10 #N
cal_M = 10 #Nm

d = 0.05 #m
N = 10000

# gradients of calibration curves
a1 = 5
a2 = 5

delta_Mod_s = np.zeros(thetas.shape[0])
delta_F_s = np.zeros(thetas.shape[0])

i = 0

for theta in thetas:

    # calibration values
    low_F = np.random.normal(0, 0.01, N)
    low_Mod = np.random.normal(0, 0.01, N) / d
    cal_F = np.random.normal(cal_F, cal_F * 0.01, N)
    cal_Mod = np.random.normal(cal_M, cal_M * 0.01, N) / d

    uV1 = np.random.normal(0, 0.01, N)
    uV2 = np.random.normal(0, 0.01, N)

    # calculate absolute uncertainty in uM and uF
    uF = np.std(cal_F)
    uMod = np.std(cal_Mod)

    # uncertainty due to 
    uR2 = np.sqrt(
        (uF * np.cos(theta)) ** 2 - (uMod * np.sin(theta)) ** 2
    )
    uR1 = np.sqrt(
        - (uF * np.sin(theta)) ** 2 + (uMod * np.cos(theta)) ** 2
    )

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

    A = np.zeros((N, 4, 4)) # A is currently always singular so need to fix this
    A[:, 0, 0] = R1_cal
    A[:, 1, 1] = R2_cal
    A[:, 2, 0] = R1_low
    A[:, 3, 1] = R2_low
    A[:, 0, 2] = 1
    A[:, 1, 3] = 1
    A[:, 2, 2] = 1
    A[:, 3, 3] = 1

    # if A singular continue
    if np.isclose(np.linalg.det(A), 0, atol = 0.001).any():
        continue

    B = np.zeros((N, 4))

    B[:, 0] = (uV1 - a1 * uR1).reshape(N)
    B[:, 1] = (uV2 - a2 * uR2).reshape(N) 
    B[:, 2] = (uV1 - a1 * uR1).reshape(N)
    B[:, 3] = (uV2 - a2 * uR2).reshape(N)

    # calculate calibration R1 and R2
    ua1,ub1,ua2,ub2 = np.linalg.solve(A, B).T

    v = 5
    delta_R1 = np.sqrt((v * ua1)**2 + ub1**2)
    delta_R2 = np.sqrt((v * ua2)**2 + ub2**2)

    delta_Mod = np.sqrt((np.cos(theta) * delta_R1)**2 + (-np.sin(theta) * delta_R2)**2)
    delta_F = np.sqrt((np.sin(theta) * delta_R1)**2 + (np.cos(theta) * delta_R2)**2)


    delta_Mod_s[i] = np.mean(delta_Mod)
    delta_F_s[i] = np.mean(delta_F)

    i += 1


plt.plot(thetas, d * delta_Mod_s)
plt.plot(thetas, delta_F_s)

plt.show()