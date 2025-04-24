import matplotlib.pyplot as plt
import numpy as np

# plot cal data


thrust_cal_data = np.load(
    "app/results/thrust_cal.npy"
)  # shape n, 2, 2 representing n samples, 0 for sensor data, 1 for applied force, 0 for thrust, 1 for torque
torque_cal_data = np.load("app/results/torque_cal.npy")
mixed_cal_data = np.load("app/results/mixed_cal.npy")

combined = np.concatenate((thrust_cal_data, torque_cal_data, mixed_cal_data), axis=0)

torque_cal_data = np.delete(
    torque_cal_data,
    np.where(np.isclose(torque_cal_data[:, 1, 1], 0.07, atol=0.01)),
    axis=0,
)


D = 5 * 25.4e-3


def sortx(xdata, ydata):
    args_sorted = np.argsort(xdata)
    return xdata[args_sorted], ydata[args_sorted]


def plot_raw(axes):

    axes[0].plot(
        *sortx(thrust_cal_data[:, 0, 0], thrust_cal_data[:, 1, 0]),
        "-o",
        label="Thrust calibration",
    )
    axes[0].plot(
        torque_cal_data[:, 0, 0],
        torque_cal_data[:, 1, 0],
        "-o",
        label="Variation from torque calibration",
    )
    axes[0].set_ylabel("Force (N)")
    axes[1].plot(
        *sortx(torque_cal_data[:, 0, 1], torque_cal_data[:, 1, 1]),
        "-o",
        label="Torque calibration",
    )
    axes[1].set_ylabel("Moment (Nm)")
    axes[1].plot(
        thrust_cal_data[:, 0, 1],
        thrust_cal_data[:, 1, 1],
        "-o",
        label="Variation from thrust calibration",
    )

    axes[0].plot(
        *sortx(mixed_cal_data[:, 0, 0], mixed_cal_data[:, 1, 0]),
        "-o",
        label="Mixed calibration",
    )
    axes[1].plot(
        *sortx(mixed_cal_data[:, 0, 1], mixed_cal_data[:, 1, 1]),
        "-o",
        label="Mixed calibration",
    )

    axes[0].grid()
    axes[1].grid()
    axes[0].legend()
    axes[1].legend()

    axes[1].set_xlabel("Raw ADC value")


def determine_coefficients(combined):

    # combined = np.delete(combined, np.where(np.isclose(combined[:,1,1], 0.07, atol=0.01)), axis=0)

    S_T = combined[:, 0, 0]  # thrust force sensor data
    S_N = combined[:, 0, 1]  # torque sensor data
    T = combined[:, 1, 0]  # applied force
    N = combined[:, 1, 1] / D  # applied torque
    # the torque is divided by a length to weight the torque and force equally
    # in the least squares fit

    X = np.column_stack((S_T, np.ones_like(S_T)))

    coeff_T, residuals_T, rank_T, s_T = np.linalg.lstsq(X, T, rcond=None)
    a, c = coeff_T
    b = 0

    S_T = combined[:, 0, 0]  # thrust force sensor data
    S_N = combined[:, 0, 1]  # torque sensor data
    T = combined[:, 1, 0]  # applied force
    N = combined[:, 1, 1] / D  # applied torque
    # the torque is divided by a length to weight the torque and force equally
    # in the least squares fit

    X = np.column_stack((S_T, S_N, np.ones_like(S_T)))

    coeff_N, residuals_N, rank_N, s_N = np.linalg.lstsq(X, N, rcond=None)
    d, e, f = coeff_N

    mse_T = residuals_T[0] / len(T)
    mse_N = residuals_N[0] / len(N)

    print(f"Force coefficients: a={a}, b={b}, c={c} mse {mse_T}")
    print(f"Torque coefficients: d={d}, e={e}, f={f} mse {mse_N}")

    return a, b, c, d, e, f


def calculate_point(S_T, S_N, *coefficients):

    a, b, c, d, e, f = coefficients

    T = a * S_T + b * S_N + c
    N = d * S_T + e * S_N + f

    return T, N


def single_regression(thrust_cal_data, torque_cal_data):

    S_T = thrust_cal_data[:, 0, 0]  # thrust force sensor data
    S_N = thrust_cal_data[:, 0, 1]  # torque sensor data
    T = thrust_cal_data[:, 1, 0]  # applied force
    N = thrust_cal_data[:, 1, 1] / D  # applied torque
    # the torque is divided by a length to weight the torque and force equally
    # in the least squares fit

    X = np.column_stack(
        (S_N, np.ones_like(S_N))
    )  # relation between torque sensor and thrust
    coeff_T, residuals_N, rank_N, s_N = np.linalg.lstsq(X, T, rcond=None)

    X = np.column_stack(
        (S_T, np.ones_like(S_T))
    )  # relation between thrust sensor and thrust
    coeff_N, residuals_T, rank_T, s_T = np.linalg.lstsq(X, T, rcond=None)

    # then cal signal due to torque
    S_T = torque_cal_data[:, 0, 0]  # thrust force sensor data
    S_N = torque_cal_data[:, 0, 1]  # torque sensor data
    T = torque_cal_data[:, 1, 0]  # applied force
    N = torque_cal_data[:, 1, 1] / D  # applied torque

    # T = a*S_T + b
    S_N_due_to_thrust = (T - coeff_T[1]) / coeff_T[0]
    S_N_due_to_torque = S_N - S_N_due_to_thrust

    # then get the coefficients
    X = np.column_stack((S_N_due_to_torque, np.ones_like(S_T)))
    coeff_Ni, residuals_N, rank_N, s_N = np.linalg.lstsq(X, N, rcond=None)

    return coeff_T, coeff_N, coeff_Ni


def calculate_sr_point(S_T, S_N, *coefficients):

    coeff_T, coeff_N, coeff_Ni = coefficients

    T = coeff_T[0] * S_N + coeff_T[1]
    S_N_due_to_thrust = (T - coeff_T[1]) / coeff_T[0]
    S_N_due_to_torque = S_N - S_N_due_to_thrust
    N = coeff_Ni[0] * S_N_due_to_torque + coeff_Ni[1]

    return T, N


calcoefs = determine_coefficients(combined)
calcoefs_sr = single_regression(thrust_cal_data, torque_cal_data)


def plot_cal(axes):

    S_T = np.linspace(
        thrust_cal_data[:, 0, 0].min(), thrust_cal_data[:, 0, 0].max(), 100
    )
    S_N = np.linspace(
        thrust_cal_data[:, 0, 1].min(), thrust_cal_data[:, 0, 1].max(), 100
    )

    T, N = calculate_point(S_T, S_N, *calcoefs)
    # T, N = calculate_sr_point(S_T, S_N, *calcoefs_sr)
    N *= D

    axes[0].plot(S_T, T, label="Thrust from Thrust calibration (should be linear fit)")
    axes[1].plot(S_N, N, label="Torque from Thrust calibration (should be 0)")

    axes[0].set_ylabel("Force (N)")
    axes[1].set_ylabel("Moment (Nm)")
    axes[0].grid()
    axes[1].grid()
    axes[0].legend()
    axes[1].legend()

    axes[1].set_xlabel("Raw ADC value")

    S_T = np.linspace(
        torque_cal_data[:, 0, 0].min(), torque_cal_data[:, 0, 0].max(), 100
    )
    S_N = np.linspace(
        torque_cal_data[:, 0, 1].min(), torque_cal_data[:, 0, 1].max(), 100
    )

    T, N = calculate_point(S_T, S_N, *calcoefs)
    N *= D

    axes[0].plot(S_T, T, label="Thrust from Torque calibration (should be 0)")
    axes[1].plot(S_N, N, label="Torque from Torque calibration (should be linear fit)")

    # plot mixed data
    S_T = np.linspace(mixed_cal_data[:, 0, 0].min(), mixed_cal_data[:, 0, 0].max(), 100)
    S_N = np.linspace(mixed_cal_data[:, 0, 1].min(), mixed_cal_data[:, 0, 1].max(), 100)

    axes[0].plot(S_T, T, label="Thrust from Mixed calibration (should be linear fit)")
    axes[1].plot(S_N, N, label="Torque from Mixed calibration (should be linear fit)")

    axes[0].set_ylabel("Force (N)")
    axes[1].set_ylabel("Moment (Nm)")
    axes[0].grid()
    axes[1].grid()
    axes[0].legend()
    axes[1].legend()


def plot_dimensional(aero_data, ax1, ax2, label=None):

    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    avg_speed = np.mean(motor_data[:, :, 1], axis=1) * 2 * np.pi / 60
    avg_raw_forces = np.mean(force_data[:, :, 1:], axis=1)

    T, N = calculate_point(avg_raw_forces[:, 0], avg_raw_forces[:, 1], *calcoefs)

    ax1.loglog(avg_speed, T, "-o", label=label)
    ax1.grid(which="both")
    ax2.loglog(avg_speed, N, "-o", label=label)
    ax2.grid(which="both")

    return ax1, ax2


def plot_non_dimensional(aero_data, ax1, ax2, label=None):

    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    avg_speed = np.mean(motor_data[:, :, 1], axis=1) * 2 * np.pi / 60
    avg_raw_forces = np.mean(force_data[:, :, 1:], axis=1)

    T, N = calculate_point(avg_raw_forces[:, 0], avg_raw_forces[:, 1], *calcoefs)
    N *= D

    rho = 1.225
    A = np.pi * (D / 2) ** 2

    thrust_coefficient = T / (rho * A * avg_speed**2)
    torque_coefficient = N / (rho * A * avg_speed**2 * D / 2)

    ax1.semilogx(avg_speed, thrust_coefficient, "-o", label=label)
    ax1.grid(which="both")
    ax2.semilogx(avg_speed, torque_coefficient, "-o", label=label)
    ax2.grid(which="both")

    ax1.set_ylim([0, 4e-2])
    ax2.set_ylim([0, 5e-2])

    return ax1, ax2


def troubleshooting():

    S_T = torque_cal_data[:, 0, 0]  # thrust force sensor data
    S_N = torque_cal_data[:, 0, 1]  # torque sensor data
    T = torque_cal_data[:, 1, 0]  # applied force
    N = torque_cal_data[:, 1, 1] / D  # applied torque

    X = np.column_stack((S_T, S_N, np.ones_like(S_T)))

    coeff_N, residuals_N, rank_N, s_N = np.linalg.lstsq(X, N, rcond=None)
    d, e, f = coeff_N

    fig, ax = plt.subplots()

    ax.plot(*sortx(N, S_T), "-o")
    ax.plot(*sortx(N, S_N), "-o")


def plot_FM(aero_data, ax, label=None):

    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    avg_speed = np.mean(motor_data[:, :, 1], axis=1) * 2 * np.pi / 60
    avg_raw_forces = np.mean(force_data[:, :, 1:], axis=1)

    T, N = calculate_point(avg_raw_forces[:, 0], avg_raw_forces[:, 1], *calcoefs)
    N *= D

    rho = 1.225
    A = np.pi * (D / 2) ** 2

    thrust_coefficient = T / (rho * A * avg_speed**2)
    torque_coefficient = N / (rho * A * avg_speed**2 * D / 2)

    FM = thrust_coefficient ** (3 / 2) / torque_coefficient

    ax.plot(avg_speed, FM, "-o", label=label)
    ax.grid(which="both")

    return ax


fig, axes = plt.subplots(2, 1)

plot_raw(axes)
plot_cal(axes)

fig, axes = plt.subplots(2, 1)

folder = "app/results/0.2kg_torque"
toroidal_data = np.load(folder + "/foxeer_toroidal.npz")
dalprop_2blade = np.load(folder + "/dalprop5045.npz")
dalprop_3blade = np.load(folder + "/dalprop5045bnr.npz")

plot_non_dimensional(toroidal_data, *axes, label="Toroidal")
plot_non_dimensional(dalprop_2blade, *axes, label="2blade")
plot_non_dimensional(dalprop_3blade, *axes, label="3blade")
axes[-1].legend()

fig, ax = plt.subplots()

plot_FM(toroidal_data, ax, label="Toroidal")
plot_FM(dalprop_2blade, ax, label="2blade")
plot_FM(dalprop_3blade, ax, label="3blade")
ax.legend()
ax.set_ylim([0, 0.002])

troubleshooting()

plt.show()
