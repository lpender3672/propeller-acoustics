import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from app.routines import (
    load_prop_from_file
)

from app.routines_aero import (
    calc_mean_forces,
    calc_std_forces,
    fit_cexp,
    cexp,
    load_cell_calibration,
)


def plot_calibration(cal_data):

    tcal_data, qcal_data = cal_data
    
    to_plot = [
        ("Thrust",    tcal_data),
        ("Torque",    qcal_data),
    ]
    output_dir = Path("deliverables/final_report/tikz")

    sensor_names = ["Thrust", "Torque"]
    units = {"Thrust": "N", "Torque": "Nm"}
    for cal_label, data in to_plot:
        for idx, sensor in enumerate(sensor_names):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(data[:, 0, idx], data[:, 1, idx])
            #ax.set_title(f"{sensor} — {cal_label} Calibration")
            ax.set_xlabel(f"Measured {sensor} (-)")
            ax.set_ylabel(f"Applied {cal_label} ({units[cal_label]})")
            ax.grid(True)

            if cal_label == sensor:
                # fit a line
                pfit = np.polyfit(data[:, 0, idx], data[:, 1, idx], 1)
                x_fit = np.linspace(np.min(data[:, 0, idx]), np.max(data[:, 0, idx]), 100)
                y_fit = np.polyval(pfit, x_fit)
                label_fit = f"y = {pfit[0]:.2e}x {pfit[1]:+.2e}".replace("+-", "- ")
                ax.plot(x_fit, y_fit, "r--", label=label_fit)
                ax.legend(loc="upper left")

            # build a safe filename, e.g. Sensor_A_thrust.png
            safe_name = sensor.replace(" ", "_")
            filename = f"{safe_name}_{cal_label.lower()}.png"
            path = output_dir / filename
            fig.tight_layout()
            fig.savefig(path, bbox_inches='tight', dpi=300)
            plt.close(fig)


def plot_prop(
    prop_result_path,
    ax1,
    ax2,
    cal_data,
    label=None,
    rho=1.225,
):
    """
    Plot thrust and torque coefficients vs. speed with error bars.
    """
    tcal_data, qcal_data = cal_data
    full_path = Path(prop_result_path).resolve()
    fprop = full_path.parent.parent / "props" / full_path.name
    prop = load_prop_from_file(fprop)

    # Get the next color from the current color cycle using the correct method
    prop_color = ax1._get_lines.get_next_color()

    # find latest aero file for this prop
    candidates = list(full_path.glob("aero_*"))
    if not candidates:
        print("Warning: No aero data found for", prop_result_path)
        return None
    faero = max(candidates, key=lambda f: f.stat().st_mtime)
    aero_data = np.load(faero)

    # calculate means and stds
    mean_speed, mean_thrust, mean_torque = calc_mean_forces(
        aero_data, tcal_data, qcal_data
    )
    std_speed, std_thrust, std_torque = calc_std_forces(aero_data, tcal_data, qcal_data)

    rt = prop["rt"]
    A = np.pi * rt**2

    mask_t = mean_thrust > 1e-2
    mask_q = mean_torque > 1e-5

    speed_t = mean_speed[mask_t]
    thrust = mean_thrust[mask_t]
    err_speed_t = std_speed[mask_t]
    err_thrust = std_thrust[mask_t]

    speed_q = mean_speed[mask_q]
    torque = mean_torque[mask_q]
    err_speed_q = std_speed[mask_q]
    err_torque = std_torque[mask_q]

    # compute coefficients
    CT = thrust / (rho * A * speed_t**2 * rt**2)
    CQ = torque / (rho * A * speed_q**2 * rt**3)
    # propagate errors (approx)

    yerr_CT = err_thrust / (rho * A * speed_t**2 * rt**2)
    yerr_CQ = err_torque / (rho * A * speed_q**2 * rt**3)

    # Use explicit color for thrust coefficient plot
    ax1.plot(speed_t, CT, "o-", markersize=5, label=label, zorder=1, color=prop_color)
    # ax1.errorbar(speed_t, CT, xerr=err_speed_t, yerr=yerr_CT, fmt='none', ecolor='black', zorder=2)
    ax1.set_xscale("log")

    if speed_t.size:
        curr_min_t, curr_max_t = ax1.get_xlim()
        xmin_t = np.min(speed_t - err_speed_t) * 0.9
        xmax_t = np.max(speed_t + err_speed_t) * 1.1
        ax1.set_xlim(min(curr_min_t, xmin_t), max(curr_max_t, xmax_t))
    ax1.grid(True, which="both")
    ax1.set_xlabel("Speed [rad/s]")
    ax1.set_ylabel("Thrust Coefficient $C_T$")

    # Use same color for torque coefficient plot
    ax2.plot(speed_q, CQ, "o", markersize=5, label=label, zorder=1, color=prop_color)

    # ax2.errorbar(speed_q, CQ, xerr=err_speed_q, yerr=yerr_CQ, fmt='none', ecolor='black', zorder=2)
    ax2.set_xscale("log")
    if speed_q.size:
        curr_min_q, curr_max_q = ax2.get_xlim()
        xmin_q = np.min(speed_q - err_speed_q) * 0.9
        xmax_q = np.max(speed_q + err_speed_q) * 1.1
        ax2.set_xlim(min(curr_min_q, xmin_q), max(curr_max_q, xmax_q))

    # ok now fit curves to estimate the constant coefficients
    # thrust curve is constant enough to be averaged

    converged_thrust_coefficient = np.mean(CT)
    
    # fit torque curve
    params_q = fit_cexp(np.log10(speed_q), CQ)
    speed_cont_q = np.logspace(
        np.log10(np.min(speed_q)), np.log10(np.max(speed_q)), 1000
    )
    CQ_cont = cexp(np.log10(speed_cont_q), *params_q)
    
    # Use same color for fit line with solid style
    ax2.plot(speed_cont_q, CQ_cont, "-", color=prop_color)

    converged_torque_coefficient = params_q[0] + params_q[2]

    FM = converged_thrust_coefficient**(3 / 2) / (converged_torque_coefficient * np.sqrt(2))

    print(
        f"Prop: {label} | CT: {converged_thrust_coefficient:.7f} | CQ: {converged_torque_coefficient:.7f}, | FM: {FM:.7f}"
    )
    
    ax2.grid(True, which="both")
    ax2.set_xlabel("Speed [rad/s]")
    ax2.set_ylabel("Torque Coefficient $C_Q$")

    return ax1, ax2




def plot_FM(prop_result_path, ax, cal_data, label=None):

    tcal_data, qcal_data = cal_data

    full_path = Path(prop_result_path).resolve()
    fprop = full_path.parent.parent / "props" / full_path.name
    prop = load_prop_from_file(fprop)

    candidates = list(full_path.glob("aero_*"))
    if not candidates:
        print("Warning: No aero data found for", prop_result_path)
        return None

    faero = max(candidates, key=lambda f: f.stat().st_mtime)

    aero_data = np.load(faero)

    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    avg_speed = np.mean(motor_data[:, :, 1], axis=1) * 2 * np.pi / 60
    avg_speed = np.abs(avg_speed)
    avg_raw_forces = np.mean(force_data[:, :, 1:], axis=1)

    avg_thrust = np.interp(avg_raw_forces[:, 0], tcal_data[:, 0, 0], tcal_data[:, 1, 0])
    avg_torque = np.interp(avg_raw_forces[:, 1], qcal_data[:, 0, 1], qcal_data[:, 1, 1])

    filtered_speed = avg_speed[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]
    filtered_thrust = avg_thrust[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]
    filtered_torque = avg_torque[(avg_thrust > 1e-2) & (avg_torque > 1e-4)]

    D = 0.127
    rho = 1.225
    A = np.pi * (D / 2) ** 2
    CT = filtered_thrust / (rho * A * filtered_speed**2 * D**2)
    CQ = filtered_torque / (rho * A * filtered_speed**2 * D**2 * D / 2)

    FM = CT ** (3 / 2) / CQ

    ax.semilogx(filtered_speed, FM, "-o", label=label)

    ax.grid(True, which="both")
    ax.set_xlabel("Speed [rad/s]")
    ax.set_ylabel("Figure of Merit (FM)")

    return ax


def sweep_comparison_plot(cal_data):

    fig, ax = plt.subplots(2, 1)

    plot_prop("app/results/printed5045.prop", ax[0], ax[1], cal_data, label="0 deg sweep")
    plot_prop("app/results/5045_s15.prop", ax[0], ax[1], cal_data, label="15 deg sweep")
    plot_prop("app/results/5045_s30.prop", ax[0], ax[1], cal_data, label="30 deg sweep")
    plot_prop("app/results/5045_s45.prop", ax[0], ax[1], cal_data, label="45 deg sweep")

    ax[1].legend(loc="upper left")


def printed_comparison_plot(cal_data):

    fig, ax = plt.subplots(2, 1)

    plot_prop("app/results/dalprop5045.prop", ax[0], ax[1], cal_data, label="dalprop 2 blade")
    plot_prop("app/results/printed5045.prop", ax[0], ax[1], cal_data, label="printed 2 blade")
    plot_prop("app/results/dalprop5045bnr.prop", ax[0], ax[1], cal_data, label="dalprop 3 blade")
    plot_prop("app/results/printed5045bnr.prop", ax[0], ax[1], cal_data, label="printed 3 blade")

    ax[1].legend(loc="upper left")


def diameter_comparison_plot(cal_data):

    fig, ax = plt.subplots(2, 1)

    plot_prop("app/results/dalprop5045.prop", ax[0], ax[1], cal_data, label="dalprop 5045")
    plot_prop("app/results/dalprop6045.prop", ax[0], ax[1], cal_data, label="dalprop 6045")
    plot_prop("app/results/dalprop4045.prop", ax[0], ax[1], cal_data, label="dalprop 4045")
    plot_prop("app/results/printed5045.prop", ax[0], ax[1], cal_data, label="printed 5045")

    ax[1].legend(loc="upper left")


def FoM_comparison_plot(cal_data):
    
    fig, ax = plt.subplots(1, 1)

    plot_FM("app/results/dalprop5045.prop", ax, cal_data)
    plot_FM("app/results/printed5045.prop", ax, cal_data)
    plot_FM("app/results/dalprop4045.prop", ax, cal_data)
    plot_FM("app/results/dalprop6045.prop", ax, cal_data)

    ax.legend(loc="upper left")


if __name__ == "__main__":

    cal_data = load_cell_calibration()

    plot_calibration(cal_data)

    sweep_comparison_plot(cal_data)
    printed_comparison_plot(cal_data)
    diameter_comparison_plot(cal_data)
    FoM_comparison_plot(cal_data)
    plt.tight_layout()
    plt.show()

