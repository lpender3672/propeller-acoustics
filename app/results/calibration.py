import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from app.routines_aero import (
    load_cell_calibration,
)

from app.routines_audio import (
    load_microphone_calibration,
)

def plot_loadcell_calibration(cal_data):

    tcal_data, qcal_data = cal_data

    tcal_data[:, 1, 1] = tcal_data[:, 1, 0]
    qcal_data[:, 1, 0] = qcal_data[:, 1, 1]

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
                ax.plot(x_fit, y_fit, "r--", label=label_fit, linewidth=1)
                ax.legend(loc="upper left")

            else:

                ax.errorbar(
                    data[:, 0, idx], data[:, 1, idx],
                    xerr = 200,
                    fmt='o', markersize=4)

            # build a safe filename, e.g. Sensor_A_thrust.png
            safe_name = sensor.replace(" ", "_")
            filename = f"{safe_name}_{cal_label.lower()}.png"
            path = output_dir / filename
            fig.tight_layout()
            fig.savefig(path, bbox_inches='tight', dpi=300)
            plt.close(fig)

def plot_microphone_calibration(raw_calib_data):

    calib_freqs = raw_calib_data[:, 0::2].astype(float)
    calib_data_dB = raw_calib_data[:, 1::2].astype(float)

    layoutf = np.loadtxt('app/results/microphone_states/gantry45.csv',
                         delimiter=',', skiprows=1, dtype=str)
    
    labels = layoutf[:,1]

    assert labels.shape[0] == calib_data_dB.shape[1], "Frequency and dB data must have the same length"

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, label in enumerate(labels):
        ax.plot(calib_freqs[:, i], calib_data_dB[:, i],
                marker='o', linestyle='-', label=label,
                markersize=3, linewidth=1.5)

    ax.set_xscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Serial Number')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Sound Pressure Level (dB)')

    fig.tight_layout()
    fig.savefig(
        'deliverables/final_report/figures/microphone_calibration.png',
        dpi=300
    )


if __name__ == "__main__":

    loadcell_cal_data = load_cell_calibration()

    plot_loadcell_calibration(loadcell_cal_data)

    microphone_cal_data = load_microphone_calibration()

    plot_microphone_calibration(microphone_cal_data)

    plt.show()