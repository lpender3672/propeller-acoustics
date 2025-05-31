from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.optimize import curve_fit, OptimizeWarning
from scipy import stats

from app.routines import (
    load_prop_from_file,
)


def load_cell_calibration():
    tcal_data = np.load("app/bcal_thrust.npy")
    qcal_data = np.load("app/bcal_torque.npy")
    
    cal_data = (tcal_data, qcal_data)

    return cal_data


def cexp(x, a, b, c):
    """Converging exponential function."""
    return a * (1 - np.exp(-b * x)) + c

def fit_cexp(x, y, initial_guess = [1000, 5, -1000]):
    """Fits a converging exponential to the data."""

    # sort x and y to ensure they are in the same order
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    try:
        params, pcov = curve_fit(cexp, x, y, p0=initial_guess)
        
        # calc 90% confidence intervals
        n = len(x)
        p = len(params)
        dof = max(0, n - p)
        
        # t-value for 90% confidence interval
        t_value = stats.t.ppf(1.0 - 0.1/2, dof)
        
        #  standard errors
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate confidence intervals
        conf_intervals = []
        for i, p in enumerate(params):
            conf_intervals.append([p - t_value * perr[i], p + t_value * perr[i]])
        
        return params, conf_intervals
    except RuntimeError:
        
        print("Error: Curve fitting failed. Returning max value.")
        return [0, 0, np.max(y)], None


def calc_mean_forces(aero_data, tcal_data, qcal_data):
    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    mean_speed = np.mean(motor_data[:, :, 1], axis=1) * -2 * np.pi / 60

    mean_raw_forces = np.mean(force_data[:, :, 1:], axis=1)

    # interpolate to calibration data
    mean_thrust = np.interp(
        mean_raw_forces[:, 0], tcal_data[:, 0, 0], tcal_data[:, 1, 0], np.nan, np.nan
    )
    mean_torque = np.interp(
        mean_raw_forces[:, 1], qcal_data[:, 0, 1], qcal_data[:, 1, 1], np.nan, np.nan
    )
    return mean_speed, mean_thrust, mean_torque


def calc_std_forces(aero_data, tcal_data, qcal_data):

    force_data = aero_data["force_data"]
    motor_data = aero_data["motor_data"]

    # speed samples and std
    speed_samples = motor_data[:, :, 1] * 2 * np.pi / 60
    std_speed = np.std(speed_samples, axis=1)

    raw_thrust = force_data[:, :, 1]
    raw_torque = force_data[:, :, 2]

    thrust_samples_cal = np.stack(
        [
            np.interp(
                raw_thrust[i, :], tcal_data[:, 0, 0], tcal_data[:, 1, 0], np.nan, np.nan
            )
            for i in range(raw_thrust.shape[0])
        ],
        axis=0,
    )
    torque_samples_cal = np.stack(
        [
            np.interp(
                raw_torque[i, :], qcal_data[:, 0, 1], qcal_data[:, 1, 1], np.nan, np.nan
            )
            for i in range(raw_torque.shape[0])
        ],
        axis=0,
    )

    std_thrust = np.std(thrust_samples_cal, axis=1)
    std_torque = np.std(torque_samples_cal, axis=1)

    return std_speed, std_thrust, std_torque



def calc_aero_coefficients(
    results_path,
    cal_data,
    rho=1.225,
):

    tcal_data, qcal_data = cal_data
    results_path = Path(results_path).resolve()

    output_aero_coefficients = {}

    for prop_result_path in results_path.glob("**/*.prop/"):

        full_path = Path(prop_result_path).resolve()

        if ('old' in full_path.parts or
            'noprop.prop' in full_path.parts or
            'nonlinear' in full_path.name):
            continue

        fprop = full_path.parent.parent / "props" / full_path.name
        if not fprop.exists():
            print("Warning: No prop file found for", prop_result_path)
            continue
        prop = load_prop_from_file(fprop)

        # find latest aero file for this prop
        candidates = list(full_path.glob("aero_*"))
        if not candidates:
            print("Warning: No aero data found for", prop_result_path)
            continue
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
        mask_q = mean_torque > 1e-4

        speed_t = mean_speed[mask_t]
        thrust = mean_thrust[mask_t]
        std_thrust_filtered = std_thrust[mask_t]

        speed_q = mean_speed[mask_q]
        torque = mean_torque[mask_q]
        std_torque_filtered = std_torque[mask_q]

        # compute coefficients
        CT = thrust / (rho * A * speed_t**2 * rt**2)
        CQ = torque / (rho * A * speed_q**2 * rt**3)
        
        # compute standard deviations in coefficients
        # std_CT = CT * (std_thrust/thrust)
        std_CT = CT * (std_thrust_filtered / thrust)
        std_CQ = CQ * (std_torque_filtered / torque)

        if np.std(CT) / np.mean(CT) > 0.1:
            print(f"Warning: High variance in thrust coefficients for {prop_result_path.name}")
            params_t, _ = fit_cexp(np.log10(speed_q), CT)
            converged_thrust_coefficient = params_t[0] + params_t[2]
        else:
            converged_thrust_coefficient = np.mean(CT)
        
        # fit torque curve
        params_q, bounds_q = fit_cexp(np.log10(speed_q), CQ)
        
        converged_torque_coefficient = params_q[0] + params_q[2]

        # calc improved std_CQ using the confidence intervals
        improved_std_CQ = None
        if bounds_q is not None:
              # [lower, upper]
            a_bounds = bounds_q[0]
            c_bounds = bounds_q[2]
            
            # coefficient is (a+c)
            coef_lower = a_bounds[0] + c_bounds[0]
            coef_upper = a_bounds[1] + c_bounds[1]
            
            # z-score (1.645 for 90% confidence)
            improved_std_CQ = (coef_upper - coef_lower) / (2 * 1.645)
            # this is rubbish 
        
        prop_name = prop_result_path.name.replace(".prop", "")

        output_aero_coefficients[prop_name] = [
            converged_thrust_coefficient, 
            converged_torque_coefficient,
            np.max(std_CT),  # max standard deviation in thrust coefficient
            np.max(std_CQ)  # Improved standard deviation in torque coefficient
        ]
        
    return output_aero_coefficients

if __name__ == "__main__":

    cal_data = load_cell_calibration()

    aerocoeffs = calc_aero_coefficients(
    "app/results/",
    cal_data,
    rho=1.225,
    )
    print(aerocoeffs)

