import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import brentq, root

from app.routines import (
    XFOIL_INSTALLED,
    AppVars,
    correct_clcd_sweep,
    interpolate_clcd,
    load_foil,
    load_oper_from_file,
    load_prop_from_file,
    run_xfoil,
)


def static_bem_basic(av):
    # most basic implementation derived in supervision
    # neglect common corrections like tip loss and swirl

    nr = av.prop["nr"]
    res = {
        "dCT": np.zeros(nr),
        "dCQ": np.zeros(nr),
        "Cl": np.zeros(nr),
        "Cd": np.zeros(nr),
        "alpha": np.zeros(nr),
        "a": np.zeros(nr),
        "invalids": np.zeros(0)
    }
    
    tol = 1e-6
    max_iter = 100
    rf = 0.3
    
    for i in range(nr):
        x = av.prop["r0_rt"][i]
        if x < 0.1:
            continue
            
        twist = av.prop["twist"][i]
        c = av.prop["c"][i]
        B = av.prop["B"]
        r = x * av.prop["rt"]
        sigma = B * c / (2 * np.pi * r)
        
        a = 0.1 * x
        for j in range(max_iter):
            phi = np.arctan(a)  # no swirl so phi = arctan(a)
            alpha = twist - phi
            
            Cl, Cd = interpolate_clcd(av.xfoil_data, alpha, 5e5)
            Cl, Cd = correct_clcd_sweep(Cl, Cd, av.prop["sweep"][i])
            c_n = Cl * np.cos(phi) + Cd * np.sin(phi)
            

            V_rel_squared_norm = 1+a**2
            new_a = sigma * c_n * V_rel_squared_norm / (4.0 + sigma * c_n)
            new_a = max(0.001, min(1.5, new_a))
            
            if abs(new_a - a) < tol:
                break
            a = rf * new_a + (1-rf) * a
        else:
            res["invalids"] = np.append(res["invalids"], i)
        
        phi = np.arctan(a)
        alpha = twist - phi
        Cl, Cd = interpolate_clcd(av.xfoil_data, alpha, 5e5)
        c_n = Cl * np.cos(phi) + Cd * np.sin(phi)
        c_t = Cl * np.sin(phi) - Cd * np.cos(phi)
        
        V_rel_squared_norm = 1+a**2
        res["dCT"][i] = 2*sigma * c_n * x**2 * V_rel_squared_norm
        res["dCQ"][i] = 2*sigma * c_t * x**3 * V_rel_squared_norm
        res["Cl"][i] = Cl
        res["Cd"][i] = Cd
        res["alpha"][i] = alpha
        res["a"][i] = a
    
    res['CT'] = np.trapezoid(res["dCT"], av.prop["r0_rt"])
    res['CQ'] = np.trapezoid(res["dCQ"], av.prop["r0_rt"])
    res['FM'] = res['CT']**(3/2) / (np.sqrt(2) * np.abs(res['CQ'])) if res['CQ'] != 0 else 0
    res["converged"] = len(res["invalids"]) == 0
    
    av.res = res
    return av

def static_bem_tiploss(av):
    # basic implemention with added correction
    # Prantl tiploss factor from his simple vortex wake model

    nr = av.prop["nr"]
    res = {
        "dCT": np.zeros(nr),
        "dCQ": np.zeros(nr),
        "Cl": np.zeros(nr),
        "Cd": np.zeros(nr),
        "alpha": np.zeros(nr),
        "a": np.zeros(nr),
        "F_tip": np.zeros(nr),
        "invalids": np.zeros(0)
    }
    
    tol, max_iter = 1e-6, 100
    B = av.prop["B"]
    
    for i in range(nr):
        x = av.prop["r0_rt"][i]
        if x < 0.1:
            continue
            
        twist = av.prop["twist"][i]
        c = av.prop["c"][i]
        r = x * av.prop["rt"]
        sigma = B * c / (2 * np.pi * r)
        
        # axial induction iteration with tip loss
        a = 0.1 * x
        for j in range(max_iter):
            phi = np.arctan(a)
            alpha = twist - phi
            
            Cl, Cd = interpolate_clcd(av.xfoil_data, alpha, 5e5)
            Cl, Cd = correct_clcd_sweep(Cl, Cd, av.prop["sweep"][i])
            c_n = Cl * np.cos(phi) + Cd * np.sin(phi)
            
            # Prandtl correction
            if x > 0.99:
                F_tip = 0.5
            else:
                f_tip = (B/2) * (1-x) / x / np.sin(phi)
                F_tip = (2/np.pi) * np.arccos(np.exp(-f_tip))
                F_tip = max(0.1, F_tip) # limit is referenced somewhere

            c_n_corrected = c_n * F_tip
            V_rel_squared_norm = 1+a**2
            new_a = sigma * c_n_corrected * V_rel_squared_norm / (4.0 + sigma * c_n_corrected)
            new_a = max(0.001, min(1.5, new_a))
            
            if abs(new_a - a) < tol:
                break
            a = 0.3 * new_a + 0.7 * a
        else:
            res["invalids"] = np.append(res["invalids"], i)
        
        phi = np.arctan(a)
        alpha = twist - phi
        Cl, Cd = interpolate_clcd(av.xfoil_data, alpha, 5e5)
        
        if x > 0.99:
            F_tip = 0.5
        else:
            f_tip = (B/2) * (1-x) / x / np.sin(phi)
            F_tip = (2/np.pi) * np.arccos(np.exp(-f_tip))
            F_tip = max(0.1, F_tip)
        
        c_n = (Cl * np.cos(phi) + Cd * np.sin(phi)) * F_tip
        c_t = (Cl * np.sin(phi) - Cd * np.cos(phi)) * F_tip
        
        V_rel_squared_norm = 1+a**2
        res["dCT"][i] = 2*sigma * c_n * x**2 * V_rel_squared_norm
        res["dCQ"][i] = 2*sigma * c_t * x**3 * V_rel_squared_norm
        res["Cl"][i] = Cl
        res["Cd"][i] = Cd
        res["alpha"][i] = alpha
        res["a"][i] = a
        res["F_tip"][i] = F_tip
    
    res['CT'] = np.trapezoid(res["dCT"], av.prop["r0_rt"])
    res['CQ'] = np.trapezoid(res["dCQ"], av.prop["r0_rt"])
    res['FM'] = res['CT']**(3/2) / (np.sqrt(2) * np.abs(res['CQ'])) if res['CQ'] != 0 else 0
    res["converged"] = len(res["invalids"]) == 0
    
    av.res = res
    return av


def static_bem_swirl(av):

    nr = av.prop["nr"]
    res = {
        "dCT": np.zeros(nr),
        "dCQ": np.zeros(nr),
        "Cl": np.zeros(nr),
        "Cd": np.zeros(nr),
        "alpha": np.zeros(nr),
        "a": np.zeros(nr),
        "a_prime": np.zeros(nr),
        "invalids": np.zeros(0)
    }
    
    tol, max_iter = 1e-6, 100
    
    for i in range(nr):
        x = av.prop["r0_rt"][i]
        if x < 0.1:
            continue
            
        twist = av.prop["twist"][i]
        c = av.prop["c"][i]
        B = av.prop["B"]
        r = x * av.prop["rt"]
        sigma = B * c / (2 * np.pi * r)
        
        # now have an additional swirl induction factor a_prime
        # still with respect to same local omega*r velocity
        a, a_prime = 0.1 * x, 0.01
        for j in range(max_iter):
            phi = np.arctan(a / (1 - a_prime))
            phi = max(0.01, min(np.pi/2 - 0.01, phi))
            alpha = twist - phi
            
            Cl, Cd = interpolate_clcd(av.xfoil_data, alpha, 5e5)
            Cl, Cd = correct_clcd_sweep(Cl, Cd, av.prop["sweep"][i])
            c_n = Cl * np.cos(phi) + Cd * np.sin(phi)
            c_t = Cl * np.sin(phi) - Cd * np.cos(phi)
            
            V_rel_squared_norm = a**2 + (1 - a_prime)**2
            
            # axial induction update
            if c_n > 0:
                denom = 4.0/x - sigma * c_n
                if denom > 0:
                    a_squared = sigma * c_n * (1 - a_prime)**2 / denom
                    new_a = np.sqrt(max(0, a_squared))
                else:
                    new_a = sigma * c_n * V_rel_squared_norm / (4.0 + sigma * c_n)
            else:
                new_a = a
            
            # swirl induction update
            if c_t != 0 and a > 1e-6:
                new_ap = sigma * c_t * V_rel_squared_norm * x / (4.0 * a)
            else:
                new_ap = a_prime
            
            new_a = max(0.001, min(1.5, new_a))
            new_ap = max(-0.9, min(0.9, new_ap))
            
            if max(abs(new_a - a), abs(new_ap - a_prime)) < tol:
                break
            
            a = 0.2 * new_a + 0.8 * a
            a_prime = 0.2 * new_ap + 0.8 * a_prime
        else:
            res["invalids"] = np.append(res["invalids"], i)
        
        phi = np.arctan(a / (1 - a_prime))
        alpha = twist - phi
        Cl, Cd = interpolate_clcd(av.xfoil_data, alpha, 5e5)
        c_n = Cl * np.cos(phi) + Cd * np.sin(phi)
        c_t = Cl * np.sin(phi) - Cd * np.cos(phi)
        
        V_rel_squared_norm = a**2 + (1 - a_prime)**2
        res["dCT"][i] = 2*sigma * c_n * x**2 * V_rel_squared_norm
        res["dCQ"][i] = 2*sigma * c_t * x**3 * V_rel_squared_norm
        res["Cl"][i] = Cl
        res["Cd"][i] = Cd
        res["alpha"][i] = alpha
        res["a"][i] = a
        res["a_prime"][i] = a_prime
    
    res['CT'] = np.trapezoid(res["dCT"], av.prop["r0_rt"])
    res['CQ'] = np.trapezoid(res["dCQ"], av.prop["r0_rt"])
    res['FM'] = res['CT']**(3/2) / (np.sqrt(2) * np.abs(res['CQ'])) if res['CQ'] != 0 else 0
    res["converged"] = len(res["invalids"]) == 0
    
    av.res = res
    return av


def compare_tilloss_and_swirl(av):
    # to compare the different methods and their extensions

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    av = static_bem_basic(av)
    print(f"AXIAL: CT: {av.res['CT']}, CQ: {av.res['CQ']}, FM: {av.res['FM']}")
    axes[0].plot(av.prop["r0_rt"], av.res["dCT"], label="axial")
    axes[1].plot(av.prop["r0_rt"], av.res["dCQ"], label="axial")

    av = static_bem_swirl(av)
    print(f"SWIRL: CT: {av.res['CT']}, CQ: {av.res['CQ']}, FM: {av.res['FM']}")
    axes[0].plot(av.prop["r0_rt"], av.res["dCT"], label="swirl")
    axes[1].plot(av.prop["r0_rt"], av.res["dCQ"], label="swirl")

    av = static_bem_tiploss(av)
    print(f"TIPLOSS: CT: {av.res['CT']}, CQ: {av.res['CQ']}, FM: {av.res['FM']}")
    axes[0].plot(av.prop["r0_rt"], av.res["dCT"], label="tiploss")
    axes[1].plot(av.prop["r0_rt"], av.res["dCQ"], label="tiploss")

    axes[0].legend()
    axes[1].legend()
    axes[0].grid(True, which='both')
    axes[1].grid(True, which='both')

    axes[0].set_ylabel("Local thrust coefficient $dC_T$ [-]")
    axes[1].set_ylabel("Local torque coefficient $dC_Q$ [-]")

    axes[1].set_xlabel("Normalized radius $r/r_t$ [-]")

    return fig, axes


def main():

    av = AppVars()

    av.oper = load_oper_from_file("app/app_vars.json")
    av.prop = load_prop_from_file("app/props/dalprop5045.prop")

    #av.airfoil_data = load_foil(av.prop["foil_path"])
    av.airfoil_data = load_foil("app/foils/naca4412.surf")
    av.xfoil_data = run_xfoil(av.airfoil_data)

    #av = static_bem(av)

    fig, axes = compare_tilloss_and_swirl(av)
    fig.savefig('deliverables/final_report/figures/bem_comparison.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    main()
