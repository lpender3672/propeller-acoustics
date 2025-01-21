import numpy as np
from matplotlib import pyplot as plt

from routines import (
    load_oper_from_file,
    load_prop_from_file,
    run_xfoil,
    AppVars
)
from bem import (
    betz_off_design
)
# analysis of constant chord


def plot_twist_vs_FM(av):

    Ndists = 31
    alpha_tip = np.linspace(-5, 15, Ndists) * np.pi / 180
    # use arctan 1/r
    avcopy = av.copy()

    FMs = np.zeros(alpha_tip.shape)

    for i, alpha_t in enumerate(alpha_tip):
        avcopy.prop['twist'] = alpha_t * 4/np.pi * np.arctan(1/avcopy.prop['r0'])
        avcopy = betz_off_design(avcopy)

        if not avcopy.res['converged']:
            FMs[i] = np.nan
            continue

        ivlds = avcopy.res['invalids']
        if (len(ivlds) > 0 and ivlds[-1] > av.prop['nr'] // 2):
            FMs[i] = np.nan
            continue

        FMs[i] = avcopy.res['FM']

    fig, ax = plt.subplots()
    ax.plot(alpha_tip * 180 / np.pi, FMs)
    ax.set_xlabel('Tip Twist (deg)')
    ax.set_ylabel('FM')

def plot_foil_lift(av):

    fig, ax = plt.subplots()

    ax.plot(
        av.airfoil_data[:,2],
        av.airfoil_data[:,3],
        label='Cl'
    )
    ax.plot(
        av.airfoil_data[:,2],
        av.airfoil_data[:,4],
        label='Cd'
    )


def main():

    av = AppVars()

    av.oper = load_oper_from_file('app/app_vars.json')
    av.prop = load_prop_from_file('app/props/constant_chord.prop')

    av.airfoil_data = np.loadtxt(av.prop['foil_path'])
    av.airfoil_data = run_xfoil(av.airfoil_data)

    plot_twist_vs_FM(av)
    plot_foil_lift(av)

    plt.show()




if __name__ == '__main__':
    main()