

# so i want to compute aero coefficients for all tested propellers.
# but for some stinky reason the clarkY airfoil is xfoils nemesis

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from app.bem import(
    static_bem_swirl
)
from app.routines import (
    AppVars,
    load_prop_from_file,
    load_oper_from_file,
    run_xfoil,
    load_foil
)

from app.routines_audio import (
    parse_lookup_df,
)


def load_prop_and_run_bem_from_given_operating_and_xfoil_data(prop_dict, oper_dict, xfoil_data):

    av = AppVars()
    av.prop.update(prop_dict)
    av.oper.update(oper_dict)
    av.xfoil_data = xfoil_data
    av = static_bem_swirl(av)

    return av.res


def compile_props_and_plot(results_path):

    # chosing this airfoil which is very similar to ClarkY
    # but has no issues with xfoil
    fpath = 'app/foils/naca4412.surf'
    airfoil_data = load_foil(fpath)
    xfoil_data = run_xfoil(airfoil_data)

    results_path = Path(results_path).resolve()

    oper = load_oper_from_file('app/app_vars.json')

    row_dicts = []

    for propf in results_path.glob("*.prop"):
        propf = propf.resolve()
        print(f"Processing {propf}")

        # Load operating conditions
        prop = load_prop_from_file(propf)

        # Run BEM and get coefficients
        res = load_prop_and_run_bem_from_given_operating_and_xfoil_data(prop, oper, xfoil_data)

        res['propeller'] = propf.name

        row_dicts.append(res)

    print("Collected data for all propellers:")
    df = pd.DataFrame(row_dicts)
    
    lookup_df = parse_lookup_df('app/results')
    propfs_of_interest = lookup_df['prop_path'].unique()
    names = [propf.name for propf in propfs_of_interest]
    # filter df for propellers with 'propeller' in propfs_of_interest

    names.remove('dalprop5045_nonlinear_test2.prop')
    names.remove('dalprop5045_nonlinear_test.prop')
    names.remove('5045_s60.prop')

    df = df[df['propeller'].isin(names)]
    df = df.sort_values(by='propeller')
    df = df.reset_index(drop=True)

    # now we can plot the results on a barchart
    ax = df.plot.bar(
        x='propeller',
        y='CT',
        rot=45,
        figsize=(12, 6),
        title='Aero Coefficients for Different Propellers'
    )
    plt.gcf().tight_layout()
    ax = df.plot.bar(
        x='propeller',
        y='CQ',
        rot=45,
        figsize=(12, 6),
        title='Aero Coefficients for Different Propellers'
    )
    plt.gcf().tight_layout()
    ax = df.plot.bar(
        x='propeller',
        y='FM',
        rot=45,
        figsize=(12, 6),
        title='FM for Different Propellers'
    )
    plt.gcf().tight_layout()
    

    plt.show()


if __name__ == "__main__":

    compile_props_and_plot('app/props')

