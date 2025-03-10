
import json 
from pathlib import Path
import numpy as np

import os
import contextlib

XFOIL_INSTALLED = True
try: 
    from xfoil import XFoil
    from xfoil.model import Airfoil
except ModuleNotFoundError:
    XFOIL_INSTALLED = False
    print("Warning Xfoil not installed")


from scipy.interpolate import CubicSpline, interp1d

    
class AppVars():
    def __init__(self):
        self.oper = {
        }

        self.prop = {
        }

        self.dist = {
        }

        self.res = {
        }

        self.airfoil_data = np.zeros((0, 2)) # geometry

        self.xfoil_data = np.zeros((0, 5)) # alpha, Re, cl, cd, *section_cp_profile

    def copy(self):
        av = AppVars()
        av.oper = self.oper.copy()
        av.prop = self.prop.copy()
        av.dist = self.dist.copy()
        av.res = self.res.copy()
        av.airfoil_data = self.airfoil_data.copy()
        av.xfoil_data = self.xfoil_data.copy()
        return av

def foil_data(airfoil_data, alpha, Re, collect_cp=False):
    
    xf = XFoil()
    xf.airfoil = Airfoil(
        airfoil_data[:,0],
        airfoil_data[:,1]
        )
    #xf.M = 0.0
    xf.max_iter = 100
    xf.verbose = False

    if isinstance(alpha, (int, float)):
        cls = np.zeros(1)
        cds = np.zeros(1)
        alpha = [alpha]
    else:
        cls = np.zeros(len(alpha))
        cds = np.zeros(len(alpha))

    if isinstance(Re, (int, float)):
        Re = Re * np.ones(len(alpha))

    cps = np.zeros((alpha.shape[0], airfoil_data.shape[0]))

    for i, alf in enumerate(alpha):
        xf.Re = Re[i]

        with open(os.devnull, "w") as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                # this isnt working
                out = xf.a(alf)

        cl, cd, _, _ = out
        cls[i] = cl
        cds[i] = cd
        # plot cps against airfoil x

        if collect_cp:
            _, _, cps[i] = xf.get_cp_distribution()

    if collect_cp:
        return cls, cds, cps

    return cls, cds

def load_foil(fpath):
    raw_coords = np.loadtxt(fpath)
    nxnodes = raw_coords.shape[0] * 2 + 1
    fx, fy = sample_airfoil(raw_coords, nxnodes)

    return np.column_stack([fx, fy])


def run_xfoil(airfoil_data, Np = 10, collect_cp=True):
    Re = 5e5 * np.ones(Np)

    alphas = np.linspace(-20, 20, Np)
    if collect_cp:
        cls, cds, cps = foil_data(airfoil_data, alphas, Re, collect_cp)
        return np.column_stack((alphas, Re, cls, cds, *cps.T))

    cls, cds = foil_data(airfoil_data, alphas, Re)
    return np.column_stack((alphas, Re, cls, cds))

def correct_clcd_sweep(cl, cd, sweep):
    return cl * np.cos(sweep) ** 2, cd * np.cos(sweep) ** 2

def Viterna_extrapolation(xfoil_data, alpha, Re):
    alpha_data = xfoil_data[:, 0]
    Cl_data = xfoil_data[:, 2]
    Cd_data = xfoil_data[:, 3]
    Cl_valid = ~np.isnan(Cl_data)

    alpha_stall = alpha_data[Cl_valid][-1] * np.pi / 180
    CLstall = Cl_data[Cl_valid][-1]
    CDstall = Cd_data[Cl_valid][-1]

    AR = 10 # not important
    CDmax = 1.11 + 0.018 * AR
    a1 = CDmax / 2
    b1 = CDmax
    a2 = (CLstall - CDmax * np.sin(alpha_stall) * np.cos(alpha_stall)) * np.sin(alpha_stall) / (np.cos(alpha_stall) ** 2)
    b2 = (CDstall - CDmax * np.sin(alpha_stall) ** 2) / np.cos(alpha_stall)

    cl = a1 * np.sin(2 * alpha) + a2 * np.cos(alpha) ** 2 / np.sin(alpha)
    cd = b1 * np.sin(alpha) ** 2 + b2 * np.cos(alpha)

    return cl, cd

def interpolate_clcd(xfoil_data, alpha, Re):
    if XFOIL_INSTALLED and xfoil_data.shape[1] >= 4:
            
        alpha_data = xfoil_data[:, 0]
        Cl_data = xfoil_data[:, 2]
        Cd_data = xfoil_data[:, 3]
        Cl_valid = ~np.isnan(Cl_data)
        Cd_valid = ~np.isnan(Cd_data)

        if isinstance(alpha, float):


            if alpha > np.max(alpha_data[Cl_valid]):
                Cl, Cd = Viterna_extrapolation(xfoil_data, alpha, Re)
            else:
                Cl = np.interp(alpha * 180/np.pi, alpha_data[Cl_valid], Cl_data[Cl_valid])
                Cd = np.interp(alpha * 180/np.pi, alpha_data[Cd_valid], Cd_data[Cd_valid])
        
        else:
            #mask = alpha*180/np.pi > np.max(alpha_data[Cl_valid])
            mask = alpha*180/np.pi > alpha_data[Cl_valid][-1]
            Cl = np.interp(alpha * 180/np.pi, alpha_data[Cl_valid], Cl_data[Cl_valid])
            Cd = np.interp(alpha * 180/np.pi, alpha_data[Cd_valid], Cd_data[Cd_valid])
            Cl[mask], Cd[mask] = Viterna_extrapolation(xfoil_data, alpha[mask], Re)

    else:
        Cl = 2 * np.pi * alpha
        Cd = 0.0087 - 0.021 * alpha + 0.400 * alpha ** 2
    
    return Cl, Cd


def fit_quadratic(x, y):
    A = np.array([
        [x[0]**2, x[0], 1],
        [x[1]**2, x[1], 1],
        [x[2]**2, x[2], 1]
    ])
    b = np.array(y)
    coeffs = np.linalg.solve(A, b)
    return coeffs

def calc_distribution(dist_index, ctrl_pts, x_dist):
    ctrl_pts = np.array(ctrl_pts)
    x = ctrl_pts[:, 0]
    y = ctrl_pts[:, 1]

    if dist_index == 0:
        b = (y[1] - y[0]) / (x[1] - x[0])
        a = y[0] - b * x[0]
        y_dist = a + b * x_dist
    elif dist_index == 1:
        coefficients = fit_quadratic(x, y)
        y_dist = coefficients[0] * x_dist**2 + coefficients[1] * x_dist + coefficients[2]
    elif dist_index == 2:
        idx = np.argsort(x)
        spline = CubicSpline(x[idx], y[idx])
        y_dist = spline(x_dist)
    elif dist_index == 3:
        # custom
        #if x_dist.shape != self.xb.shape:
        #    self.yb = np.interp(x_dist, self.xb, self.yb)
        #    self.xb = x_dist
        raise IndexError("Custom distribution cannot be calculated")
    elif dist_index == 4:
        # y = a/x + b
        a = (y[0] - y[1]) / (1/x[0] - 1/x[1])
        b = y[0] - a / x[0]
        y_dist = a / x_dist + b
    elif dist_index == 5:
        # y = k * arctan(1/x)
        k = y[0] / np.arctan(1/x[0])
        y_dist = k * np.arctan(1/x_dist)
    else:
        y_dist = np.zeros_like(x_dist)
        raise IndexError("Unknown distribution index")

    return y_dist


def load_prop_from_file(file_path, include_dists=False):
    
    with open(file_path, 'r') as propj:
        propf = json.load(propj)
        prop = propf['prop']
        dists = propf['dist']

    xc_pts = np.linspace(-0.5,0.5,prop['nx']+1)
    if prop['rdist'] == "Linear":
        rarr_pts = np.linspace(prop['rh'], prop['rt'], prop['nr']+1)
    elif prop['rdist'] == "Cosine":
        rarr_pts = prop['rh'] + (prop['rt']-prop['rh'])/2 * (1 - np.cos(np.linspace(0,np.pi,prop['nr']+1)))

    prop['xc']  = (xc_pts[1:] + xc_pts[:-1]) / 2
    prop['r0'] = (rarr_pts[1:] + rarr_pts[:-1]) / 2
    prop['r0_rt'] = prop['r0'] / prop['rt']
    prop['dz'] = np.diff(rarr_pts)

    distypes = [
            "linear",
            "quadratic",
            "spline",
            "custom",
            "inverse",
            "arctan"
        ]

    prop['c'] = calc_distribution(
        distypes.index(dists['CTL_c_type']), dists['CTL_c'], prop['r0_rt']
    ) * prop['c75']
    prop['twist'] = calc_distribution(
        distypes.index(dists['CTL_twist_type']), dists['CTL_twist'], prop['r0_rt']
    ) * np.pi / 180
    prop['sweep'] = calc_distribution(
        distypes.index(dists['CTL_sweep_type']), dists['CTL_sweep'], prop['r0_rt']
    ) * np.pi / 180
    
    if include_dists:
        return prop, dists
    return prop

def save_custom_prop_to_file(file_path, prop):
    dists = {
        'CTL_c_type': 'custom',
        'CTL_c': np.zeros((1,2)),
        'CTL_twist_type': 'custom',
        'CTL_twist': np.zeros((1,2)),
        'CTL_sweep_type': 'custom',
        'CTL_sweep': np.zeros((1,2)),
    }
    save_prop_to_file(file_path, prop, dists)

def save_prop_to_file(file_path, prop, dists):

    propc = prop.copy()
    distsc = dists.copy()

    for key,elem in propc.items():
        if isinstance(elem, np.ndarray):
            distsc[key] = list(elem)

    for key,elem in distsc.items():
        if isinstance(elem, np.ndarray):
            distsc[key] = list(elem)
    
    propf = {
        'prop': propc,
        'dist': distsc
    }
    with open(file_path, 'w') as propj:
        json.dump(propf, propj, indent=4)

def load_oper_from_file(file_path):
    
    with open(file_path, 'r') as operj:
        oper = json.load(operj)['oper']

    return oper

def append_audiof_to_metaf(audiof, metaf, row):

    try:
        metadata = np.load(metaf)
    except FileNotFoundError:
        metadata = np.zeros((0, 2))

    newrow = np.array([audiof, row])
    metadata = np.vstack((metadata, newrow))
    np.save(metaf, metadata)

def _separate(arr): # NODES
    npts = arr.shape[0] // 2
    up_var = np.flip(arr[:npts], axis=0)
    low_var = arr[npts+1:]
    return up_var, low_var

def _separate_panels(arr):
    npx = arr.shape[0] // 2
    up_var = np.flip(arr[:npx], axis=0)
    low_var = arr[npx:]
    return up_var, low_var

def upscale_airfoil_geometry(airfoil_data, n_panels):

    newidx = np.linspace(0, 1, n_panels + 1)
    oldidx = np.linspace(0, 1, airfoil_data.shape[0])

    newx = np.interp(newidx, oldidx, airfoil_data[:,0])
    newz = np.interp(newidx, oldidx, airfoil_data[:,1])

    return np.column_stack((newx, newz))

def sample_airfoil(airfoil_data, nxnodes = None):

    xs = airfoil_data[:, 0]
    ys = airfoil_data[:, 1]

    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.sqrt(dx**2 + dy**2)
    sarr = np.concatenate(([0], np.cumsum(ds)))  # Arc length array

    f_x = interp1d(sarr, xs, kind='cubic', fill_value="extrapolate")
    f_y = interp1d(sarr, ys, kind='cubic', fill_value="extrapolate")

    if nxnodes is None:
        nxnodes = xs.shape[0]

    s_uniform = np.linspace(0, sarr[-1], nxnodes)

    xs_resampled = f_x(s_uniform)
    ys_resampled = f_y(s_uniform)

    return xs_resampled, ys_resampled

def calc_chordwise_loading(airfoil_data, xfoil_data):

    xs = airfoil_data[:, 0]  # x-coordinates of the panels
    ys = airfoil_data[:, 1]  # y-coordinates of the panels
    alphas = xfoil_data[:, 0] * np.pi / 180  # AoA in radians
    cps = xfoil_data[:, 4:]  # Pressure coefficient data for each AoA case

    # Compute panel properties
    dx = np.diff(xs)  # x-component of panel vectors
    dy = np.diff(ys)  # y-component of panel vectors
    ds = np.sqrt(dx**2 + dy**2)  # Panel lengths

    # Unit normal vectors (perpendicular to panels in airfoil frame)
    nx = dy / ds  # Normal vector x-component
    ny = -dx / ds  # Normal vector y-component

    # Initialize arrays for lift and drag distributions
    npx = (cps.shape[1] - 1) // 2
    cl_x_alpha = np.zeros((npx, len(alphas)))  # Chordwise lift loading
    cd_x_alpha = np.zeros((npx, len(alphas)))  # Chordwise drag loading

    # Loop over AoA cases
    for i in range(cps.shape[0]):
        cp = cps[i, :]  # Pressure coefficients for this AoA
        cp_mid = 0.5 * (cp[:-1] + cp[1:])  # Panel midpoint pressures (average)

        # Compute force components in the airfoil frame
        fx = -cp_mid * nx * ds  # Force in the x-direction
        fy = -cp_mid * ny * ds  # Force in the y-direction

        # Rotate forces by AoA to compute lift and drag
        alpha = alphas[i]
        cl = fy * np.cos(alpha) - fx * np.sin(alpha)  # Lift
        cd = fx * np.cos(alpha) + fy * np.sin(alpha)  # Drag

        cl_u, cl_l = _separate_panels(cl)
        cd_u, cd_l = _separate_panels(cd)

        cl_x_alpha[:, i] = cl_u + cl_l
        cd_x_alpha[:, i] = cd_u + cd_l

    return cl_x_alpha, cd_x_alpha

def set_intergrands(prop):
    
    xc_pts = np.linspace(-0.5,0.5,prop['nx']+1)
    if prop['rdist'] == "Linear":
        rarr_pts = np.linspace(prop['rh'], prop['rt'], prop['nr']+1)
    elif prop['rdist'] == "Cosine":
        rarr_pts = prop['rh'] + (prop['rt']-prop['rh'])/2 * (1 - np.cos(np.linspace(0,np.pi,prop['nr']+1)))

    prop['xc']  = (xc_pts[1:] + xc_pts[:-1]) / 2
    prop['r0'] = (rarr_pts[1:] + rarr_pts[:-1]) / 2
    prop['r0_rt'] = prop['r0'] / prop['rt']
    prop['dz'] = np.diff(rarr_pts)

if __name__ == "__main__":
    raw = load_foil("app/foils/naca0012.surf")
    airfoil_data = upscale_airfoil_geometry(raw, 100)
    airfoil_data = run_xfoil(airfoil_data, collect_cp=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)

    xs_u, xs_l = _separate(airfoil_data[:,0])

    cl_x_alpha, cd_x_alpha = calc_chordwise_loading(airfoil_data)

    plt.plot(xs_u, cl_x_alpha[:,0])

    plt.show()
