
import json 
from pathlib import Path
import numpy as np

XFOIL_INSTALLED = True
try: 
    from xfoil import XFoil
    from xfoil.model import Airfoil
except ModuleNotFoundError:
    XFOIL_INSTALLED = False
    print("Warning Xfoil not installed")


from scipy.interpolate import CubicSpline

    
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

    def copy(self):
        av = AppVars()
        av.oper = self.oper.copy()
        av.prop = self.prop.copy()
        av.dist = self.dist.copy()
        av.res = self.res.copy()
        av.airfoil_data = self.airfoil_data.copy()
        return av

def foil_data(airfoil_data, alpha, Re, collect_cp=False):
    
    xf = XFoil()
    xf.airfoil = Airfoil(
        airfoil_data[:,0],
        airfoil_data[:,1]
        )
    xf.Re = Re
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

    cps = np.zeros((alpha.shape[0], airfoil_data.shape[0]))

    for i, alf in enumerate(alpha):
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

def run_xfoil(airfoil_data, collect_cp=True):
    Re = 5e5

    alphas = np.linspace(-20, 20, airfoil_data.shape[0])
    if collect_cp:
        cls, cds, cps = foil_data(airfoil_data, alphas, Re, collect_cp)
        return np.column_stack((airfoil_data, alphas, cls, cds, *cps))

    cls, cds = foil_data(airfoil_data, alphas, Re)
    return np.column_stack((airfoil_data, alphas, cls, cds))

def correct_clcd_sweep(cl, cd, sweep):
    return cl * np.cos(sweep) ** 2, cd * np.cos(sweep) ** 2

def Viterna_extrapolation(airfoil_data, alpha, Re):
    alpha_data = airfoil_data[:, 2]
    Cl_data = airfoil_data[:, 3]
    Cd_data = airfoil_data[:, 4]
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

def interpolate_clcd(airfoil_data, alpha, Re):
    if XFOIL_INSTALLED and airfoil_data.shape[1] >= 5:
            
        alpha_data = airfoil_data[:, 2]
        Cl_data = airfoil_data[:, 3]
        Cd_data = airfoil_data[:, 4]
        Cl_valid = ~np.isnan(Cl_data)
        Cd_valid = ~np.isnan(Cd_data)

        if isinstance(alpha, float):


            if alpha > np.max(alpha_data[Cl_valid]):
                Cl, Cd = Viterna_extrapolation(airfoil_data, alpha, Re)
            else:
                Cl = np.interp(alpha * 180/np.pi, alpha_data[Cl_valid], Cl_data[Cl_valid])
                Cd = np.interp(alpha * 180/np.pi, alpha_data[Cd_valid], Cd_data[Cd_valid])
        
        else:
            #mask = alpha*180/np.pi > np.max(alpha_data[Cl_valid])
            mask = alpha*180/np.pi > alpha_data[Cl_valid][-1]
            Cl = np.interp(alpha * 180/np.pi, alpha_data[Cl_valid], Cl_data[Cl_valid])
            Cd = np.interp(alpha * 180/np.pi, alpha_data[Cd_valid], Cd_data[Cd_valid])
            Cl[mask], Cd[mask] = Viterna_extrapolation(airfoil_data, alpha[mask], Re)

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


def load_prop_from_file(file_path):
    
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
    propf = {
        'prop': prop,
        'dist': dists
    }
    with open(file_path, 'w') as propj:
        json.dump(propf, propj, indent=4)

def load_oper_from_file(file_path):
    
    with open(file_path, 'r') as operj:
        oper = json.load(operj)['oper']

    return oper


def _separate(arr):
    npts = arr.shape[0] // 2
    up_var = np.flip(arr[:npts], axis=0)
    low_var = arr[npts+1:]
    return up_var, low_var

def upscale_airfoil_geometry(airfoil_data, n_panels):

    newidx = np.linspace(0, 1, n_panels + 1)
    oldidx = np.linspace(0, 1, airfoil_data.shape[0])

    newx = np.interp(newidx, oldidx, airfoil_data[:,0])
    newz = np.interp(newidx, oldidx, airfoil_data[:,1])

    return np.column_stack((newx, newz))

def calc_chordwise_loading(airfoil_data):

    # airfoil_data[:,n]
    # n = 0 -> x
    # n = 1 -> z
    # n = 2 -> alpha
    # n = 3 -> cl
    # n = 4 -> cd
    # n = 5 -> cp at alpha[0]
    # n = 6 -> cp at alpha[1]
    # etc

    xs_u, xs_l = _separate(airfoil_data[:,0])
    ys_u, ys_l = _separate(airfoil_data[:,1])

    cl_x_alpha = np.zeros((airfoil_data.shape[0] // 2 - 1, airfoil_data.shape[0]))
    cd_x_alpha = np.zeros((airfoil_data.shape[0] // 2 - 1, airfoil_data.shape[0]))

    for i,n in enumerate(range(5, airfoil_data.shape[1])):

        cpup, cplo = _separate(airfoil_data[:,n])

        alpha = airfoil_data[i,2] * np.pi / 180

        dtheta_u = np.arctan2(np.diff(ys_u), np.diff(xs_u))
        dtheta_l = np.arctan2(np.diff(ys_l), np.diff(xs_l))

        ds_u = np.sqrt(np.diff(xs_u)**2 + np.diff(ys_u)**2)
        ds_l = np.sqrt(np.diff(xs_l)**2 + np.diff(ys_l)**2)

        # the panel lengths are not included because currently they are
        # very discontinuous, which causes false discontinuities in the loading
        cl_upper = -cpup[:-1] * np.cos(dtheta_u - alpha) #* ds_u
        cl_lower = -cplo[:-1] * np.cos(dtheta_l - alpha) #* ds_l

        cd_upper = -cpup[:-1] * np.sin(dtheta_u - alpha) #* ds_u
        cd_lower = -cplo[:-1] * np.sin(dtheta_l - alpha) #* ds_l

        cl_x_alpha[:, i] = cl_lower - cl_upper
        cd_x_alpha[:, i] = cd_lower + cd_upper

    ## insert 0 at the start of the array
    cl_x_alpha = np.insert(cl_x_alpha, 0, 0, axis=0)
    cd_x_alpha = np.insert(cd_x_alpha, 0, 0, axis=0)

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
    raw = np.loadtxt("app/foils/naca0012.surf")
    airfoil_data = upscale_airfoil_geometry(raw, 100)
    airfoil_data = run_xfoil(airfoil_data, collect_cp=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)

    xs_u, xs_l = _separate(airfoil_data[:,0])

    cl_x_alpha, cd_x_alpha = calc_chordwise_loading(airfoil_data)

    ax[0].plot(xs_u[:-1], cl_x_alpha[:,::20])
    ax[0].legend()

    plt.show()
