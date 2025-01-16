
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

def foil_data(airfoil_data, alpha, Re):
    
    xf = XFoil()
    xf.airfoil = Airfoil(
        airfoil_data[:,0],
        airfoil_data[:,1]
        )
    xf.Re = Re
    xf.M = 0.0
    xf.max_iter = 100
    xf.verbose = False

    if isinstance(alpha, float):
        cls = np.zeros(1)
        cds = np.zeros(1)
        alpha = [alpha]
    else:
        cls = np.zeros(len(alpha))
        cds = np.zeros(len(alpha))

    for i, a in enumerate(alpha):
        out = xf.a(a)
        cl, cd, _, _ = out
        cls[i] = cl
        cds[i] = cd

    return cls, cds

def run_xfoil(airfoil_data):

    alphas = np.linspace(-20, 20, airfoil_data.shape[0])
    cls, cds = foil_data(airfoil_data, alphas, 1e6)
    return np.column_stack((airfoil_data, alphas, cls, cds))

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

def load_oper_from_file(file_path):
    
    with open(file_path, 'r') as operj:
        oper = json.load(operj)['oper']

    return oper
    