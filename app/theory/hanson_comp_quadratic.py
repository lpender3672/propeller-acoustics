

# this file will assume quadratic chordwise distributions
# and perform the static swirl analysis for lift and drag.

import matplotlib as mpl
import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt
from PyQt6.QtWidgets import QApplication

from scipy.integrate import (
    cumulative_trapezoid, simpson, trapezoid
)
from scipy.optimize import minimize
from scipy.special import jv as besselj



