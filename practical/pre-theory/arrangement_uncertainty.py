
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

global k, c
k = sp.symbols('k')
c = sp.symbols('c')

def f(F):
    #return k * F + c
    return 1 * F + 0

d = sp.symbols('d')

v2 = sp.symbols('v2')
v1 = sp.symbols('v1')
Fplus = sp.symbols('F_+')
Fminus = sp.symbols('F_-')
theta = sp.symbols('theta')

R1 = (v1 - f(Fminus * sp.sin(theta))) / (f(Fplus*sp.sin(theta)) + f(Fminus * sp.sin(theta))) * (Fplus - Fminus) * sp.sin(theta) - Fminus * sp.sin(theta)
R2 = (v2 - f(Fminus * sp.cos(theta))) / (f(Fplus*sp.cos(theta)) + f(Fminus * sp.cos(theta))) * (Fplus - Fminus) * sp.cos(theta) - Fminus * sp.cos(theta)

M = d * (R1 * sp.cos(theta) - R2 * sp.sin(theta))
F = R1 * sp.sin(theta) + R2 * sp.cos(theta)

# now define uncertainties
uv1 = sp.symbols('u(v1)')
uv2 = sp.symbols('u(v2)')
uFplus = sp.symbols('u(F)')
uFminus = sp.symbols('u(F)')
ud = sp.symbols('u(d)')

uM = sp.sqrt(
    (sp.diff(M, v1) * uv1)**2 +
    (sp.diff(M, v2) * uv2)**2 +
    (sp.diff(M, Fplus) * uFplus)**2 +
    (sp.diff(M, Fminus) * uFminus)**2 +
    (sp.diff(M, d) * ud)**2
)

uF = sp.sqrt(
    (sp.diff(F, v1) * uv1)**2 +
    (sp.diff(F, v2) * uv2)**2 +
    (sp.diff(F, Fplus) * uFplus)**2 +
    (sp.diff(F, Fminus) * uFminus)**2 +
    (sp.diff(F, d) * ud)**2
)

# simplify and latex

print(sp.latex(uM))
print(sp.latex(uF))

# sub in some base values
uM = uM.subs({v1: 5, v2: 5, Fplus: 10, Fminus: 1, d: 0.05})
uM = uM.subs({uv1: 0.01, uv2: 0.01, uFplus: 0.01, uFminus: 0.01, ud: 0.01})

uF = uF.subs({v1: 5, v2: 5, Fplus: 10, Fminus: 1, d: 0.05})
uF = uF.subs({uv1: 0.01, uv2: 0.01, uFplus: 0.01, uFminus: 0.01, ud: 0.01})

# now find the value of theta where value of uM + uF is minimum
func = uM + uF
min_thetas = sp.solve(sp.diff(func, theta), theta)

# plot theta

lin_theta = np.linspace(- np.pi / 2, np.pi / 2, 200)
# plot uM and uF
uMs = []
uFs = []

uM_t_values = [uM.subs({theta: float(val)}) for val in lin_theta]
uF_t_values = [uF.subs({theta: float(val)}) for val in lin_theta]


min_uTotal = []
for t in min_thetas:
    min_uTotal.append(uF.subs({theta: t}))

argmin = np.argmin(min_uTotal)
theta_min = min_thetas[argmin] * 180 / np.pi
min_u = min_uTotal[argmin]

print(f"Minimum value of uF is {min_u} at theta = {theta_min}")

print(min_thetas)
print(min_uTotal)

plt.plot(lin_theta, uM_t_values, label='uM')
plt.plot(lin_theta, uF_t_values, label='uF')

plt.ylabel('Added uncertainty in M and F')
plt.xlabel('Theta (radians)')

plt.grid()
plt.legend()
plt.show()

