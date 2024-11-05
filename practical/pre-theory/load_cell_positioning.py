import numpy as np
import matplotlib.pyplot as plt

def wheatstone(vex, r1, r2, r3, r4):
    vout = vex * (r3 / (r4 + r3) - r2 / (r1 + r2))
    return vout

class strain_gauge:
    def __init__(self, gf):
        self.gf = gf 

        self.strain = 0
        self.resistance = 0

        self.max_strain = 1e-3


class full_bridge_cantilever:
    # full bridge cantilever

    def __init__(self, gauges, dx, dy, E, I):
        # dx is distance from strain gauge to location of force
        # dy is distance from centroid to strain gauge
        self.gauges = gauges

        for g in self.gauges:
            g.resistance = 120

        self.force = 0
        self.moment = 0

        self.dx = dx
        self.dy = dy
        self.E = E
        self.I = I
    
    def calc_vout(self, vin):

        M = self.moment + self.force * self.dx
        # the length of strain gauge is assumed negligible compared to dx
        # this assumption isnt necessary as long as gauge center is at dx
        axial_stress = M * self.dy / self.I
        strain = (axial_stress) / self.E

        if strain > 1e-3:
            print("Warning: strain is too high")

        self.gauges[1].strain = +strain
        self.gauges[3].strain = +strain
        self.gauges[0].strain = -strain
        self.gauges[2].strain = -strain

        for g in self.gauges:
            g.resistance = 120 * (1 + g.gf * g.strain)
        
        return wheatstone(
            vin,
            self.gauges[0].resistance,
            self.gauges[1].resistance,
            self.gauges[2].resistance,
            self.gauges[3].resistance
        )
    
    def t_from_max_strain(self, max_strain, w):
        stress = max_strain * self.E
        M = self.moment + self.force * self.dx
        I = M * self.dy / stress

        # now I = w * t  * dy**2 + w * t**3 / 12
        # but lets say t small and ignore second term
        t = np.roots([w/12, 0, w * self.dy**2, - I])
        
        t = t[(t.real > 0)*(t.imag == 0)]
        if len(t) == 0:
            print("Warning: t not found")
            return None

        if t*1e2 > self.dy:
            print("Warning: t is too large")

        return t


gauges = [strain_gauge(2.1) for _ in range(4)]

h = 1e-2 # 1 cm
w = 1e-2 # 1 cm
l = 1e-1 # 10 cm

I = w * h ** 3 / 12
E = 70e9

cell = full_bridge_cantilever(gauges, 5e-2, h / 2, E, I)

N = 100
forces = np.linspace(10, 100, N)
vouts = np.zeros(N)

for i, f in enumerate(forces):
    cell.force = f
    vouts[i] = cell.calc_vout(5)

t = cell.t_from_max_strain(1e-3, w)
print(f'Thickness of points is {t * 1e3} mm')

plt.plot(forces, vouts)
plt.show()

