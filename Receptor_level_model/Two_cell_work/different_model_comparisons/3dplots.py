import numpy as np
from scipy.integrate import solve_ivp
import math
from scipy.constants import N_A
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

kon = 1e5
koff = 1e-4
Target_cell_number = 5e3
rtot_t = 1e5
tumour_cell_radius = 8e-6
tumour_cell_surface_area = 4*math.pi*((tumour_cell_radius)**2)
r_ab = 1.25e-8
reaction_volume = (2/3)*math.pi*((r_ab)**3)
target_effective_conc =(rtot_t)*(tumour_cell_surface_area)*(1/N_A)/(reaction_volume)

k2 = target_effective_conc*kon

def dimensionless_model(t, z, A0, rtot_t, kon):
    k = kon*A0
    alpha = k/koff
    Atot = 1e16*A0/Target_cell_number
    beta = Atot/rtot_t
    target_effective_conc =(rtot_t)*(tumour_cell_surface_area)*(1/N_A)/(reaction_volume)
    k2 = target_effective_conc*kon
    alpha2 = k2/koff

    dA1 = 2*alpha*(1-z[0]-2*z[1])*(beta-z[0]-z[1]) - z[0] - alpha2*(1-z[0]-2*z[1])*z[0] + 2*z[1]
    dA2 = alpha2*(1-z[0]-2*z[1])*z[0] - 2*z[1]

    return [dA1, dA2]

vals = np.linspace(1, 9, 40)
vals1 = np.linspace(1, 9, 15)
rtots1 = [1e3, 1e4, 1e5]
rtots = []
A0s1 = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
A0s = []
t_end = 1000
t = np.geomspace(1e-10, t_end, 500)
tspan = [1e-10, t_end]
z0 = [0, 0]

for x in A0s1:
    for val in vals1:
        A0s.append(x*val)

for x in rtots1:
    for val in vals:
        rtots.append(x*val)


Y1, Y2 = np.meshgrid(A0s, rtots)
rvals = np.zeros(Y1.shape)

Ni, Nj = Y1.shape

for i in range(Ni):
    for j in range(Nj):
        A0_val = Y1[i, j]
        rtot_val = Y2[i, j]
        z = solve_ivp(dimensionless_model, tspan, z0, method='Radau', t_eval=t, args=(A0_val, rtot_val, kon))
        A1 = z.y[0]
        A2 = z.y[1]
        r = 1 - (1 - A1 - 2*A2)
        rvals[i, j] = r[-1]

"""
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(np.log10(A0s), np.log10(rtots), rvals, cmap='viridis')
plt.show()
"""
print(len(np.log10(A0s)), len(np.log10(rtots)))
import seaborn as sns
sns.heatmap(np.log10(A0s), np.log10(rtots), rvals)
plt.show()