import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.constants import N_A
from scipy import linalg
import math, cmath

Target_cell_number = 2e5
well_size = 150e-6
t_end = 60*60*10*10
t = np.geomspace(1e-10, t_end, 150)
tspan = [1e-10, t_end]
z0 = [0, 0]

def model_S_monospecific(t, z, Ainit, rtot, kon, k2, koff):
     k = Ainit*kon
     Atot = well_size*N_A*Ainit/Target_cell_number
     A0 = Atot - z[0] - z[1] 
     Ag = rtot - z[0] - 2*z[1]

     dA10 = 2*(k*Ag*A0/rtot) - koff*z[0] - (k2*Ag*z[0]/rtot) + 2*koff*z[1]
     dA11 = (k2*Ag*z[0]/rtot) - 2*koff*z[1]

     return [dA10, dA11]

def model_S_dimensionless(t, z, alpha1, alpha2, beta):

     A0 = beta - z[0] - z[1] 
     Ag = 1 - z[0] - 2*z[1]

     dA10 = 2*(alpha1*Ag*A0) - z[0] - (alpha2*Ag*z[0]) + 2*z[1]
     dA11 = (alpha2*Ag*z[0]) - 2*z[1]

     return [dA10, dA11]

def dxdt(x, y, alpha1, alpha2, beta):
     return 2*alpha1*(1-x-2*y)*(beta-x-y) -x - alpha2*x*(1-x-2*y) + 2*y

def dydt(x, y, alpha2):
     return  alpha2*x*(1-x-2*y) - 2*y

A0s = np.geomspace(1e-12, 1e-5, 50)
alpha1s = np.geomspace(1e-3, 1e5, len(A0s))
alpha2s = np.geomspace(1, 1e6, len(A0s))
rtot = 1e5
kon = 1e5

koff=1e-4

Y = np.zeros((len(A0s), len(alpha1s), len(alpha2s)))

for i, Ainit in enumerate(A0s):
    print(i)
    for k, alpha1 in enumerate(alpha1s):
        for j, alpha2 in enumerate(alpha2s):
            Atot = well_size*N_A*Ainit/Target_cell_number
            beta = Atot/rtot
            z = solve_ivp(model_S_dimensionless, t_span=tspan, y0=[0,0], method='Radau', t_eval=t, args=(alpha1, alpha2, beta))
            A1 = z.y[0]
            A2 = z.y[1]
            Y[j, k, i] = A1[-1]

from  matplotlib.widgets import Slider, Button
import PyQt5


def f(value):
    
    for i, val in enumerate(alpha1s):
        if val == value:
            arg = i
    array = Y[arg][:][:]
    return array


fig, ax = plt.subplots(figsize=(8, 6))
line = sns.heatmap(f(alpha1s[0]), xticklabels=True, yticklabels=True, ax=ax) 
fig.subplots_adjust(left=0.25, bottom = 0.25)
#ax = sns.heatmap(Y[0,:,:], xticklabels=True, yticklabels=True, ax=ax)
ax.set_xticks([0, 7, 15, 21, 29, 35, 42, 49])
ax.set_yticks([0, 8, 16, 25, 33, 41, 49])
ax.set_yticklabels([1, 10, r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'])
ax.set_xticklabels([r'$10^{-12}$', r'$10^{-11}$', r'$10^{-10}$', r'$10^{-9}$', r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$'])
ax.set_xlabel(r'$A_{init}$' + ' (M)')
ax.set_ylabel(r'$\alpha_2$')
ax.set_title(r'$A_1$' + ' steady state heatmap')

axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
a1_slider = Slider(
    ax=axfreq,
    label=r'$\alpha_1$',
    valmin=alpha1s[0],
    valmax= alpha1s[-1],
    valinit = alpha1s[49]
)

def update(val):
    line.set_ydata(f(a1_slider.val))
    fig.canvas.draw_idle()

a1_slider.on_changed(update)

fig.canvas.draw()