import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math
from scipy.constants import N_A

Target_cell_number = 2e5
well_size = 125e-6
t_end = 60*60*1
t = np.geomspace(1e-10, t_end, 150)
t_span = [1e-10, t_end]
z0 = [0, 0]
tumour_cell_radius = 8e-6
tumour_cell_surface_area = 4*math.pi*((tumour_cell_radius)**2)

def model_S_monospecific(t, z, Ainit, rtot, kon, k2, koff):
     k = Ainit*kon
     Atot = well_size*N_A*Ainit/Target_cell_number
     A0 = Atot - z[0] - z[1] 
     Ag = rtot - z[0] - 2*z[1]

     dA10 = 2*(k*Ag*A0/Atot) - koff*z[0] - (k2*Ag*z[0]) + 2*koff*z[1]
     dA11 = (k2*Ag*z[0]) - 2*koff*z[1]

     return [dA10, dA11]

def monovalent_binding(t, z, Ainit, rtot, kon, koff):
     k = Ainit*kon
     Atot = well_size*N_A*Ainit/Target_cell_number
     A0 = Atot - z
     Ag = rtot - z

     dA1 = k*Ag*A0/Atot - koff*z

     return dA1

A0s = np.geomspace(1e-12, 1e-6, 1000)

# Desired KDs: 1nm 10nm 100nm 1000nm
kons = [10**(4.5), 10**4, 10**(3.5), 10**3]
koffs = [10**(-4.5), 10**(-4), 10**(-3.5), 10**(-3)]

bivalent = np.zeros((len(kons), len(A0s)))
monovalent = np.zeros((len(kons), len(A0s)))

D = 1e-14
k2 = 4*D/tumour_cell_surface_area
rtot = 1e3


for j, kon in enumerate(kons):
    koff = koffs[j]
    for i, Ainit in enumerate(A0s):

        # simulate monovalent binding

        z = solve_ivp(monovalent_binding, t_span, [0], method='Radau', t_eval=t, args=(Ainit, rtot, kon, koff)) 
        bound_ab = z.y[0]
        monovalent[j, i] = bound_ab[-1]

        # simulate bivalent binding
    
        z = solve_ivp(model_S_monospecific, t_span, z0, method='Radau', t_eval=t, args=(Ainit, rtot, kon, k2, koff))
        A1 = z.y[0]
        A2 = z.y[1]
        bivalent[j, i] = A1[-1] + 2*A2[-1]
    
sns.set_context('talk')
fig, ax = plt.subplots(1, 4)
ax[0].semilogx(A0s, monovalent[0, :], c='red', label='monovalent')
ax[0].semilogx(A0s, bivalent[0, :], c='blue', label = 'bivalent')
ax[1].semilogx(A0s, monovalent[1, :], c='red', label='monovalent')
ax[1].semilogx(A0s, bivalent[1, :], c='blue', label = 'bivalent')
ax[2].semilogx(A0s, monovalent[2, :], c='red', label='monovalent')
ax[2].semilogx(A0s, bivalent[2, :], c='blue', label = 'bivalent')
ax[3].semilogx(A0s, monovalent[3, :], c='red', label='monovalent')
ax[3].semilogx(A0s, bivalent[3, :], c='blue', label = 'bivalent')
plt.show()
