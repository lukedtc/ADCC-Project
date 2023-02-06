import numpy as np
from scipy.integrate import solve_ivp
from SALib.sample import saltelli
from SALib.analyze import sobol
import seaborn as sns
import matplotlib.pyplot as plt

# base params
Target_cell_number = 5e3
E_T_ratio = 10
t_end = 100
t = np.geomspace(1e-8, t_end, 500)
tspan = [0, t_end]
z0 = [0, 0]
t_end1 = 1
t1 = np.geomspace(1e-10, t_end1, 100)
t_span1 = [0, t_end1]
tend2 = 1000
t2 = np.geomspace(1e-10, tend2, 100)
t_span2 = [1e-10, tend2]

# tumour cell binding model
def tumour_cell_binding(t, z, A0, delta, rtot, kon, koff):

    Ainit = A0
    k = Ainit*kon
    k1 = 2*k
    k1off = koff
    alpha1 = k1/k1off
    Atot = 1e16*Ainit/Target_cell_number
    beta_t = Atot/rtot

    dA1 = alpha1*(1-z[0]-2*z[1])*(beta_t-z[0]-z[1]) - z[0] - delta*alpha1*(1-z[0]-2*z[1])*z[0] + 2*z[1]
    dA2 = delta*alpha1*(1-z[0]-2*z[1])*z[0] - 2*z[1]

    return [dA1, dA2]

def NK_cell_binding(t, z, A0, qon, koff, qoff, rtot_f, Atot_system):
    q = A0*qon
    sigma = q/koff
    epsilon = qoff/koff
    Atot_effector = Atot_system/E_T_ratio
    eta = Atot_effector/rtot_f

    return sigma*(eta-z)*(1-z) - epsilon*z

# extract tumour cell antibody levels

def tumour_cell_stst(A0, delta, rtot, kon, koff):

    z = solve_ivp(tumour_cell_binding, tspan, z0, method='Radau', args=(A0, delta, rtot, kon, koff), t_eval=t)

    A1 = z.y[0]
    A2 = z.y[1]
    A1_stst = A1[-1]
    A2_stst = A2[-1]

    return [A1_stst, A2_stst]

def NK_cell_stst(A0, qon, koff, qoff, rtot_f, rtot_t, A1_stst, A2_stst):

    Atot = 1e16*A0/Target_cell_number
    beta = Atot/rtot_t

    A0 = beta - A1_stst - A2_stst
    A0_IC = A0*Atot

    z2 = solve_ivp(NK_cell_binding, t_span1, [0], method = 'Radau', t_eval=t1, args=(A0, qon, koff, qoff, rtot_f, A0_IC))

    A01_IC = z2.y[0]

    return A01_IC[-1]

# synapse model
def synapse_model(t, z, A0, delta2, rtot_t, kon, koff, rtot_f, qon, qoff, delta4, delta7, A10_0, A20_0, A01_0):

    Ainit = A0
    k = Ainit*kon
    q = qon*Ainit
    delta5 = delta4*delta7/delta2
    delta6 = delta4
    k1 = 2*k
    k1off = koff
    k2 = delta2*k
    k2off = 2*koff
    k3 = q
    k3off = qoff
    k4 = delta4*q
    k4off = qoff
    k5 = delta5*q
    k5off = qoff
    k6 = 2*delta6*k
    k6off = koff
    k7 = delta7*k
    k7off = 2*koff

    alpha1 = k1/k1off
    alpha2 = k2/k1off
    gamma2 = k2off/k1off
    alpha3 = k3/k1off
    gamma3 = k3off/k1off
    alpha4 = k4/k1off
    gamma4 = k4off/k1off
    alpha5 = k5/k1off
    gamma5 = k5off/k1off
    alpha6 = k6/k1off
    gamma6 = k6off/k1off
    alpha7 = k7/k1off
    gamma7 = k7off/k1off

    beta_synapse = 1e1*Ainit
    phi = rtot_t/rtot_f
    beta_t = beta_synapse + A10_0 + A20_0 + (A01_0/phi)
    beta_f = beta_t*phi

    A00 = 1 - (1/beta_t)*(z[0] + z[1] + z[3] + z[4]) - (1/beta_f)*z[2]
    rt = 1- z[0] - z[3] - 2*(z[1] + z[4])
    rf = 1 - z[2] - phi*(z[3] + z[4])
    
    dA10 = alpha1*beta_t*(A00)*(rt) - z[0] - alpha2*z[0]*(rt) + gamma2*z[1] - alpha4*z[0]*(rf) + gamma4*z[3]
    dA20 = alpha2*z[0]*(rt) - gamma2*z[1] - alpha5*z[1]*(rf) + gamma5*z[4]
    dA01 = alpha3*beta_f*(A00)*(rf) - gamma3*z[2] - alpha6*z[2]*(rt) + gamma6*phi*z[3]
    dA11 = alpha4*z[0]*(rf) - gamma4*z[3] + (alpha6/phi)*z[2]*(rt) - gamma6*z[3] - alpha7*z[3]*(rt) + gamma7*z[4]
    dA21 = alpha5*z[1]*(rf) - gamma5*z[4] + alpha7*z[3]*(rt) - gamma7*z[4]
    dz = [dA10, dA20, dA01, dA11, dA21]

    return dz

def calc_fc(A0, delta2, rtot_t, kon, koff, rtot_f, qon, qoff, delta4, delta7):

    # calculate tumour bound antibody levels
    ICS = tumour_cell_stst(A0, delta2, rtot_t, kon, koff)

    # calculate NK cell bound antibody levels
    A01_IC = NK_cell_stst(A0, qon, koff, qoff, rtot_f, rtot_t, ICS[0], ICS[1])
    z02 = [ICS[0], ICS[1], A01_IC, 0, 0]
    
    # solve synapse model with ICS from previous calculations
    z = solve_ivp(synapse_model, t_span2, z02, method='Radau', t_eval=t2, args=(A0, delta2, rtot_t, kon, koff, rtot_f, qon, qoff, delta4, delta7, ICS[0], ICS[1], A01_IC))

    A11 = z.y[3]
    A21 = z.y[4]
    fc = np.max(A11 + A21)

    return fc*rtot_t

# sobol sensitivity analysis

problem = {
    'num_vars': 9,
    'names': ['delta2', 'rtot_t', 'kon', 'koff', 'rtot_f', 'qon', 'qoff', 'delta4', 'delta7'],
    'bounds': [[0.1, 10],
               [5e4, 1e6],
               [1e3, 1e7],
               [5e-5, 5e-4],
               [4e4, 3e5],
               [8e3, 1e5],
               [5e-3, 1e-1],
               [0.1, 10],
               [0.1, 10]]
}

# generate samples

lost = []
vals = saltelli.sample(problem, 2048)

Y = np.zeros(len(vals))
indicies = []
A0s = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

for A0 in A0s:
    for i in range(len(vals)):
        params = vals[i]
        delta2 = params[0]
        rtot_t = params[1]
        kon = params[2]
        koff = params[3]
        rtot_f = params[4]
        qon = params[5]
        qoff = params[6]
        delta4 = params[7]
        delta7 = params[8]
        Y[i] = calc_fc(A0, delta2, rtot_t, kon, koff, rtot_f, qon, qoff, delta4, delta7)
    Si = sobol.analyze(problem, Y)
    indicies.append(Si['ST'])
    lost.append(Si)

delta2_si = []
rtot_t_si = []
kon_si = []
koff_si = []
rtot_f_si = []
qon_si = []
qoff_si = []
delta4_si = []
delta7_si = []

for i in range(len(indicies)):
    vals = indicies[i]
    delta2_si.append(vals[0])
    rtot_t_si.append(vals[1])
    kon_si.append(vals[2])
    koff_si.append(vals[3])
    rtot_f_si.append(vals[4])
    qon_si.append(vals[5])
    qoff_si.append(vals[6])
    delta4_si.append(vals[7])
    delta7_si.append(vals[8])

data = [delta2_si, rtot_t_si, kon_si, koff_si, rtot_f_si, qon_si, qoff_si, delta4_si, delta7_si]
labels = [r'$\delta_2$', r'$r_{tot}^t$', r'$k_{on}$', r'$k_{off}$', r'$r_{tot}^f$', r'$q_{on}$', r'$q_{off}$', r'$\delta_4$', r'$\delta_7$']

print(lost[0])
sns.set_theme()

for i in range(len(data)):
    plt.plot(A0s, data[i], label=labels[i])
    plt.scatter(A0s, data[i])

plt.xscale('log')
plt.xlabel('Antibody concentration')
plt.legend(loc='best')
plt.show()

print('done')