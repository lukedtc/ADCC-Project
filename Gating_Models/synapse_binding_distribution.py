from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.special import comb

sns.set_theme()

def prob_eqns (t, z, k_on, k_off, q_on, q_off, A0):

    dn = k_on*A0*z[3]*(1-z[0]-z[2]) - k_off*z[0] - q_on*A0*z[0]*(1-z[1] - z[2]) + q_off*z[2]
    dm = q_on*A0*z[3]*(1-z[1]-z[2]) - q_off*z[1] - k_on*A0*z[1]*(1-z[0]-z[2]) + k_off*z[2]
    dalpha = q_on*A0*z[0]*(1-z[1]-z[2]) + k_on*A0*z[1]*(1-z[0]-z[2]) - (q_off + k_off)*z[2]
    dA =- k_on*A0*z[3]*(1-z[0]-z[2]) + k_off*z[0] -  q_on*A0*z[3]*(1-z[1]-z[2]) + q_off*z[1]

    dz = [dn, dm, dalpha, dA]

    return dz

def bindings(i, prob, total_receptors):

    t_recep = int(total_receptors)
    i = int(i)

    return comb(t_recep, i)*(prob**i)*((1-prob)**(t_recep - i))

def solve_prob(t_end, k_on, k_off, q_on, q_off, A0):
    z0 = [1, 0, 0, 0]
    t = np.geomspace(1e-6, t_end, 10000)
    t_span = [0, t_end]

    z = solve_ivp(prob_eqns, t_span, z0, args = (k_on, k_off, q_on, q_off, A0), method="Radau", t_eval=t)

    n_vals = z.y[0]
    m_vals = z.y[1]
    alpha_vals = z.y[2]
    n = n_vals[-1]
    m = m_vals[-1]
    alpha = alpha_vals[-1]
    #plt.plot(z.t, z.y[0], label='n')
    #plt.plot(z.t, z.y[1], label='m')
    #plt.plot(z.t, z.y[2], label=r'$\alpha$')
    #plt.plot(z.t, z.y[3], label='A')
    #plt.legend()
    #plt.show()
    return [n, m, alpha]

def calc_distribution(ints, prob, total_receptors):

    x = []

    for i in ints:
        x.append(bindings(i, prob, total_receptors))
    
    return x

NK_receptors = 2.2e3
T_receptors = 1e3
NK_ints = np.arange(0, NK_receptors)
T_ints = np.arange(0, T_receptors)
k_on = 1e5
k_off = 1e-4
q_on = 6.5e5
q_off = 4.7e-3
A0 = 1e-5
A0s = [1e-12, 5e-12, 1e-11, 5e-11, 1e-10, 5e-10, 1e-9, 5e-9, 
        1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
[n, m, alpha] = solve_prob(100, k_on, k_off, q_on, q_off, A0)

n_probs = calc_distribution(T_ints, n, T_receptors)
m_probs = calc_distribution(NK_ints, m, NK_receptors)
alpha_probs = calc_distribution(T_ints, alpha, T_receptors)


plt.plot(T_ints, n_probs, label='tumour')
#plt.plot(NK_ints, m_probs, label='NK')
plt.plot(T_ints, alpha_probs, label='complex')
plt.legend()
plt.show()

max_tumour_prob = []
max_complex_prob = []

for A0 in A0s:
    [n, m, alpha] = solve_prob(100, k_on, k_off, q_on, q_off, A0)
    n_probs = calc_distribution(T_ints, n, T_receptors)
    alpha_probs = calc_distribution(T_ints, alpha, T_receptors) 

    max_tumour_prob.append(np.argmax(n_probs))
    max_complex_prob.append(np.argmax(alpha_probs))

plt.scatter(A0s, max_tumour_prob, label = 'tumour receptors')
plt.plot(A0s, max_tumour_prob)
plt.scatter(A0s, max_complex_prob, label = 'complexes')
plt.plot(A0s, max_complex_prob)
plt.legend(loc='best')
plt.xlabel('antibody concentration')
plt.ylabel('number of receptors')
plt.xscale('log')
plt.show()
    
