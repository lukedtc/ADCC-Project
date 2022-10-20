import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.special import comb

def prob_eqns (t, z, k_on, k_off, q_on, q_off, A):

    dn = k_on*A*(1-z[0]) - k_off*z[0] - q_on*z[0]*(1-z[1]) + q_off*z[2]
    dm = q_on*A*(1-z[1]) - q_off*z[1] - k_on*z[1]*(1-z[0]) + k_off*z[2]
    dalpha = q_on*z[0]*(1-z[1]) + k_on*z[1]*(1-z[0]) - (q_off + k_off)*z[2]

    dz = [dn, dm, dalpha]

    return dz

def bindings(i, prob, total_receptors):

    t_recep = int(total_receptors)
    i = int(i)

    return comb(t_recep, i)*(prob**i)*((1-prob)**(t_recep - i))

def solve_prob(t_end, k_on, k_off, q_on, q_off, A0):
    z0 = [0, 0, 0]
    t = np.geomspace(1e-6, t_end, 10000)
    t_span = [0, t_end]

    z = solve_ivp(prob_eqns, t_span, z0, args = (k_on, k_off, q_on, q_off, A0), method="Radau", t_eval=t)

    n_vals = z.y[0]
    m_vals = z.y[1]
    alpha_vals = z.y[2]
    #plt.plot(z.t, alpha_vals)
    n = n_vals[-1]
    m = m_vals[-1]
    alpha = alpha_vals[-1]

    return [n, m, alpha]

def calc_distribution(ints, prob, total_receptors):

    x = []

    for i in ints:
        x.append(bindings(i, prob, total_receptors))
    
    return x

NK_receptors = 2.2e2
T_receptors = 1e2
NK_ints = np.arange(0, NK_receptors)
T_ints = np.arange(0, T_receptors)
k_on = 1e5
k_off = 1e-4
q_on = 6.5e-3
q_off = 4.7e-3
A0 = 1e-6

[n, m, alpha] = solve_prob(1000, k_on, k_off, q_on, q_off, A0)

n_probs = calc_distribution(T_ints, n, T_receptors)
m_probs = calc_distribution(NK_ints, m, NK_receptors)
alpha_probs = calc_distribution(T_ints, alpha, T_receptors)

print(alpha_probs)
plt.plot(T_ints, alpha_probs)
plt.show()