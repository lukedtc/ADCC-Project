import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.special import comb

def prob_eqn(t, z, k_on, k_off, A):

    return  k_on*A*(1 - z) - k_off*z

def bindings(i, n, s):
    s = int(s)
    i = int(i)
    #coeff = np.math.factorial(s)//(np.math.factorial(i)*np.math.factorial(s-i))
    return comb(s, i)*(n**i)*((1-n)**(s-i))

def solve_n(t_end, k_on, k_off, A0):
    z0 = [0]
    t = np.geomspace(1e-6, t_end, 1000)
    t_span = [0, t_end]

    z = solve_ivp(prob_eqn, t_span, z0, args = (k_on, k_off, A0), method="Radau", t_eval=t)

    vals = z.y[0]
    #plt.semilogx(z.t, vals)
    n= vals[-1]

    return n

def calc_distribution(ints, n, s):

    probs = []

    for i in ints:
        probs.append(bindings(i, n, s))
    
    return probs

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x <= 1e-6 else x for x in values]

A0s = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
number_receptors = 1e5
ints = np.arange(0, number_receptors)
y = []
for A0 in A0s:

    n = solve_n(1000, 1e5, 1e-4, A0)
    probs = calc_distribution(ints, n, number_receptors)
    y.append(probs)

for i in range(len(y)):
    probs = y[i]
    filtered_probs = zero_to_nan(probs)
    plt.plot(ints, filtered_probs, label=r'$A_0 = $' + str(A0s[i]))

plt.xlabel('Number of bound receptors')
plt.legend()
plt.show()
    


