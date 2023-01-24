import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp

sns.set_theme()

#params
Ainit = 1e-7
kon = 1e5
koff = 1e-4
rtot= 1e5
rho = 1
k = kon*Ainit*rho
alpha = k/koff
Target_cell_number = 5e3
Atot = 1e16*Ainit/Target_cell_number
beta = Atot/rtot
delta = 4

#sim params
t_end = 10
t = np.geomspace(1e-10, t_end, 5000)
tspan = [1e-10, t_end]
z0 = [0, 0]

#model
def dimensionless_model(t, z):
    dA1 = 2*alpha*(1-z[0]-2*z[1])*(beta-z[0]-z[1]) - z[0] - delta*alpha*(1-z[0]-2*z[1])*z[0] + 2*z[1]
    dA2 = delta*alpha*(1-z[0]-2*z[1])*z[0] - 2*z[1]

    return [dA1, dA2]

#solve model
z = solve_ivp(dimensionless_model, tspan, z0, method='Radau', t_eval=t)

#set up grid for quiver plot
y1 = np.linspace(0, 1, 25)
y2 = np.linspace(0, 1, 25)

Y1, Y2 = np.meshgrid(y1, y2)

t = 0

A1, A2 = np.zeros(Y1.shape), np.zeros(Y2.shape)

NI, NJ = Y1.shape

for i in range(NI):
    for j in range(NJ):
        x = Y1[i, j]
        y = Y2[i, j]
        yprime = dimensionless_model(t, [x, y])
        A1[i, j] = yprime[0]
        A2[i, j] = yprime[1]

#define nullclines
def A2_null1(x):

    return (delta*alpha*(1-x)*x)/(2*(1+delta*alpha*x))

def A2_null2(x):

    return 0.5*(-((1/2)*(3*x - 1 - beta) + (delta/2)*x + (1/(2*alpha))) + ((((1/2)*(3*x - 1 - beta) + (delta/2)*x + (1/(2*alpha)))**2) - 4*((1/2)*(beta - x - beta*x + x**2) - (x/(4*alpha)) - (delta/4)*(x - x**2)))**(1/2))

def A2_null3(x):
    return 0.5*(-((1/2)*(3*x - 1 - beta) + (delta/2)*x + (1/(2*alpha))) - ((((1/2)*(3*x - 1 - beta) + (delta/2)*x + (1/(2*alpha)))**2) - 4*((1/2)*(beta - x - beta*x + x**2) - (x/(4*alpha)) - (delta/4)*(x - x**2)))**(1/2))

def A1_null1(x):
    return 0.5*((1-2*x) + (((2*x - 1)**2) - (8/(delta*alpha)*x))**(0.5))

def A1_null2(x):
    return 0.5*((1-2*x) - (((2*x - 1)**2) - (8/(delta*alpha)*x))**(0.5))

def A1_null3(x):
    return (1/(2*(alpha*(2+delta))))*(-(2*alpha*(-1 - beta + 3*x) - 1 - delta*alpha*(1 - 2*x)) + (((2*alpha*(-1 - beta + 3*x) - 1 - delta*alpha*(1 - 2*x))**2) - 4*(alpha*(2+delta))*(2*alpha*(beta - x - 2*beta*x + 2*(x**2)) + 2*x))**(0.5))

def A1_null4(x):
    return (1/(2*(alpha*(2+delta))))*(-(2*alpha*(-1 - beta + 3*x) - 1 - delta*alpha*(1 - 2*x)) - (((2*alpha*(-1 - beta + 3*x) - 1 - delta*alpha*(1 - 2*x))**2) - 4*(alpha*(2+delta))*(2*alpha*(beta - x - 2*beta*x + 2*(x**2)) + 2*x))**(0.5))


A1s = np.linspace(0, 1, 200)
A2s = np.linspace(0, 1, 200)

#plot quiver plot
Q = plt.quiver(Y1, Y2, A1, A2, color='r')

#plot nullclines
plt.plot(A1_null1(A2s), A2s, label='A1 null 1')
plt.plot(A1_null2(A2s), A2s, label='A1 null 1')
plt.plot(A1_null3(A2s), A2s, label='A1 null 3')
plt.plot(A1_null4(A2s), A2s, label='A1 null 4')
plt.plot(A1s, A2_null1(A1s), label='A2 null 1')
plt.plot(A1s, A2_null2(A1s), label='A2 null 2')
plt.plot(A1s, A2_null3(A1s), label='A2 null 3')

#plot sample trajectory
z1 = z.y[0]
z2 = z.y[1]
plt.plot(z1, z2, '--', label='trajectory')
plt.plot([z1[0]], [z2[0]], 'o')
plt.plot([z1[-1]], [z2[-1]], 'o')
plt.xlabel(r'$A_1$')
plt.ylabel(r'$A_2$')
plt.xlim(-0.05, 1)
plt.ylim(-0.05, 1)
plt.legend(loc='best')
plt.show()