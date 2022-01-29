import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate
import seaborn as sns
sns.set()

k_1 = 6.5
k_2 = 6.5
m_1 = 11
m_2 = 2
A = np.array([[0,              1,        0, 0],
              [-(k_1+k_2)/m_1, 0,  k_2/m_1, 0],
              [0,              0,        0, 1],
              [k_2/m_2,        0, -k_2/m_2, 0]])


def sho(t, x):
    return A.dot(x)

x0 = np.array([2, 5, 5, 10])


tf = 35


sol = scipy.integrate.solve_ivp(fun=sho, y0=x0, t_span=(0, tf), method='DOP853', atol=1e-12, rtol=1e-12)

dt_exp = 0.01
t_exp = np.arange(0, tf, dt_exp)
exp_dtA = scipy.linalg.expm(dt_exp * A)
sol_exp = [x0]
for t in t_exp[1:]:
    sol_exp.append(exp_dtA.dot(sol_exp[-1]))
sol_exp = np.array(sol_exp).T

plt.figure()

plt.plot(sol_exp[0, :], sol_exp[1, :], color='b', linestyle='-')
plt.plot(sol_exp[2, :], sol_exp[3, :], color='orange', linestyle='-')
plt.ylabel('Wychylenie')
plt.xlabel('Czas')
plt.title('Przestrzeń fazowa dwóch ciał', fontsize=20)

plt.show()