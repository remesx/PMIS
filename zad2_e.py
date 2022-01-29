import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

t = np.linspace(0, 50, 1000)
x = [2, 5]
x_2 = [5, 10]
delta = 0.14
k = 6.5
m = 11


def sho(t, x):
    result = [x[1], (-2 * delta * x[1] - k / m * x[0])]
    return result


def sho_2(t, x_2):
    result = [x_2[1], (-2 * delta * x_2[1] - k / m * x_2[0])]
    return result


solution = solve_ivp(sho, [0, 1000], y0=x, t_eval=t)
solution_2 = solve_ivp(sho_2, [0, 1000], y0=x_2, t_eval=t)
plt.plot(t, solution.y[0], 'b')
plt.plot(t, solution_2.y[0], 'orange')
# plt.axis([200, 220, -0.05, 0.05])
plt.xlabel("Czas")
plt.ylabel("Wychylenie")
plt.title('Oscylator harmoniczny z tłumieniem i bez wymuszenia', fontsize=12)
plt.grid(True)
plt.show()