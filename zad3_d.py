import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()

t = np.linspace(0, 40, 400)
x = [0, 0]
delta = 0.14
k = 6.5
m = 11
omega =  math.sqrt(k/m)
A = 0.1


def sho(t, x):
    result = [x[1], (-2 * delta * x[1] - (k / m) * x[0] + A * math.sin(0.5*omega*t))]
    return result

def sho2(t, x_2):
    result = [x_2[1], (-2 * delta * x_2[1] - (k / m) * x_2[0] + A * math.sin(1*omega*t))]
    return result

def sho3(t, x_3):
    result = [x_3[1], (-2 * delta * x_3[1] - (k / m) * x_3[0] + A * math.sin(2*omega*t))]
    return result



solution = solve_ivp(sho, [0, 1000], y0=x, t_eval=t)
solution2 = solve_ivp(sho2, [0, 1000], y0=x, t_eval=t)
solution3 = solve_ivp(sho3, [0, 1000], y0=x, t_eval=t)
plt.plot(t, solution.y[0], 'b', label='$\omega = 0.5$')
plt.plot(t, solution2.y[0], 'orange',label='$\omega = 1$')
plt.plot(t, solution3.y[0], 'red', label='$\omega = 2$')

plt.xlabel("Czas")
plt.ylabel("Wychylenie")
plt.title('Oscylator z tłumieniem i z wymuszeniem', fontsize=20)
plt.grid(True)
plt.legend()
plt.show()