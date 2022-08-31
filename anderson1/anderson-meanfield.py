import numpy as np
import math

from matplotlib import pyplot as plt

### comentar as 4 linhas abaixo caso nao tenha o LaTeX no matplotlib ###
from matplotlib import rc
plt.style.use('bmh')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
########################################################################

pi = math.pi
Delta = 1.0
ERR = 0.0001

def f(up, down, eps0, U):
    #up0, down0 = up, down
    up   = 0.5 - (1/pi) * math.atan((eps0 + U * down)/Delta)
    down = 0.5 - (1/pi) * math.atan((eps0 + U * up) / Delta)
    return up, down

def solve(up0, down0, eps0, U, err):
    up, down = f(up0, down0, eps0, U)
    while abs(up - up0) + abs(down - down0) > err:
        up0, down0 = up, down
        up, down = f(up0, down0, eps0, U)
        #up, down = (up + up0)/2, (down + down0)/2
    return up, down

def main():
    U_range = [200]
    for U in U_range:
        print(U)
        for vec in [[0.0, 0.2], [0.9, 0.4], [0.8, 0.0], [2.0, 0.1], [0.5, 0.5]]:
            up0, down0 = vec[0], vec[1]
            eps_range = np.linspace(-1.5*U, U/2.0, 100)
            up_list, down_list = [], []
            for eps0 in eps_range:
                up, down = solve(up0, down0, eps0, U, ERR)
                up_list.append(up)
                down_list.append(down)
            up_array = np.array(up_list)
            down_array = np.array(down_list)
            plt.plot(eps_range, up_array, label=r'$\langle \hat{n}_{d\uparrow} \rangle$')
            plt.plot(eps_range, down_array, label=r'$\langle \hat{n}_{d\downarrow} \rangle$')
            plt.plot(eps_range, up_array + down_array, label=r'$\langle \hat{n}_{d\uparrow} \rangle + \langle \hat{n}_{d\downarrow} \rangle$')
            plt.xlabel(r'$\overline{\epsilon}_0$', fontsize=20)
            plt.ylabel(r'$\langle \hat{n} \rangle$', fontsize=20)
            plt.legend(fontsize=16)
            plt.title(r'Mean-field Anderson Model - $U=%.2f$, $\hat{n}_{d\uparrow}=%.2f$, $\hat{n}_{d\downarrow}=%.2f$' % (U, up0, down0))
            plt.savefig('AndersonMF-U_%.1f-up_%.2f-down_%.2f.png' % (U, up0, down0), dpi=300, format='png', bbox_inches="tight")
            plt.clf()
    return 0

if __name__ == '__main__':
    main()
