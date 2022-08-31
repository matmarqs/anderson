import numpy as np
import math
from scipy.integrate import quad
from matplotlib import pyplot as plt

### comentar as 4 linhas abaixo caso nao tenha o LaTeX no matplotlib ###
from matplotlib import rc
plt.style.use('bmh')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
########################################################################

pi = math.pi
Delta0 = 5
band = 100   # np.inf for infinity
ERR = 0.001
kB = 1


# funcao de input Delta(w)
def Delta(w):
    return Delta0 * (1 - (w/band)**2)


# Lambda(w) = P.V. integral( - Delta(E) / (E - w)  dE)
def Lambda(w):
    integral, error = quad(Delta, -band, band, weight='cauchy', wvar=w)
    return -integral


def A(w, n, e0, U):
    return (Delta(w) / pi) / ((w - e0 - U*n - Lambda(w))**2 + Delta(w)**2)


def A_nF(w, n, e0, U, T):
    return A(w, n, e0, U) / (math.exp(w / (kB * T)) + 1)


def F(up, dn, e0, U, T):
    up, errup = quad( A_nF, -band, band, args=(dn, e0, U, T) )
    dn, errdn = quad( A_nF, -band, band, args=(up, e0, U, T) )
    return up, dn


# F, mas com T = 0
def f(up, dn, e0, U, T):    # it just ignores the variable T
    up, errup = quad( A, -band, 0, args=(dn, e0, U) )
    dn, errdn = quad( A, -band, 0, args=(up, e0, U) )
    return up, dn


def solve(up0, dn0, e0, U, T, err):
    if T == 0:
        func = f
    else:
        func = F
    up, dn = func(up0, dn0, e0, U, T)
    while abs(up - up0) + abs(dn - dn0) > err:  # criterio de convergencia
        up0, dn0 = up, dn
        up , dn  = func(up0, dn0, e0, U, T)
    return up, dn


def main():
    U_range = [1.0, 2.0, 3.0, 4.0, 5.0]
    #T_range = [0.0, 1.0, 10.0, 100.0, 1000.0]
    #U_range = [1.0]
    T_range = [0.0]

    for U in U_range:
        print("U =", U)

        # Plotando A_{d\sigma}(w)
        e_range = [-2*U, -U, -U/2, U/2, U]
        n_range = [0, 0.5, 1, 1.5, 2]
        for n in n_range:
            w_range = np.linspace( U/2 + U*n - 10, U/2 + U*n + 10, 150)
            for e0 in e_range:
                A_range = np.array([ A(w, n, e0, U) for w in w_range ])
                plt.plot(w_range, A_range, label=r'$\epsilon_0 = %.1f$' % (e0))
            plt.xlabel(r'$\omega$', fontsize=20)
            plt.ylabel(r'$A(\omega)$', fontsize=20)
            plt.legend(fontsize=16)
            plt.title(r'Spectral function $A(\omega, n)$: $U=%.0f$, $\Delta_0=%.1f$, $D=%.0f$, $n=%.1f$' % (U, Delta0, band, n))
            plt.savefig('spectralfunc-U_%.0f-Delta0_%.1f-D_%.0f-n_%.1f.png' % (U, Delta0, band, n), dpi=300, format='png', bbox_inches="tight")
            plt.clf()

        for T in T_range:
            print("    T =", T)

            # Plotando <n_{d\sigma}> em função de e0
            e_range = np.linspace(-2*U, U, 100)
            for vec in [[0.8, 0.6]]:
                up0, dn0 = vec[0], vec[1]
                up_list, dn_list = [], []
                for e0 in e_range:
                    up, dn = solve(up0, dn0, e0, U, T, ERR)
                    up_list.append(up)
                    dn_list.append(dn)
                up_array = np.array(up_list)
                dn_array = np.array(dn_list)
                plt.plot(e_range, up_array, label=r'$\langle \hat{n}_{d\uparrow} \rangle$')
                plt.plot(e_range, dn_array, label=r'$\langle \hat{n}_{d\downarrow} \rangle$')
                plt.plot(e_range, up_array + dn_array, label=r'$\langle \hat{n}_{d\uparrow} \rangle + \langle \hat{n}_{d\downarrow} \rangle$')
                plt.xlabel(r'$\epsilon_0$', fontsize=20)
                plt.ylabel(r'$\langle \hat{n}_{d\sigma} \rangle$', fontsize=20)
                plt.legend(fontsize=16)
                plt.title(r'Occupation number: $U=%.0f$, $T=%.0f$, $\Delta_0=%.1f$, $D=%.0f$, $\hat{n}_{d\uparrow}=%.1f$, $\hat{n}_{d\downarrow}=%.1f$' % (U, T, Delta0, band, up0, dn0))
                plt.savefig('occupnumber-U_%.0f-T_%.0f-Delta0_%.1f-D_%.0f-up_%.1f-down_%.1f.png' % (U, T, Delta0, band, up0, dn0), dpi=300, format='png', bbox_inches="tight")
                plt.clf()

    return 0


if __name__ == '__main__':
    main()
