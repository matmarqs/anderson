import numpy as np
from math import exp
from sympy import *
from scipy.integrate import quad
from matplotlib import pyplot as plt

### comentar as 4 linhas abaixo caso nao tenha o LaTeX no matplotlib ###
from matplotlib import rc
plt.style.use('bmh')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
########################################################################

pi = np.pi
Delta0 = 0.5
band = 10   # np.inf for infinity
ERR = 0.001
kB = 1


# funcao de input Delta(w)
def Delta(w):
    return Delta0 * (1 - (w/band)**2)


# Lamb(w) = P.V. integral( - Delta(E) / (E - w)  dE)
def Lamb(w):
    integral, error = quad(Delta, -band, band, weight='cauchy', wvar=w)
    return -integral


# Here the Green's function G is defined
# A_func returns the spectral function 'A' and its LaTeX expression
def A_lamb():
    w, e = symbols(r'\omega \epsilon_{\sigma}', real=True)
    # 'n' is for n_{\overline{\sigma}}
    n = Symbol(r'\langle n_{\overline{\sigma}} \rangle', positive=True)
    U = Symbol(r'U', positive=True)
    # Sigma_0 = L - i D
    L, D = symbols(r'\Lamb \Delta', real=True)
    # Green's function definition
    G, A_sympy = symbols('G A', cls=Function)
    G = (1-n)/(w - e - (L - I * D)) + n/(w - e - U - (L - I * D))
    # Spectral function
    A_sympy = -(1/pi) * im(G)
    A_func = lambdify([w, e, n, U, L, D], A_sympy)
    return A_func, latex(A_sympy)


def A_def(A_func):
    A = lambda w, n, e, U: np.abs(A_func(w, n, e, U, Lamb(w), Delta(w)))
    return A


def AnF_def(A_func):
    AnF = lambda w, n, e, U, T: np.abs(A_func(w, n, e, U, Lamb(w), Delta(w)))/(exp(w/(kB*T))+1)
    return AnF


# fixed-point function, mas com T = 0
def f(up, dn, e, U, A):
    up, errup = quad( A, -band, 0, args=(dn, e, U) )
    dn, errdn = quad( A, -band, 0, args=(up, e, U) )
    return up, dn


# for T = 0
def solve(up0, dn0, e, U, err, A):
    up, dn = f(up0, dn0, e, U, A)
    while abs(up - up0) + abs(dn - dn0) > err:  # criterio de convergencia
        up0, dn0 = up, dn
        up , dn  = f(up0, dn0, e, U, A)
    return up, dn


# fixed-point function
def f_nF(up, dn, e, U, T, AnF):
    up, errup = quad( AnF, -band, band, args=(dn, e, U, T) )
    dn, errdn = quad( AnF, -band, band, args=(up, e, U, T) )
    return up, dn


# for T finite
def solveT(up0, dn0, e, U, T, err, AnF):
    up, dn = f_nF(up0, dn0, e, U, T, AnF)
    while abs(up - up0) + abs(dn - dn0) > err:  # criterio de convergencia
        up0, dn0 = up, dn
        up , dn  = f_nF(up0, dn0, e, U, T, AnF)
    return up, dn



def main():
    A_func, latex_A = A_lamb()
    A = A_def(A_func)
    AnF = AnF_def(A_func)
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
            w_range = np.linspace(-12, 12, 150)
            for e in e_range:
                A_range = np.array([ A(w, n, e, U) for w in w_range ])
                plt.plot(w_range, A_range, label=r'$\epsilon_0 = %.1f$' % (e))
            plt.xlabel(r'$\omega$', fontsize=20)
            plt.ylabel(r'$A(\omega)$', fontsize=20)
            plt.legend(fontsize=16)
            plt.title(r'Spectral function $A(\omega, n)$: $U=%.0f$, $\Delta_0=%.1f$, $D=%.0f$, $n=%.1f$' % (U, Delta0, band, n))
            plt.savefig('spectralfunc-U_%.0f-Delta0_%.1f-D_%.0f-n_%.1f.png' % (U, Delta0, band, n), dpi=300, format='png', bbox_inches="tight")
            plt.clf()

        for T in T_range:
            print("    T =", T)

            # Plotando <n_{d\sigma}> em função de e
            e_range = np.linspace(-4*U, U, 100)
            for vec in [[0.8, 0.6]]:
                up0, dn0 = vec[0], vec[1]
                up_list, dn_list = [], []
                if T == 0:
                    for e in e_range:
                        up, dn = solve(up0, dn0, e, U, ERR, A)
                        up_list.append(up)
                        dn_list.append(dn)
                else:
                    for e in e_range:
                        up, dn = solve(up0, dn0, e, U, T, ERR, AnF)
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
