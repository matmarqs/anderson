#!/usr/bin/env python3

import numpy as np
from scipy.integrate import simpson as integral
#from scipy.integrate import quad
#from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
### comentar as 4 linhas abaixo caso nao tenha o LaTeX no matplotlib ###
from matplotlib import rc
plt.style.use('bmh')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
########################################################################


sig_file = "sig.inp"
ERR = 0.001
Delta0 = 1    # Gregorio: Delta = U/20
D = 10   # -D <= omega <= D
U = 4
kB = 1
T = 0.1 # Gregorio: T = 0.1
e0 = -U/2
lamb0 = -2.0


class PP:
    # eps is the energies of PPs
    # freq is the range of frequencies
    # sig is the pseudo-particle self-energy at each frequency
    def __init__(self, Eps, Freq, SigM):
        self.n_pp = 4   # number of PP's
        self.eps = Eps  # energy of each PP
        self.freq = Freq    # range of frequencies
        self.n_freq = Freq.shape[0]  # number of frequencies
        # if Sig has columns missing, we complete them setting everything to zero
        if SigM.shape[0] < 2*self.n_pp:  # 2*n_pp = real, imaginary parts for n_pp particles
            SigM = np.append(SigM, np.zeros((2*self.n_pp - SigM.shape[0], self.n_freq)), axis=0)
        # self.Sig = self-energies of PP's (complex numbers)
        self.Sig = np.array([SigM[2*i]+1j*SigM[2*i+1] for i in range(self.n_pp)])
        # G is the green function of each PP
        # G[1] and G[2] are always equal, because the system is symmetric between spins up and down
        self.G = np.array([1/(self.freq - self.eps[i] - self.Sig[i]) for i in range(self.n_pp)])
    def update_G(self, i):
        self.G[i] = 1/(self.freq - self.eps[i] - self.Sig[i])
    def update_Sig(self):
        for i in range(self.n_freq):
            # calculate Sig_b
            self.Sig[0][i] = (1/np.pi) * integral( n_F(self.freq - self.freq[i]) *
                Delta(self.freq - self.freq[i]) * (2 * self.G[1]), self.freq )
        # update G_b
        self.update_G(0)
        for i in range(self.n_freq):
            # calculate Sig_up
            self.Sig[1][i] = integral( n_F(self.freq - self.freq[i]) *
            (
                Delta(-self.freq + self.freq[i]) * self.G[0] +
                Delta( self.freq - self.freq[i]) * self.G[3]
            ), self.freq)
        # update G_sigma
        self.update_G(1)
        self.update_G(2)
        for i in range(self.n_freq):
            # calculate Sig_b
            self.Sig[3][i] = (1/np.pi) * integral( n_F(self.freq - self.freq[i]) *
                Delta(-self.freq + self.freq[i]) * (2 * self.G[1]), self.freq )
        # update G_a
        self.update_G(3)


# fermi function
def n_F(e):
    return 1/(np.exp(e/(kB*T))+1)

# hybridization function
def Delta(w):
    return (np.abs(w) <= D) * Delta0  # returns Delta0 if inside band, 0 otherwise

def spectral(G):
    return np.abs(np.imag(G))/np.pi

def get_index_greater(array, x):
    N = len(array)
    for i in range(N):
        if array[i] > x:
            return i
    return -1

def get_index_lesser(array, x):
    N = len(array)
    for i in range(N):
        if array[N-i-1] < x:
            return N-i-1
    return -1

def main():
    # we only take input from pseudo-particle self-energies
    # because for now we assume hybridization function Delta = cte.
    SigInput = np.loadtxt(sig_file, unpack=True, ndmin=2)   # ndmin=2 is to be a matrix
    Freq, SigMatrix = SigInput[0], SigInput[1:]
    # energies of pseudo-particles
    Eps = np.array([0, e0, e0, 2*e0]) + lamb0
    pp = PP(Eps, Freq, SigMatrix)

    for _ in range(20):
        pp.update_Sig()

    A_b = spectral(pp.G[0])
    A_up = spectral(pp.G[1])
    A_a = spectral(pp.G[3])

    plt.plot(Freq[250:752],  A_b[250:752], label=r'$A_{b}$')
    plt.plot(Freq[250:752], A_up[250:752], label=r'$A_{\uparrow}$')
    plt.plot(Freq[250:752],  A_a[250:752], label=r'$A_{a}$')
    plt.xlabel(r'$\omega$', fontsize=20)
    plt.ylabel(r'$A(\omega)$', fontsize=20)
    plt.legend(fontsize=16)
    plt.title(r'$A(\omega)$ of PPs')
    plt.savefig("spectral-func-pp.png", dpi=300, format='png', bbox_inches="tight")
    plt.clf()


    #A_b = interp1d(Freq, spectral(pp.G[0]), kind='cubic')
    #A_up = interp1d(Freq, spectral(pp.G[1]), kind='cubic')
    #A_dn = A_up
    #A_a = interp1d(Freq, spectral(pp.G[3]), kind='cubic')
    #normalization, nerr = quad(lambda e: np.exp(-e/(kB*T))*(A_b(e)+A_up(e)+A_dn(e)+A_a(e)),-D,D)
    #nerr = nerr
    ##i_min = get_index_greater(pp.freq, -10)
    ##i_max = get_index_lesser(pp.freq, 10) + 1
    #def rho_up(w):
    #    r, err = quad(lambda e: np.exp(-e/(kB*T))*(A_b(e)*A_up(e+w)+A_dn(e)*A_a(e+w)),-D,D)
    #    err = err   # err is not accessed warning
    #    return r / normalization
    #def rho_dn(w):
    #    r, err = quad(lambda e: np.exp(-e/(kB*T))*(A_b(e)*A_dn(e+w)+A_up(e)*A_a(e+w)),-D,D)
    #    err = err   # err is not accessed warning
    #    return r / normalization

    #plt.plot(Freq, rho_up(Freq), label=r'$\rho_{\uparrow}$')
    #plt.plot(Freq, rho_dn(Freq), label=r'$\rho_{\downarrow}$')
    #plt.xlabel(r'$\omega$', fontsize=20)
    #plt.ylabel(r'$\rho_{\sigma}(\omega)$', fontsize=20)
    #plt.legend(fontsize=16)
    #plt.title(
    #r'$\rho_{\sigma}(\omega, n)$: $U=%.1f$, $T=%.1e$, $\Delta_0=%.1f$, $D=%.0f$, $\lambda=%.1e$, $\epsilon_0=%.1f$' % (U, T, Delta0, D, lamb0, e0))
    #plt.savefig('spectralfunc-U_%.0f-T_%.1e-Delta0_%.1f-D_%.0f-lamb_%.1f-e0_%.1f.png' % (U, T, Delta0, D, lamb0, e0),
    #        dpi=300, format='png', bbox_inches="tight")
    #plt.clf()

    return 0


if __name__ == '__main__':
    main()
