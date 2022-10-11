#!/usr/bin/env python3

import numpy as np
from scipy.integrate import simpson as integral
#from scipy.integrate import quad
#from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
### comentar as 4 linhas abaixo caso nao tenha o LaTeX no matplotlib ###
#from matplotlib import rc
#plt.style.use('bmh')
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)
########################################################################


def spectral(ImG):
    return -ImG / np.pi

def main():
    # loading data
    freq, ReG, ImG = np.loadtxt("gloc.out", unpack=True)
    spectral_func = spectral(ImG)
    # plotting
    plt.plot(freq,  spectral_func, label=r'$A(\omega)$')
    plt.xlabel(r'$\omega$', fontsize=20)
    plt.ylabel(r'$A(\omega)$', fontsize=20)
    plt.legend(fontsize=16)
    plt.title(r'$A(\omega)$ of the physical impurity')
    plt.savefig("spectral-func-physical.png", dpi=300, format='png', bbox_inches="tight")
    plt.clf()
    return 0


if __name__ == '__main__':
    main()
