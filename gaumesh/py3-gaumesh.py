#!/usr/bin/env python3
#
# Program gaumesh.py
#
# (c) 2008 by David Jacob.
# Modified by Mateus Marques on September 2022
#
# constructs a Gaussian mesh i.e. the sampling rate dN/dx is given by a
# superposition of Gaussian functions centered at different points x0
# of the abscissa. Thus the mesh points are given by a superposition
# of error functions.
#

import sys
import os
import math

from scipy.special import erf
from scipy.optimize import brentq as find_root
from math import *

class GauMesh:
    # Construct a mesh:
    #  A's:     weights of Gaussians
    #  alpha's: gaussian exponents
    #  x0's:    positions of Gaussian
    #  xmin:    start of mesh
    #  xmax:    end of mesh
    #  Npoints: number of mesh points
    def __init__(self, A, alpha, xpos, xmin, xmax):
        self.ng = len(A)
        self.A = A
        self.alpha = alpha
        self.xpos = xpos
        self.xmin = xmin
        self.xmax = xmax
        self.Npoints = floor(self.N(self.xmax) - self.N(self.xmin))
    def N(self, x):
        val = 0.0
        for i in range(self.ng):
            Ai = self.A[i]
            ai = self.alpha[i]
            xi = self.xpos[i]
            val += 0.5*Ai*sqrt(pi/ai)*erf(sqrt(ai)*(x-xi))
        return val


#### MAIN PROGRAM ####

if len(sys.argv[1:]) == 0 or sys.argv[1] == "--help" or sys.argv[1] == "-h":
    help_message = """

    *** Program gaumesh.py ***

    (c) 2008 by David Jacob

    constructs a Gaussian mesh i.e. the sampling rate dN/dx is given by a
    superposition of Gaussian functions centered at different points x0
    of the abscissa. Thus the mesh points are given by a superposition
    of error functions.

    Usage:

    gaumesh.py         \
    gaumesh.py -h       - Print this help
    gaumesh.py --help  /

    gaumesh.py [list of input file(s)]

    or

    gaumesh.py [list of variable definitions]

    Variables:
    x0[:]   : Positions of Gaussians on the abscissa
    dx0[:]  : Resolutions of the mesh at the center of a Gaussian
    fwhm[:] : FWHMs (Full width at half maximum) of the Gaussians
    xmin    : Lower mesh bound
    xmax    : Upper mesh bound

    Hints:

    1) The number of points in the mesh is approximately given by the sum of the
       ratios 2*fwhm/dx0:

       Npoints = 2*fwhm[0]/dx0[0] + 2*fwhm[1]/dx0[1] + ...

    2) xmin and xmax are merely upper and lower bounds. If the Gaussians are very
       localized (i.e. small fwhm) the actual range of the mesh will be much smaller
       than [xmin,xmax]. To get a mesh with an approximate range of [xmin,xmax]
       have one gaussian centered at (xmin+xmax)/2 with a FWHM of at least (xmax-xmin)/2

    """
    print(help_message)
    exit(1)

x0 = []
dx0 = []
fwhm = []
xmin = 0
xmax = 0

A = []
alpha = []

for arg in sys.argv[1:]:
    if os.path.isfile(arg):
        exec(open(arg).read())
    else:
        exec(arg)

if not x0:
    sys.stderr.write("Error: x0[:] was not defined. Abort.\n")
    sys.exit(1)
if not dx0:
    sys.stderr.write("Error: dx0[:] was not defined. Abort.\n")
    sys.exit(1)
if not fwhm:
    sys.stderr.write("Error: fwhm[:] was not defined. Abort.\n")
    sys.exit(1)
if (not xmin) or (not xmax):
    sys.stderr.write("Error: xmin,xmax have not been defined.Abort.\n")
    sys.exit(1)
if len(x0) != len(dx0) or len(x0) != len(fwhm):
    sys.stderr.write("Error: Lists x0[:],dx0[0],fwhm[:] have different lengths. Abort.\n")
    sys.exit(1)

# Number of Gaussians that build up mesh
ng = len(x0)

for i in range(ng):
    alpha.append(4*log(2)/fwhm[i]**2)
    A.append(2*sqrt(alpha[i]/pi)/erf(sqrt(alpha[i])*dx0[i]/2))

mesh = GauMesh(A, alpha, x0, xmin, xmax)
N_points = mesh.Npoints
Func_min = mesh.N(xmin)
# we define Func = mesh.N(x) - mesh.N(xmin)
# in order for Func(xmin) = 0 and ceil( Func(xmax) ) = N_points
Func = lambda x, n: mesh.N(x) - Func_min - n

print("# N_points =",N_points,", NGauss =",ng,", x0 =",x0,", dx0 =",dx0,", fwhm =",fwhm,", xmin =",xmin,", xmax =",xmax)

# x \in [x_min, x_max]
# F \in [0, N_points + epsilon]
# The idea is
# For each function f_n(x) = F(x) - n, we find its root x_n such that f_n(x_n) = 0
# This is equivalent to calculating the inverse of F at the integers
# This will give us the Gaussian mesh
print("%15.10f" % xmin) # We always print xmin
xn0 = xmin
for n in range(1, N_points + 1):
    xn = find_root(f=Func, a=xn0, b=xmax, args=(n), xtol=1e-15, maxiter=10000)
    print("%15.10f" % xn)
    xn0 = xn
print("%15.10f" % xmax) # We always print xmax
