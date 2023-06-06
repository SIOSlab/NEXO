from math import *

import numpy as np

import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

def count_it_nr(dm, ex, ey):

    maxit = 10

    phitol = 1.0E-9

    phi = dm - ey * cos(dm) + ex * sin(dm)
    
    ok = False

    it = 0

    try:

        while ((not ok) and (it < maxit)):

            cos_phi = cos(phi)
            sin_phi = sin(phi)

            f = phi - dm - ex * sin_phi + ey * cos_phi
        
            fd = 1 - ex * cos_phi - ey * sin_phi

            dphi = f / fd

            phi = phi - dphi

            it = it + 1

            ok = abs(dphi) < phitol

    except ValueError:

        ok = False

    if (not ok):
        it = float('inf')

    return it

#-------------------------------------------------------------------------------

M0_deg = 120

e = np.linspace(0.0, 0.99, 100)

dm_deg = np.linspace(0.0, 360.0, 100)

#-------------------------------------------------------------------------------

M0 = np.radians(M0_deg)

dm = np.radians(dm_deg)

ex =  e * cos(M0)
ey = -e * sin(M0)

it = np.empty((e.size, dm.size))

for i in range(e.size):
    for j in range(dm.size):
        it[i, j] = count_it_nr(dm[j], ex[i], ey[i])

#-------------------------------------------------------------------------------

plt.rc('font', size=10)

plt.figure(figsize = (6, 4))

heatmap = plt.pcolormesh(dm_deg, e, it)

colorbar = plt.colorbar(heatmap)

colorbar.set_label("Number of Iterations")

plt.xlabel(r"$\Delta M$ ($^\circ$)")
plt.ylabel(r"$e$")

plt.tight_layout()
plt.savefig("plots/kepler_nr_fixed_M0.pdf")
plt.close()

#-------------------------------------------------------------------------------

dm_deg = 120

e = np.linspace(0.0, 0.99, 100)

M0_deg = np.linspace(0.0, 360.0, 100)

#-------------------------------------------------------------------------------

M0 = np.radians(M0_deg)

dm = np.radians(dm_deg)

ex = np.outer(e,  np.cos(M0))
ey = np.outer(e, -np.sin(M0))

it = np.empty((e.size, M0.size))

for i in range(e.size):
    for j in range(M0.size):
        it[i, j] = count_it_nr(dm, ex[i, j], ey[i, j])

#-------------------------------------------------------------------------------

plt.rc('font', size=10)

plt.figure(figsize = (6, 4))

heatmap = plt.pcolormesh(M0_deg, e, it)

colorbar = plt.colorbar(heatmap)

colorbar.set_label("Number of Iterations")

plt.xlabel(r"$M_0$ ($^\circ$)")
plt.ylabel(r"$e$")

plt.tight_layout()
plt.savefig("plots/kepler_nr_fixed_dm.pdf")
plt.close()
