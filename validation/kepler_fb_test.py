from math import *

import numpy as np

from scipy.special import jv

import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

def phi_fb(dm, ex, ey, n):

    phi = dm

    for k in range(1, n+1):

        a = 0
        b = 0

        for m in range(-n, n+1):

            a = (-1)**m * jv(k + 2*m + 1, ex) * jv(2*m + 1, ey)
            b = (-1)**m * jv(k + 2*m,     ex) * jv(2*m,     ey)

        a = -2.0 * a / k
        b =  2.0 * b / k

        phi = phi + a * cos(k * dm) + b * sin(k * dm) 

    return phi

#-------------------------------------------------------------------------------

def count_it_fb(dm, ex, ey):

    nmax = 10

    phitol = 1.0E-9

    ok = False

    phi = phi_fb(dm, ex, ey, 1)

    n = 2

    while ((not ok) and (n < nmax)):

        phi_prev = phi

        phi = phi_fb(dm, ex, ey, n)

        ok = abs(phi - phi_prev) < phitol

        n = n + 1

    if (not ok):
        n = float('inf')

    return n

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
        it[i, j] = count_it_fb(dm[j], ex[i], ey[i])

#-------------------------------------------------------------------------------

plt.rc('font', size=10)

plt.figure(figsize = (6, 4))

heatmap = plt.pcolormesh(dm_deg, e, it)

colorbar = plt.colorbar(heatmap)

colorbar.set_label("Number of Terms")
colorbar.set_ticks(range(int(np.amin(it)), int(np.amax(it))+1))

plt.xlabel(r"$\Delta M$ ($^\circ$)")
plt.ylabel(r"$e$")

plt.tight_layout()
plt.savefig("plots/kepler_fb_fixed_M0.pdf")
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
        it[i, j] = count_it_fb(dm, ex[i, j], ey[i, j])

#-------------------------------------------------------------------------------

plt.rc('font', size=10)

plt.figure(figsize = (6, 4))

heatmap = plt.pcolormesh(M0_deg, e, it)

colorbar = plt.colorbar(heatmap)

colorbar.set_label("Number of Terms")
colorbar.set_ticks(range(int(np.amin(it)), int(np.amax(it))+1))

plt.xlabel(r"$M_0$ ($^\circ$)")
plt.ylabel(r"$e$")

plt.tight_layout()
plt.savefig("plots/kepler_fb_fixed_dm.pdf")
plt.close()
