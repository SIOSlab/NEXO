import numpy as np

import matplotlib.pyplot as plt

from corner import corner

from get_priors import get_priors

#-------------------------------------------------------------------------------

mstar = 1.0

std_mstar = 0.1

plx = 10

std_plx = 1

path = 'priors/'

nsamp = 10000000

np.random.seed(808)

#-------------------------------------------------------------------------------

wgt, xm, l_xx = get_priors(mstar, std_mstar, plx, std_plx, path)

ncomp = np.size(wgt)

ind = np.random.choice(ncomp, size=nsamp, p=wgt)

xs = np.random.normal(size=(7, nsamp))

x = np.empty((nsamp, 7))

for j in range(nsamp):
    x[j, :] = np.matmul(l_xx[:, :, ind[j]], xs[:, j]) + xm[:, ind[j]]

#-------------------------------------------------------------------------------

figure = corner(x, bins=100)

figure.savefig('priors/prior_plot.pdf')
