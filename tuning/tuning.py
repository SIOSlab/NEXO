import sys
import os

from math import *

import numpy as np
import scipy as sp
import time

import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table

sys.path.append('..')

import nexo

#-------------------------------------------------------------------------------

nq_vals = [1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000]

npass_vals = [1, 2, 3, 4, 5]

min_per = 0.1

max_per = 10000

std_lam = 0.5

std_eta = 0.15

#-------------------------------------------------------------------------------

# Mean parallax (mas)
plxm = 100

# Standard deviation of parallax (mas)
std_plx = 0.1

# Mean total mass (solar masses)
mm = 1.0

# Standard deviation of total mass
std_m = 0.1

# Read classical parameters
coe_tru = ascii.read('gen/coe_tru.csv')

# Convert to nonsingular parameters
lam, eta, Xi = nexo.coe2nse(coe_tru['plx'], coe_tru['sma'], coe_tru['ecc'], \
        coe_tru['inc'], coe_tru['lan'], coe_tru['aop'], coe_tru['mae'], \
        coe_tru['per'])

# Number of cases
norb = len(coe_tru)

# Reshaped xi array
xi = np.empty((4, norb))
xi[0:2, :] = Xi[:, 0, :]
xi[2:4, :] = Xi[:, 1, :]

# True x values
x = np.vstack((lam, eta, xi))

# Reference epoch (MJD)
ref_epoch = 58849.0

#-------------------------------------------------------------------------------

# Arrays
err_dim = (len(npass_vals), len(nq_vals), norb)
lam_err = np.empty(err_dim)
eta_err = np.empty(err_dim)
xi_err  = np.empty(err_dim)
inside  = np.empty(err_dim)

# Confidence
p = 0.95

# Degrees of freedom
df = 7

# Maximum chi-square value
chi2_max = df / (1 - p)

# Iterate over number of passes
for i in range(len(npass_vals)):

    # Number of passes
    npass = npass_vals[i]

    # Iterate over number of components
    for j in range(len(nq_vals)):

        # Number of components
        nq = nq_vals[j]

        # Print status
        print('---------------------------------------------------------------')
        print(str(npass) + ' passes; ' + str(nq) + ' mixture components')
        print('---------------------------------------------------------------')

        # Iterate over orbits
        for k in range(norb):
    
            # Print status
            print('Running case ' + str(k+1) + ' of ' + str(norb))

            # Read measurements
            meas_table = ascii.read('gen/meas_' + str(k+1) + '.csv')

            # Measurement times
            t = (meas_table['epoch'] - ref_epoch) / 365.25
    
            # Filter-friendly measurements
            z, cov_ww = nexo.radec2z(
                meas_table['raoff'],
                meas_table['decoff'],
                meas_table['raoff_err'],
                meas_table['decoff_err'],
                meas_table['radec_corr']
                )

            # Run filter
            xm, l_xx = nexo.mix_filter(npass, nq, plxm, std_plx, mm, std_m, \
                    min_per, max_per, std_lam, std_eta, t, z, cov_ww)

            # Estimated nonsingular parameters
            lam_est = xm[0]
            eta_est = xm[1:3]
            xi_est  = xm[3:7]

            # Errors
            lam_err[i, j, k] = np.abs(lam_est - lam[k])
            eta_err[i, j, k] = 100 * np.linalg.norm(eta_est - eta[:, k]) \
                                    / np.linalg.norm(eta[:, k])
            xi_err [i, j, k] = 100 * np.linalg.norm(xi_est - xi[:, k]) \
                                    / np.linalg.norm(xi[:, k])

            # Value of chi-square
            chi2 = np.linalg.norm(np.linalg.solve(l_xx, xm - x[:, k]))**2

            # Outside of credible ellipsoid
            if chi2 > chi2_max:
                inside[i, j, k] = 100
            else:
                inside[i, j, k] = 0


# Root-mean-square errors
lam_rmse = np.sqrt(np.mean(lam_err**2, axis = 2))
eta_rmse = np.sqrt(np.mean(eta_err**2, axis = 2))
xi_rmse  = np.sqrt(np.mean(xi_err**2,  axis = 2))
perc_out = np.mean(inside, axis = 2)

#-------------------------------------------------------------------------------

elmt = ['lam', 'eta', 'xi', 'cred']

label = ['RMSE', 'RMSE (%)', 'RMSE (%)', \
         'Percentage Outside of Credible Region']

rmse = [lam_rmse, eta_rmse, xi_rmse, perc_out]

perc = [False, True, True]

for i in range(4):

    plt.rc('font', size=10)

    plt.figure(figsize = (6, 4))

    plt.xscale('log')

    heatmap = plt.pcolormesh(nq_vals, npass_vals, rmse[i])

    colorbar = plt.colorbar(heatmap)

    colorbar.set_label(label[i])
    
    plt.xlabel("Number of Mixture Components")
    plt.ylabel("Number of Passes")

    plt.xticks(nq_vals)
    plt.yticks(npass_vals)

    plt.tight_layout()
    plt.savefig("plots/" + elmt[i] + "_err_npq.pdf")
    plt.close()
