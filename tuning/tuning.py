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

alpha_vals = np.linspace(0.1, 0.9)

#-------------------------------------------------------------------------------

# Standard deviation of semi-major axis
std_a = 14 

# Standard deviation of eta
std_eta = 0.20

# Mean parallax (mas)
plx = 100

# Standard deviation of parallax (mas)
std_plx = 0.1

# Mean stellar mass (solar masses)
mstar = 1.0

# Standard deviation of stellar mass
std_mstar = 0.1

# Read classical parameters
coe_tru = ascii.read('gen/coe_tru.csv')

# Convert to nonsingular parameters
lam, eta, Xi = nexo.coe2nse(coe_tru['plx'], coe_tru['sma'], coe_tru['ecc'], \
        coe_tru['inc'], coe_tru['lan'], coe_tru['aop'], coe_tru['mae'], \
        coe_tru['per'])

# Number of cases
norb = len(coe_tru)

# Reference epoch (MJD)
ref_epoch = 58849.0

# Initial & final times for error metrics
ti =  0.0
tf = 10.0

#-------------------------------------------------------------------------------

# Arrays
err_dim = (len(alpha_vals), norb)
rmse    = np.empty(err_dim)
chi2m   = np.empty(err_dim)

# Iterate over number of components
for i in range(len(alpha_vals)):

    # Alpha value
    alpha = alpha_vals[i]

    # Print status
    print('---------------------------------------------------------------')
    print('alpha = ' + str(alpha)                                            )
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
        xm, l_xx = nexo.mix_filter_pop(std_a, std_eta, mstar, std_mstar, \
            plx, std_plx, alpha, t, z, cov_ww)

        # Compute errors
        rmse_k, chi2m_k, ok = nexo.eval_err_srspf(lam[k], eta[:, k], Xi[:, :, k], \
                                        xm, l_xx, ti, tf)
        # Stop if error in error
        if (not ok):
            rmse_k  = np.NAN
            chi2m_k = np.NAN

        # Save errors
        rmse [i, k] = rmse_k
        chi2m[i, k] = chi2m_k

# Overall means
rmse_overall  = np.sqrt(np.nanmean(rmse**2, axis=1))
chi2m_overall = np.nanmean(chi2m, axis=1) 

# Save mean values
np.savetxt('tables/rmse.csv',  rmse_overall,  delimiter=',')
np.savetxt('tables/chi2m.csv', chi2m_overall, delimiter=',')

# Save tuning parameters
np.savetxt('tables/alpha.csv', np.array(alpha_vals), delimiter=',')
