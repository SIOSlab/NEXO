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

# Reference epoch (MJD)
ref_epoch = 58849.0

# Initial & final times for error metrics
ti =  0.0
tf = 10.0

#-------------------------------------------------------------------------------

# Arrays
err_dim = (len(npass_vals), len(nq_vals), norb)
rmse    = np.empty(err_dim)
chi2m   = np.empty(err_dim)

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

            # Compute errors
            rmse_k, chi2m_k, ok = nexo.eval_err_srspf(lam[k], eta[:, k], Xi[:, :, k], \
                                        xm, l_xx, ti, tf)
            # Stop if error in error
            if (not ok):
                print('Error calculation failed!')
                exit()

            # Save errors
            rmse [i, j, k] = rmse_k
            chi2m[i, j, k] = chi2m_k

# Overall means
rmse_overall  = np.sqrt(np.mean(rmse**2, axis=2))
chi2m_overall = np.mean(chi2m, axis=2) 

# Save mean values
np.savetxt('tables/rmse.csv',  rmse_overall,  delimiter=',')
np.savetxt('tables/chi2m.csv', chi2m_overall, delimiter=',')

# Save tuning parameters
np.savetxt('tables/npass.csv', np.array(npass_vals), delimiter=',')
np.savetxt('tables/nq.csv',    np.array(nq_vals),    delimiter=',')
