import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table

sys.path.append('..')

import nexo

#-------------------------------------------------------------------------------

# Read classical parameters
coe_tru = ascii.read('gen/coe_tru.csv')

# Number of cases
norb = len(coe_tru)

# Convert to nonsingular parameters
lam, eta, Xi = nexo.coe2nse(coe_tru['plx'], coe_tru['sma'], coe_tru['ecc'], \
        coe_tru['inc'], coe_tru['lan'], coe_tru['aop'], coe_tru['mae'], \
        coe_tru['per'])

#-------------------------------------------------------------------------------

# Initial and final times
ti =  0.0
tf = 10.0

# Methods
methods = ['NEXO', 'MCMC', 'OFTI']

# Arrays for errors
rmse  = np.empty((3, norb))
chi2m = np.empty((3, norb))

# Post-process for each method
for m in range(3):

    # Method name
    method = methods[m]

    # Print status
    print('Post-processing for ' + method)

    # Arrays for errors
    rmse_m  = np.full(norb, np.nan)
    chi2m_m = np.full(norb, np.nan)

    # Number of failed fits
    nfail = 0

    # Iterate over orbits
    for j in range(norb):

        # Print status
        print('    Orbit ' + str(j+1) + ' of ' + str(norb))

        # Read elements
        if (method == 'NEXO'):

            # File names for estimates
            filename_xm   = 'results/NEXO_xm_'   + str(j+1) + '.csv'
            filename_l_xx = 'results/NEXO_l_xx_' + str(j+1) + '.csv'  

            # Read files
            xm   = np.genfromtxt(filename_xm,   delimiter=',')
            l_xx = np.genfromtxt(filename_l_xx, delimiter=',')

            # Compute errors
            rmse_j, chi2m_j, ok = nexo.eval_err_srspf(lam[j], eta[:, j], Xi[:, :, j], \
                                        xm, l_xx, ti, tf)
            # Stop if error in error
            if (not ok):
                print('Error calculation failed!')
                exit()

            # Save errors
            rmse [m, j] = rmse_j
            chi2m[m, j] = chi2m_j

        else:

            # File name for generated orbits
            filename = 'results/' + method + '_orbits_' + str(j+1) + '.csv'

            # Proceed if file exists
            if (os.path.isfile(filename)):

                # Read orbits
                orbits = np.genfromtxt(filename, delimiter=',')

                # Elements from file
                sma  = orbits[:, 0]
                ecc  = orbits[:, 1]
                inc  = orbits[:, 2]
                aop  = orbits[:, 3]
                lan  = orbits[:, 4]
                tau  = orbits[:, 5]
                plx  = orbits[:, 6]
                mtot = orbits[:, 7]

                # Periods
                per = np.sqrt(sma**3 / mtot) 
               
                # Mean anomalies at epoch
                mae = 360.0 * (1.0 - tau)

                # Number of sample orbits
                ns = sma.size;

                # Weights
                w = np.full(ns, 1.0/ns)

                # Convert to nonsingular elements
                lams, etas, Xis = nexo.coe2nse(plx, sma, ecc, inc, lan, aop, \
                                        mae, per)

                # Compute errors
                rmse_j, chi2m_j, ok = nexo.eval_err(lam[j], eta[:, j], \
                                        Xi[:, :, j], lams, etas, Xis, w, ti, tf)

                # Stop if error in error
                if (not ok):
                    print('Error calculation failed!')
                    exit()
                
                # Save errors
                rmse [m, j] = rmse_j
                chi2m[m, j] = chi2m_j

            else:

                # Increment number of failures
                nfail = nfail + 1
 
    # Number of failures
    print('Number of Failures: ' + str(nfail))

# Tabulate resulys
err_table = Table()
err_table[' '] = methods
err_table['RMSE'] = np.sqrt(np.nanmean(rmse**2, axis=1))
err_table[r'Mean $\chi^2$'] = np.nanmean(chi2m, axis=1)

# Save table
ascii.write(err_table, 'tables/errs.tex', format='latex', overwrite=True) 
