import sys
import os

from math import *

from joblib import Parallel, delayed

import numpy as np
import scipy as sp
import time

from astropy.io import ascii
from astropy.table import Table

sys.path.append('..')

import nexo

#-------------------------------------------------------------------------------

nq = 1000

min_per = 0.1

max_per = 10000

std_lam = 0.5

std_eta = 0.15

npass = 2

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
lam, eta, xi = nexo.coe2nse(coe_tru['plx'], coe_tru['sma'], coe_tru['ecc'], \
        coe_tru['inc'], coe_tru['lan'], coe_tru['aop'], coe_tru['mae'], \
        coe_tru['per']) 

# Number of cases
norb = len(coe_tru)

# Reference epoch (MJD)
ref_epoch = 58849.0

#-------------------------------------------------------------------------------

# Confidence
conf = 0.95

# Fit one orbit
def fit_orbit(j):

    # Print status
    print('Running case ' + str(j+1) + ' of ' + str(norb))

    # Start timer
    ti = time.time()

    # Read measurements
    meas_table = ascii.read('gen/meas_' + str(j+1) + '.csv')

    # Measurement times
    t = (meas_table['epoch'] - ref_epoch) / 365.25
    
    # Filter-friendly measurements
    z, cov_ww = nexo.radec2z(meas_table['raoff'], meas_table['decoff'], \
            meas_table['raoff_err'], meas_table['decoff_err'], \
            meas_table['radec_corr'])

    # Run filter
    xm, l_xx = nexo.mix_filter(npass, nq, plxm, std_plx, mm, std_m, \
            min_per, max_per, std_lam, std_eta, t, z, cov_ww)

    # Compute confidence intervals
    ci_sma, ci_ecc, ci_inc, ci_lan, ci_aop, ci_mae, ci_per, ci_tp = \
            nexo.coe_ci_stroud(xm, l_xx, plxm, std_plx, conf)

    # Stack confidence intervals
    ci = np.stack([ci_sma, ci_ecc, ci_inc, ci_lan, ci_aop, ci_mae, ci_per], -1)

    # Save results
    np.savetxt('results/NEXO_ci_' + str(j+1) + '.csv', ci, delimiter=',')

    # Stop timer
    tf = time.time()

    # Run time (seconds)
    trun = tf - ti

    # Save run time
    np.savetxt('results/NEXO_trun_' + str(j+1) + '.csv', np.full(1, trun))

# Run tests
for j in range(norb):
    fit_orbit(j)

#Parallel(n_jobs=os.cpu_count())(delayed(fit_orbit)(j) for j in range(norb))
