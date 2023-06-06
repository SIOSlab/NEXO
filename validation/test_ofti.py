import numpy as np
import time

import orbitize
from orbitize import driver

def test_ofti(j):

    # Mean parallax (mas)
    plxm = 100

    # Standard deviation of parallax (mas)
    std_plx = 0.1

    # Mean total mass (solar masses)
    mm = 1.0

    # Standard deviation of total mass
    std_m = 0.1

    # Random generator seed
    np.random.seed(j)

    # Number of orbits to generate
    nsamp = 10000
        
    # Print status
    print('***************************************************************')
    print('Running case ' + str(j) + ' using OFTI')
    print('***************************************************************')

    # File with measurements
    file_in = 'gen/meas_' + str(j) + '.csv' 

    # Set up orbitize_driver
    orbitize_driver = orbitize.driver.Driver(
            file_in,          # data file
            'OFTI',           # choose from: ['OFTI', 'MCMC']
            1,                # number of planets in system 
            mm,               # total mass [M_sun]
            plxm,             # system parallax [mas] 
            mass_err = std_m, # mass error [M_sun]
            plx_err = std_plx # parallax error [mas]
            )

    # Start timer
    ti = time.time()

    # Run sampler
    orbits = orbitize_driver.sampler.run_sampler(nsamp)

    # Stop timer
    tf = time.time()

    # Run time (seconds)
    trun = tf - ti
    
    # Save generated orbits
    np.savetxt('results/OFTI_orbits_' + str(j) + '.csv', orbits, delimiter=',')

    # Save run time
    np.savetxt('results/OFTI_trun_' + str(j) + '.csv', np.full(1, trun))
