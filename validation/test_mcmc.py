import numpy as np
import os
import time

import orbitize
from orbitize import driver

def test_mcmc(j):

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
       
    # MCMC parameters
    num_temps = 20
    num_walkers = 1000
    num_threads = os.cpu_count()
    thin = 2
    total_orbits = nsamp * thin
    burn_steps = 100

    # Print status
    print('***************************************************************')
    print('Running case ' + str(j) + ' using MCMC')
    print('***************************************************************')

    # File with measurements
    file_in = 'gen/meas_' + str(j) + '.csv' 

    # Set up orbitize_driver
    orbitize_driver = orbitize.driver.Driver(
            file_in,           # data file
            'MCMC',            # choose from: ['MCMC', 'MCMC']
            1,                 # number of planets in system 
            mm,                # total mass [M_sun]
            plxm,              # system parallax [mas] 
            mass_err = std_m,  # mass error [M_sun]
            plx_err = std_plx, # parallax error [mas]
            mcmc_kwargs = {
                'num_temps': num_temps,
                'num_walkers': num_walkers,
                'num_threads': num_threads
                }
            )

    # Start timer
    ti = time.time()

    # Run sampler
    res = orbitize_driver.sampler.run_sampler(
            total_orbits,
            burn_steps = burn_steps,
            thin = thin
            )

    # Stop timer
    tf = time.time()

    # Run time (seconds)
    trun = tf - ti
    
    # Generated orbits
    orbits = orbitize_driver.sampler.results.post

    # Save generated orbits
    np.savetxt('results/MCMC_orbits_' + str(j) + '.csv', orbits, delimiter=',')

    # Save run time
    np.savetxt('results/MCMC_trun_' + str(j) + '.csv', np.full(1, trun))
