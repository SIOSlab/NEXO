import numpy as np

import orbitize.kepler

from astropy.io import ascii
from astropy.table import Table

from EXOSIMS.PlanetPopulation.KeplerLike1 import KeplerLike1

import time

#-------------------------------------------------------------------------------

# Random generator seed
np.random.seed(303)

# Number of orbits
norb = 100

# Number of measurements
nmeas = 5

# Measurement time span (yr)
tspan = 2

# Measurement error standard deviation (mas)
sigerr = 5

# Star mass (solar masses)
mstar = 1.0

# Mean parallax (mas)
plxm = 100

# Standard deviation of parallax (mas)
std_plx = 0.1

#-------------------------------------------------------------------------------

# Specs
specs = {}
specs["modules"] = {}
specs["modules"]["PlanetPhysicalModel"] = " "

# Set up planet population model
pop = KeplerLike1(**specs)

# Generate planet parameters
a, e, p, rp = pop.gen_plan_params(norb)

# Planet mass (solar masses)
mp = pop.PlanetPhysicalModel.calc_mass_from_radius(rp).to("solMass").value

# Total mass (solar masses) - assuming 1 solar mass for star
mtot = mstar + mp 

# Semi-major axes
sma = a.value

# Eccentricities
ecc = e

# Inclinations (uniform cosine)
inc = np.degrees(np.arccos(np.random.uniform(-1.0, 1.0, norb)))

# Longitudes of ascending nodes (uniform circular)
lan = np.random.uniform(0.0, 180.0, norb)

# Arguments of periapsis (uniform circular)
aop = np.random.uniform(0.0, 360.0, norb)

# Mean anomalies at epoch (uniform circular)
mae = np.random.uniform(0.0, 360.0, norb)

# Periods
per = np.sqrt(sma**3 / mtot) 

# Parallaxes
plx = np.random.normal(plxm, std_plx, norb)

#-------------------------------------------------------------------------------

# Reference epoch (MJD)
ref_epoch = 58849.0

# Tau parameters
tau = 1 - ((mae + 360) % 360) / 360

# Times of periapse passage
tp = tau * per

#-------------------------------------------------------------------------------

# Save classical elements

coe_table = Table()

coe_table['sma'] = sma
coe_table['ecc'] = ecc
coe_table['inc'] = inc
coe_table['lan'] = lan
coe_table['aop'] = aop
coe_table['mae'] = mae
coe_table['per'] = per
coe_table['plx'] = plx

coe_table['tau'] = tau
coe_table['mtot'] = mtot
coe_table['tp'] = tp

ascii.write(coe_table, 'gen/coe_tru.csv', overwrite=True, format='csv')

#-------------------------------------------------------------------------------
# Measurement times (yr)
t = np.linspace(0, tspan, nmeas)

# Measurement epochs (MJD)
epoch = ref_epoch + 365.25 * t
 
# Measurements using orbitize
raoff, decoff, vz = orbitize.kepler.calc_orbit(epoch, sma, ecc, \
        np.radians(inc), np.radians(aop), np.radians(lan), tau, plx, mtot)

# Add noise
raoff  = raoff  + np.random.normal(0.0, sigerr, (nmeas, norb))
decoff = decoff + np.random.normal(0.0, sigerr, (nmeas, norb))

# Save measurements for each case
for j in range(norb):

    meas_table = Table()
    
    meas_table['epoch'] = epoch

    meas_table['object'] = np.ones(nmeas, np.int8)

    meas_table['raoff']  = raoff [:, j]
    meas_table['decoff'] = decoff[:, j]

    meas_table['raoff_err']  = np.full(nmeas, sigerr)
    meas_table['decoff_err'] = np.full(nmeas, sigerr)

    meas_table['radec_corr'] = np.zeros(nmeas)
    
    filename = 'gen/meas_' + str(j+1) + '.csv'

    ascii.write(meas_table, filename, overwrite=True, format='csv')
