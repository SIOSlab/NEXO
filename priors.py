import time

import numpy as np

from EXOSIMS.PlanetPopulation.KeplerLike1 import KeplerLike1

import nexo

#-------------------------------------------------------------------------------

# Random generator seed
np.random.seed(707)

# Number of orbits
norb = 10000

# Star mass (solar masses)
mstar = 1.0

# Mean parallax (mas)
plxm = 1

# Standard deviation of parallax (mas)
std_plx = 0

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

# Nonsingular elements
lam, eta, xi = nexo.coe2nse(plx, sma, ecc, inc, lan, aop, mae, per)

# Mean of lambda
lamm = np.mean(lam)

# Standard deviations
std_lam = np.std(lam)
std_eta = np.sqrt(np.mean(eta**2))
std_xi  = np.sqrt(np.mean(xi**2))

# Display results
print("lamm    = " + str(lamm))
print("std_lam = " + str(std_lam))
print("std_eta = " + str(std_eta))
print("std_xi  = " + str(std_xi))
