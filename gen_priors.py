import numpy as np

from EXOSIMS.PlanetPopulation.KeplerLike1 import KeplerLike1
    
# Number of orbits
norb = 10000

# Specs
specs = {}
specs["modules"] = {}
specs["modules"]["PlanetPhysicalModel"] = " "

# Set up planet population model
pop = KeplerLike1(**specs)

# Generate planet parameters
a, e, p, rp = pop.gen_plan_params(norb)

# Semi-major axes
sma = a.value

# Eta values
eta = e / np.sqrt(1 - e**2)

# Root-mean-square of a
rms_a = np.sqrt(np.mean(sma**2))

# Root-mean-square of eta
rms_eta = np.sqrt(np.mean(eta**2))

# Print results
print('RMS of a:   ' + str(rms_a) + ' au')
print('RMS of eta: ' + str(rms_eta)) 
