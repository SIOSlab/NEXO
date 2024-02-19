import numpy as np

from sklearn.mixture import GaussianMixture

from EXOSIMS.PlanetPopulation.KeplerLike1 import KeplerLike1

import nexo

def gen_priors(nmix, path):

    # Number of orbits
    norb = 100000

    # Star mass (solar masses)
    mstar = 1.0

    # Parallax
    plx = 1

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
    plx = np.full(norb, plx)

    #-------------------------------------------------------------------------------

    # Convert to nonsingular elements
    lam, eta, xi = nexo.coe2nse(plx, sma, ecc, inc, lan, aop, mae, per)

    # State vectors
    x = np.empty((norb, 7))
    x[:, 0] = lam
    x[:, 1] = eta[0, :]
    x[:, 2] = eta[1, :]
    x[:, 3] = xi[0, 0, :]
    x[:, 4] = xi[1, 0, :]
    x[:, 5] = xi[0, 1, :]
    x[:, 6] = xi[1, 1, :]

    # Generate Gaussian mixture
    gm = GaussianMixture(n_components = nmix, init_params='random_from_data').fit(x)

    # Get parameters
    wgt    = gm.weights_
    xm     = np.transpose(gm.means_)
    cov_xx = np.transpose(np.reshape(gm.covariances_, (-1, 49)))

    # Save
    np.savetxt(path + 'wgt.csv',    wgt,    delimiter=',')
    np.savetxt(path + 'xm.csv',     xm,     delimiter=',')
    np.savetxt(path + 'cov_xx.csv', cov_xx, delimiter=',')
