import numpy as np

import nexo

def get_priors(mstar, std_mstar, plx, std_plx, path):

    # Read files
    wgt       = np.genfromtxt(path + 'wgt.csv',     delimiter=',')
    xm_0      = np.genfromtxt(path + 'xm.csv',      delimiter=',')
    cov_xx_0  = np.genfromtxt(path + 'cov_xx.csv',  delimiter=',')

    # Reshape covariances
    cov_xx_0 = np.reshape(cov_xx_0, (7, 7, -1))

    # Scale priors
    xm, l_xx = nexo.scale_mix(xm_0, cov_xx_0, mstar, std_mstar, plx, std_plx)

    # Return results
    return wgt, xm, l_xx
