import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from astropy.io import ascii
from astropy.table import Table

rmse  = np.genfromtxt('tables/rmse.csv',  delimiter=',')
chi2m = np.genfromtxt('tables/chi2m.csv', delimiter=',')
npass = np.genfromtxt('tables/npass.csv', delimiter=',')
nq    = np.genfromtxt('tables/nq.csv',    delimiter=',')

name = ['rmse', 'chi2m']

label = [r'$\varepsilon_\mathrm{RMS}$ (mas)', r'$\overline{\chi^2}$']

val = [rmse, chi2m]

for i in range(2):

    plt.rc('font', size=10)

    plt.figure(figsize = (6, 4))

    plt.xscale('log')

    heatmap = plt.pcolormesh(nq, npass, val[i], \
                norm=colors.LogNorm(val[i].min(), val[i].max()))

    colorbar = plt.colorbar(heatmap)

    colorbar.set_label(label[i])
    
    plt.xlabel("Number of Mixture Components")
    plt.ylabel("Number of Passes")

    plt.xticks(nq)
    plt.yticks(npass)

    plt.tight_layout()
    plt.savefig("plots/" + name[i] + "_npq.pdf")
    plt.close()
