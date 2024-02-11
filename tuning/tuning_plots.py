import numpy as np

import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table

rmse  = np.genfromtxt('tables/rmse.csv',  delimiter=',')
chi2m = np.genfromtxt('tables/chi2m.csv', delimiter=',')
alpha  = np.genfromtxt('tables/alpha.csv', delimiter=',')

name = ['rmse', 'chi2m']

label = [r'$\varepsilon_\mathrm{RMS}$ (mas)', r'$\overline{\chi^2}$']

val = [rmse, chi2m]

for i in range(2):

    plt.rc('font', size=10)

    plt.figure(figsize = (6, 4))

    plt.yscale('log')

    plt.plot(alpha, val[i])

    plt.xlabel("Alpha")
    plt.ylabel(label[i])

    plt.tight_layout()
    plt.savefig("plots/" + name[i] + "_npq.pdf")
    plt.close()
