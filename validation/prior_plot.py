import sys

import numpy as np

from scipy.stats import norm

import matplotlib.pyplot as plt

sys.path.append('..')

import nexo

#-------------------------------------------------------------------------------

nq = 1000

min_per = 0.1

max_per = 10000

std_lam = 0.5

std_eta = 0.15

max_xi  = 300

max_eta = 2

#-------------------------------------------------------------------------------

pxm = 100

mm = 1.0

#-------------------------------------------------------------------------------

min_lam = np.log(min_per)
max_lam = np.log(max_per)

x, w = np.polynomial.legendre.leggauss(nq)

w = w / 2

x = (max_lam + min_lam) / 2 + x * (max_lam - min_lam) / 2

std_xi = pxm * mm**(1.0 / 3) * np.exp(2*x/3 + 4*std_lam**2/9) / np.sqrt(3.0)

#-------------------------------------------------------------------------------

npts = 1000

lam = np.linspace( min_lam, max_lam, npts)
eta = np.linspace(-max_eta, max_eta, npts)
xi  = np.linspace(-max_xi,  max_xi,  npts)

g_eta = norm.pdf(eta, 0.0, std_lam)

mix_g_lam = np.empty((npts, nq))
mix_g_xi  = np.empty((npts, nq))

for j in range(nq):
    mix_g_lam[:, j] = norm.pdf(lam, x[j], std_lam)
    mix_g_xi [:, j] = norm.pdf(xi,  0.0,  std_xi[j])

g_lam = np.matmul(mix_g_lam, w)
g_xi  = np.matmul(mix_g_xi,  w)

#-------------------------------------------------------------------------------

xlabels = [r'$\lambda$', r'$\eta_i$', r'$\Xi_{ij}$ (mas)']

xvals = [lam, eta, xi]

yvals = [g_lam, g_eta, g_xi]

units = ['', '', r'(mas)$^{-1}$']

plt.rc('font', size=10)

fig, axs = plt.subplots(3, 1)

for i in range(3):

    axs[i].plot(xvals[i], yvals[i])
    
    axs[i].set(xlabel = xlabels[i], ylabel = 'Probability Density ' + units[i])

    axs[i].set_ylim(ymin = 0.0)

fig.set_size_inches(6, 6)
fig.tight_layout()
plt.savefig('plots/priors.pdf')
