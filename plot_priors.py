import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import multivariate_normal

from get_priors import get_priors

#-------------------------------------------------------------------------------

mstar = 1.0

std_mstar = 0.1

plx = 10

std_plx = 1

path = 'priors/'

num = 200

#-------------------------------------------------------------------------------

wgt, xm, l_xx = get_priors(mstar, std_mstar, plx, std_plx, path)

ncomp = np.size(wgt)

cov_xx = np.empty((7, 7, ncomp))

for k in range(ncomp):
    cov_xx[:, :, k] = np.matmul(l_xx[:, :, k], np.transpose(l_xx[:, :, k]))

mu_x = np.matmul(xm, wgt)

P_xx = np.tensordot(cov_xx, wgt, (2, 0))

std_x = np.sqrt(np.diag(P_xx))

lb = np.empty(7)
ub = np.empty(7)

ub[0] = 10
ub[1] = 0.5
ub[2] = ub[1]
ub[3] = 50
ub[4] = ub[3]
ub[5] = ub[3]
ub[6] = ub[3]

lb = -ub

#-------------------------------------------------------------------------------

fig, axs = plt.subplots(7, 7, gridspec_kw = {'wspace':0, 'hspace':0})

for i in range(7):

    for j in range(i+1, 7):

        axs[i, j].set_axis_off()

labels = [r'$\lambda$', r'$\eta_1$', r'$\eta_2$', \
         r'$\Xi_{11}$', r'$\Xi_{21}$', r'$\Xi_{12}$', r'$\Xi_{22}$']

plt.rc('font', size=10)

#-------------------------------------------------------------------------------

for i in range(7):

    x = np.linspace(lb[i], ub[i], num=num)

    g = np.empty((num, ncomp))
    for k in range(ncomp):
        g[:, k] = norm.pdf(x, loc=xm[i, k], scale=np.sqrt(cov_xx[i, i, k]))

    y = np.matmul(g, wgt)

    axs[i, i].plot(x, y)

    axs[i, i].set_yticks([])

    axs[i, i].set_xlim([lb[i], ub[i]])

    axs[i, i].set_ylim([0, 1.1 * np.max(y)])

    axs[i, i].set_xticks([lb[i]/2, 0, ub[i]/2])

    axs[i, i].tick_params(axis='x', labelsize=6, labelrotation=45)
    
    if i == 6:
        axs[i, i].set_xlabel(labels[i], fontsize=8)
    else:
        axs[i, i].set_xticks([])


#-------------------------------------------------------------------------------

for i in range(7):

    for j in range(i):

        x_ij = xm[[j, i], :]

        cov_ij = cov_xx[[j, i], :, :]

        cov_ij = cov_ij[:, [j, i], :]

        x = np.linspace(lb[j], ub[j], num=num)
        y = np.linspace(lb[i], ub[i], num=num)

        x, y = np.meshgrid(x, y)

        pos = np.dstack((x, y))

        g = np.empty((num, num, ncomp))
        for k in range(ncomp):
            g[:, :, k] = multivariate_normal.pdf(pos, mean=x_ij[:, k], \
                    cov=cov_ij[:, :, k])

        rho = np.tensordot(g, wgt, (2, 0))

        axs[i, j].contourf(x, y, rho)

        axs[i, j].set_xlim([lb[j], ub[j]])
        axs[i, j].set_ylim([lb[i], ub[i]])

        axs[i, j].set_xticks([lb[j]/2, 0, ub[j]/2])
        axs[i, j].set_yticks([lb[i]/2, 0, ub[i]/2])

        axs[i, j].tick_params(axis='x', labelsize=6, labelrotation=45)
        axs[i, j].tick_params(axis='y', labelsize=6)

        if i == 6:
            axs[i, j].set_xlabel(labels[j], fontsize=8)
        else:
            axs[i, j].set_xticklabels([])

        if j == 0:
            axs[i, j].set_ylabel(labels[i], fontsize=8)
        else:
            axs[i, j].set_yticks([])

#-------------------------------------------------------------------------------

fig.set_size_inches(6, 6)
fig.tight_layout()
plt.savefig('priors/prior_plots.pdf')
