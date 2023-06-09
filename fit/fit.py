from math import *

import numpy as np

from scipy.stats import norm

from astropy.io import ascii
from astropy.table import Table
from astropy.time import Time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sys
sys.path.append('..')

import nexo

#-------------------------------------------------------------------------------

nq = 1000

min_per = 0.1

max_per = 10000

std_lam = 0.5

std_eta = 0.15

npass = 2

def nexo_fit(fname, ref):

    tref = 2000

    data_file = 'data/meas_' + fname + '.csv'

    table = ascii.read(data_file)

    t = Time(table['imdateobs']).decimalyear

    yri = np.floor(np.amin(t))
    yrf = np.ceil(np.amax(t))

    sep = table['imangsep']
    pa  = table['impa']

    std_sep = np.maximum(
            np.absolute(table['imangseperr1']),
            np.absolute(table['imangseperr2'])
            )

    std_pa = np.maximum(
            np.absolute(table['impaerr1']),
            np.absolute(table['impaerr2'])
            )

    dist = np.mean(table['imdist']) 

    dist_err = np.amax(np.maximum(
            np.absolute(table['imdisterr1']),
            np.absolute(table['imdisterr2'])
            ))

    mm = table['immass'][-1]

    std_m = np.maximum(
            table['immasserr1'][-1],
            table['immasserr2'][-1]
            )

    px = 1 / dist

    std_px = dist_err / dist**2

    nmeas = t.size

    corr_seppa = np.zeros(nmeas)

    z, cov_ww = nexo.seppa2z(sep, pa, std_sep, std_pa, corr_seppa)

    #---------------------------------------------------------------------------

    std_xi = 100 * px

    # Run filter
    xm, l_xx = nexo.mix_filter(npass, nq, px, std_px, mm, std_m, \
            min_per, max_per, std_lam, std_eta, t-tref, z, cov_ww)

    # Confidence
    conf = 0.95
    
    # Compute confidence intervals
    ci_sma, ci_ecc, ci_inc, ci_lan, ci_aop, ci_mae, ci_per, ci_tp = \
            nexo.coe_ci_stroud(xm, l_xx, px, std_px, conf)

    # Table of confidence intervals
    ci_table = Table()
    ci_table['sma'] = ci_sma
    ci_table['ecc'] = ci_ecc 
    ci_table['inc'] = ci_inc
    ci_table['lan'] = ci_lan
    ci_table['aop'] = ci_aop
    ci_table['mae'] = ci_mae
    ci_table['per'] = ci_per
    ci_table['tp']  = ci_tp + tref

    # Save confidence intervals
    ascii.write(ci_table, 'results/coe_ci_' + fname + '.csv', \
            overwrite=True, format='csv')

    keys = ['sma', 'ecc', 'inc', 'lan', 'aop', 'per', 'tp']
    
    #---------------------------------------------------------------------------
    
    ci_latex = Table()

    ci_latex['Element'] = [r'$a$', r'$e$', r'$I$', r'$\Omega$', \
                    r'$\omega$', r'$M_0$', r'$P$']

    ci_latex['Units'] = ['(au)', '--', r'($^\circ$)', r'($^\circ$)', \
                    r'($^\circ$)', r'($^\circ$)', '(yr)']

    means = []
    errbs = []
    pm    = []

    keys_table = ['sma', 'ecc', 'inc', 'lan', 'aop', 'mae', 'per']

    for key in keys_table:
        ci_k = ci_table[key]
        means.append(ci_k[1])
        errbs.append(ci_k[2] - ci_k[1])
        pm.append(r'$\pm$')

    ci_latex['Mean']  = means
    ci_latex[' ']     = pm
    ci_latex['Error'] = errbs

    ci_latex.round(1)

    ascii.write(ci_latex, 'results/coe_ci_' + fname + '.tex', format='latex', \
            overwrite=True) 
    
    #---------------------------------------------------------------------------

    plt.rc('font', size=10)
    
    t_p = np.linspace(yri, yrf, 1000)

    zm, cov_zz = nexo.predict_z(xm, l_xx, t_p-tref)

    sep_p, pa_p, std_sep_p, std_pa_p, corr_seppa_p = nexo.z2seppa(zm, cov_zz)

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(t_p, sep_p, 'b-', label='Mean Fit')
    axs[0].plot(t_p, sep_p - 3*std_sep_p, 'b:', label=r'$3\sigma$ Bound')
    axs[0].plot(t_p, sep_p + 3*std_sep_p, 'b:', label='_nolegend_')
    axs[0].errorbar(t, sep, yerr=3*std_sep, label='Measurements', fmt='rs') 
    axs[0].legend()
    axs[0].set(xlabel = 'Epoch', ylabel = r'$\rho$ (as)')

    axs[1].plot(t_p, pa_p, 'b-', label='Mean Fit')
    axs[1].plot(t_p, pa_p - 3*std_pa_p, 'b:', label=r'$3\sigma$ Bound')
    axs[1].plot(t_p, pa_p + 3*std_pa_p, 'b:', label='_nolegend_')
    axs[1].errorbar(t, pa, yerr=3*std_pa, label='Measurements', fmt='rs') 
    axs[1].legend()
    axs[1].set(xlabel = 'Epoch', ylabel = r'$\theta$ ($^\circ$)')

    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
 
    ymin, ymax = axs[1].get_ylim()
    axs[1].set_ylim([max(ymin, -90.0), min(ymax, 450.0)])

    fig.set_size_inches(6, 8)
    fig.tight_layout()
    plt.savefig('results/seppa_' + fname + '.pdf')

    #---------------------------------------------------------------------------

    data_file = 'data/ref_' + fname + '.csv'

    ref_table = ascii.read(data_file)
    
    keys = ['sma', 'ecc', 'inc', 'lan', 'aop', 'per', 'tp']

    names = [r'$a$ (au)', r'$e$', r'$I$ ($^\circ$)', r'$\Omega$ ($^\circ$)', \
             r'$\omega$ ($^\circ$)', r'$P$ (yr)', r'$t_p$ (yr)'] 

    coe_min = [0, 0, 0, 0, 0, 0, tref]

    infty = float('inf')

    coe_max = [inf, 1, 180, 180, 360, inf, inf]

    fig, axs = plt.subplots(2, 4)

    for j in range(2):

        for k in range(4):

            i = 4 * j + k

            if (i < 7):

                vals = ref_table[keys[i]]

                x = ref

                y = vals[1] 

                valerr = [[abs(vals[0]-y)], [abs(vals[2]-y)]]

                axs[j, k].errorbar(x, y, yerr=valerr, fmt='bo')

                vals = ci_table[keys[i]]

                x = 'NEXO'

                y = vals[1] 

                valerr = [[abs(vals[0]-y)], [abs(vals[2]-y)]]

                axs[j, k].errorbar(x, y, yerr=valerr, fmt='rs')

                axs[j, k].set(ylabel = names[i])
        
                axs[j, k].margins(0.5)

                axs[j, k].tick_params('x', labelrotation=45)

                ymin, ymax = axs[j, k].get_ylim()

                axs[j, k].set_ylim([max(ymin, coe_min[i]), min(ymax, coe_max[i])])

            else:

                axs[j, k].axis('off')

    fig.set_size_inches(6, 8)
    fig.tight_layout()
    plt.savefig('results/coe_' + fname + '.pdf')

#-------------------------------------------------------------------------------

fname = 'GJ_504_b'

ref = 'OFTI'

nexo_fit(fname, ref)

#-------------------------------------------------------------------------------

fname = 'beta_Pic_b'

ref = 'MCMC'

nexo_fit(fname, ref)
