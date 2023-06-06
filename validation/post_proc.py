import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table

sys.path.append('..')

import nexo

import conf_int

#-------------------------------------------------------------------------------

def plot_ci(tru, ci, xlbl, ylbl, xrange, title, file):

    # Figure size
    plt.figure(figsize = (6, 3))
    
    # Font size
    plt.rc('font', size=10) 

    # Median values
    median = ci[1, :]

    # Error bars
    err = np.vstack((ci[1, :] - ci[0, :], ci[2, :] - ci[1, :]))

    # Plot error bars
    plt.errorbar(tru, median, yerr=err, fmt='bs')

    # Plot line x = y
    x = np.linspace(xrange[0], xrange[1])
    y = x
    plt.plot(x, y, 'r')

    # Title
    plt.title(title, {'fontsize': 10, 'fontweight': 'bold'})

    # Set axis labels
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)

    # Set axis limits
    plt.xlim((xrange[0], xrange[1]))
    plt.ylim((xrange[0], xrange[1]))

    # Save figure
    plt.tight_layout()
    plt.savefig(file)

    # Close figure
    plt.close()

#-------------------------------------------------------------------------------

# Read classical parameters
coe_tru = ascii.read('gen/coe_tru.csv')

# Number of cases
norb = len(coe_tru)

# Element names
elname = ['sma', 'ecc', 'inc', 'lan', 'aop', 'mae', 'per']

# Element symbols & units
elsu = [r'$a$ (au)', r'$e$', r'$I$ ($^\circ$)', r'$\Omega$ ($^\circ$)', \
        r'$\omega$ ($^\circ$)', r'$M_0$ ($^\circ$)', r'$P$ (yr)']

# Element maximum values
sma_max = 60
per_max = 400
elmax = [sma_max, 1.0, 180.0, 180.0, 360.0, 360.0, per_max]

#-------------------------------------------------------------------------------

# Confidence
conf = 0.95

# Methods
methods = ['NEXO', 'MCMC', 'OFTI']

# Tables
perc_table = Table()
rmse_table = Table()

# Table row labels
perc_table[' '] = [r'$a$', r'$e$', r'$I$', r'$\Omega$', \
        r'$\omega$', r'$M_0$', r'$P$']
rmse_table[' '] = [r'$a$ (\%)', r'$e$ (\%)', r'$I$ ($^\circ$)', \
        r'$\Omega$ ($^\circ$)', r'$\omega$ ($^\circ$)', r'$M_0$ ($^\circ$)', \
        r'$P$ (\%)']

# Other labels
rmse_labels = [r'$a$ (%)', r'$e$ (%)', r'$I$ ($^\circ$)', \
        r'$\Omega$ ($^\circ$)', r'$\omega$ ($^\circ$)', r'$M_0$ ($^\circ$)', \
        r'$P$ (%)']
perc_labels = [r'$a$', r'$e$', r'$I$', r'$\Omega$', \
        r'$\omega$', r'$M_0$', r'$P$']

# Array for average run times
trun = np.empty(3)

# Post-process for each method
for m in range(3):

    # Method name
    method = methods[m]

    # Print status
    print('Post-processing for ' + method)

    # Arrays for confidence intervals
    sma_ci = np.full((3, norb), np.nan)
    ecc_ci = np.full((3, norb), np.nan)
    inc_ci = np.full((3, norb), np.nan)
    lan_ci = np.full((3, norb), np.nan)
    aop_ci = np.full((3, norb), np.nan)
    mae_ci = np.full((3, norb), np.nan)
    per_ci = np.full((3, norb), np.nan)

    # Number of failed fits
    nfail = 0
    
    # Array of run times
    truns = np.full(norb, np.nan)

    # Iterate over orbits
    for j in range(norb):

        # Print status
        print('    Orbit ' + str(j+1) + ' of ' + str(norb))

        # Read elements
        if (method == 'NEXO'):

            # File name for confidence intervals
            filename = 'results/NEXO_ci_' + str(j+1) + '.csv'

            # Read file
            ci = np.genfromtxt(filename, delimiter=',')

            # Confidence intervals
            sma_ci[:, j] = ci[:, 0]
            ecc_ci[:, j] = ci[:, 1]
            inc_ci[:, j] = ci[:, 2]
            lan_ci[:, j] = ci[:, 3]
            aop_ci[:, j] = ci[:, 4]
            mae_ci[:, j] = ci[:, 5]
            per_ci[:, j] = ci[:, 6]

            # Read run time
            filename = 'results/NEXO_trun_' + str(j+1) + '.csv'
            truns[j] = np.genfromtxt(filename) 

        else:

            # File name for generated orbits
            filename = 'results/' + method + '_orbits_' + str(j+1) + '.csv'

            # Proceed if file exists
            if (os.path.isfile(filename)):

                # Read orbits
                orbits = np.genfromtxt(filename, delimiter=',')

                # Elements from file
                sma  = orbits[:, 0]
                ecc  = orbits[:, 1]
                inc  = orbits[:, 2]
                aop  = orbits[:, 3]
                lan  = orbits[:, 4]
                tau  = orbits[:, 5]
                plx  = orbits[:, 6]
                mtot = orbits[:, 7]

                # Element conversions
                inc = np.degrees(inc)
                lan = ((2 * np.degrees(lan)) % 360) / 2
                aop = np.degrees(aop)
                mae = 360 * (1 - tau)
                per = np.sqrt(sma**3 / mtot)

                # Confidence intervals
                sma_ci[:, j] = nexo.lin_ci_eqw(sma, conf) 
                ecc_ci[:, j] = nexo.lin_ci_eqw(ecc, conf) 
                inc_ci[:, j] = nexo.cir_ci_eqw(inc, conf) 
                lan_ci[:, j] = nexo.cir_ci_eqw(lan, conf) 
                aop_ci[:, j] = nexo.cir_ci_eqw(aop, conf) 
                mae_ci[:, j] = nexo.cir_ci_eqw(mae, conf) 
                per_ci[:, j] = nexo.lin_ci_eqw(per, conf) 

                # Read run time
                filename = 'results/' + method + '_trun_' + str(j+1) + '.csv'
                truns[j] = np.genfromtxt(filename)

            else:

                # Increment number of failures
                nfail = nfail + 1
 
    # Element confidence intervals
    elci = [sma_ci, ecc_ci, inc_ci, lan_ci, aop_ci, mae_ci, per_ci]

    # Plot confidence for classical elements
    for i in range(7):

        tru = coe_tru[elname[i]][0:norb]

        ci = elci[i]

        xlbl = r'True '      + elsu[i]
        ylbl = r'Estimated ' + elsu[i]

        xrange = [0, elmax[i]]
    
        title = method

        file = 'plots/' + method + '_ci_' + elname[i] + '.pdf'

        plot_ci(tru, ci, xlbl, ylbl, xrange, title, file)

    # Number of NaN values in confidence intervals
    tot_nan = np.count_nonzero(np.isnan(np.hstack((sma_ci, ecc_ci, \
            inc_ci, lan_ci, aop_ci, mae_ci, per_ci))))
    print('Total NaN: ' + str(tot_nan))

    # Percentages of elements inside confidence intervals
    perc = np.empty(7)
    perc[0] = conf_int.lin_perc (sma_ci, coe_tru['sma'])
    perc[1] = conf_int.lin_perc (ecc_ci, coe_tru['ecc'])
    perc[2] = conf_int.lin_perc (inc_ci, coe_tru['inc'])
    perc[3] = conf_int.lin_perc (lan_ci, coe_tru['lan'])
    perc[4] = conf_int.circ_perc(aop_ci, coe_tru['aop'])
    perc[5] = conf_int.circ_perc(mae_ci, coe_tru['mae'])
    perc[6] = conf_int.lin_perc (per_ci, coe_tru['per'])
    perc_table[method] = perc

    # Root-mean-square errors
    rmse = np.empty(7)
    rmse[0] = conf_int.rel_rmse (sma_ci, coe_tru['sma'])
    rmse[1] = conf_int.rel_rmse (ecc_ci, coe_tru['ecc'])
    rmse[2] = conf_int.abs_rmse (inc_ci, coe_tru['inc'])
    rmse[3] = conf_int.abs_rmse (lan_ci, coe_tru['lan'])
    rmse[4] = conf_int.circ_rmse(aop_ci, coe_tru['aop'])
    rmse[5] = conf_int.circ_rmse(mae_ci, coe_tru['mae'])
    rmse[6] = conf_int.rel_rmse (per_ci, coe_tru['per'])
    rmse_table[method] = rmse

    # Average run time
    trun[m] = np.nanmean(truns) 

# Table of average run times
trun_table = Table()
trun_table['Method'] = methods
trun_table['Average Run Time (s)'] = trun

# Round table values
perc_table.round(1)
rmse_table.round(1)
trun_table.round(1)

print(perc_table)
print(rmse_table)
print(trun_table)

# Save percentages table
ascii.write(perc_table, 'tables/perc.tex', format='latex', overwrite=True) 
ascii.write(rmse_table, 'tables/rmse.tex', format='latex', overwrite=True) 
ascii.write(trun_table, 'tables/trun.tex', format='latex', overwrite=True)

#-------------------------------------------------------------------------------

# Bar colors
colors = ['red', 'blue', 'gray'] 

# Font size
plt.rc('font', size=10) 
    
# Figure size
plt.figure(figsize = (12, 4))

# Subplots
fig, axs = plt.subplots(2, 4)

# Plot RMSE for classical elements
for row in range(2):
    for col in range(4):

        i = 4 * row + col

        if (i < 7):

            # RMSE values
            rmse = np.empty(3)
            for j in range(3):
                rmse[j] = rmse_table[methods[j]][i]

            # Bar plot
            axs[row, col].bar(methods, rmse, color=colors)

            # Rotate x labels
            axs[row, col].tick_params('x', labelrotation=45)
        
            # Label y-axis
            axs[row, col].set(ylabel = rmse_labels[i])

        else:

                axs[row, col].axis('off')

# Save figure
plt.tight_layout()
plt.savefig('plots/RMSE.pdf')

# Close figure
plt.close()

#-------------------------------------------------------------------------------

# Font size
plt.rc('font', size=10)

# Figure size
plt.figure(figsize = (20, 3))

# Subplots
fig, axs = plt.subplots(3, 7)

# Plots
for row in range(3):
    for col in range(7):

        # Label columns
        if (row == 0):
            axs[row, col].set_title(perc_labels[col])

        # Label rows
        if (col == 0):
            axs[row, col].set_ylabel(methods[row])

        # Percentage inside of interval
        perc_in = perc_table[methods[row]][col]
        
        # Plot values
        sizes  = [perc_in, 100 - perc_in]

        # Colors
        colors = ['white', 'red']

        # Pie plot
        axs[row, col].pie(sizes, startangle=90, colors=colors, \
                wedgeprops = {"edgecolor" : "black"})

# Save figure
plt.tight_layout()
plt.savefig('plots/pies.pdf')

# Close figure
plt.close()

#-------------------------------------------------------------------------------

# Bar colors
colors = ['red', 'blue', 'gray'] 

# Font size
plt.rc('font', size=10) 
    
# Figure size
plt.figure(figsize = (4, 4))
            
# Bar plot
plt.bar(methods, trun, color=colors)

# Set to log scale
plt.yscale("log")

# Axis label
plt.ylabel("Average Run Time (s)")

# Save figure
plt.tight_layout()
plt.savefig('plots/trun.pdf')

# Close figure
plt.close()
