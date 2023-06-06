import numpy as np
import scipy as sp

#-------------------------------------------------------------------------------

def circ_diff(x, y):

    # Angular difference x - y
    return ((x - y + 540) % 360) - 180

#-------------------------------------------------------------------------------

def circ_dist(x, y):

    # Absolute value of angular difference
    return np.absolute(circ_diff(x, y))

#-------------------------------------------------------------------------------

def lin_perc(ci, tru):

    # Percentage of cases inside intervals
    return (100.0 * np.count_nonzero((tru > ci[0, :]) & (tru < ci[2, :]))) \
            / np.count_nonzero(~np.any(np.isnan(ci), 0))

#-------------------------------------------------------------------------------

def abs_rmse(ci, tru):

    # Absolute RMSE
    return np.sqrt(np.nanmean((ci[1, :] - tru)**2))

#-------------------------------------------------------------------------------

def rel_rmse(ci, tru):

    # Relative RMSE
    return 100 * np.sqrt(np.nanmean(((ci[1, :] - tru) / tru)**2))

#-------------------------------------------------------------------------------

def circ_perc(ci, tru):

    # Shift true values
    stru = ci[1, :] + circ_diff(tru, ci[1, :])

    # Percentage of cases inside intervals
    return (100.0 * np.count_nonzero((stru > ci[0, :]) & (stru < ci[2, :]))) \
            / np.count_nonzero(~np.any(np.isnan(ci), 0))

#-------------------------------------------------------------------------------

def circ_rmse(ci, tru):

    # Circular RMSE
    return np.sqrt(np.nanmean(circ_diff(ci[1, :], tru)**2))
