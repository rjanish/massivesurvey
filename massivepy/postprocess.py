"""
Functions for calculating lambda_R
"""

import functools

import numpy as np

def lam(R,V,sigma,lum):
    """
    Calculates single lambda value for the input provided.
    """
    num = np.average(R*np.abs(V),weights=lum)
    denom = np.average(R*np.sqrt(V**2 + sigma**2),weights=lum)
    return num/denom

def group_bins(bindata,n0=3):
    """
    Group bins into annuli. Does the obvious thing for outer annuli.
    Also groups the center single-fiber bins into annuli of approximately
    equal radius, based on having n0 as the number of bins in the "first"
    (center) annulus, and having the total number of bins enclosed in the nth
    annulus scale like n^2.
    E.g. for n0=3 the total number of bins enclosed goes 3, 12, 27, 48 etc, 
    so the number in each annulus goes 3, 9, 15, 21, etc.
    Note that the final annulus in the center region will have slightly more
    or fewer bins/fibers than this.
    Returns a list of arrays, each containing the bin indexes for one annulus.
    NOTE, this requires the center bins to be sorted!!
    """
    n_singlebins = np.sum(np.isnan(bindata['rmin']))
    n_centerannuli = int(np.rint(np.sqrt(n_singlebins/float(n0))))
    ii_splits = list(np.array(range(1,n_centerannuli))**2 * n0)
    ii_splits.append(n_singlebins)
    jj_annuli = np.nonzero(np.diff(bindata['rmin'][n_singlebins:]))[0]
    ii_splits.extend(n_singlebins+jj_annuli+1)
    return np.split(range(len(bindata)),ii_splits)

def bootstrap(x,y,N=10000):
    """
    Find linear slope of x,y correlation and bootstrap error bars.
    """
    slopes, intercepts = [], []
    M = len(x)
    ii_all = np.arange(M)
    for i in range(N):
        ii = np.random.choice(ii_all,size=M)
        slope, intercept = np.polyfit(x[ii],y[ii],1)
        slopes.append(slope)
        intercepts.append(intercept)
    results = {'slope': np.average(slopes),
               'intercept': np.average(intercepts),
               'slope_err': np.std(slopes),
               'intercept_err': np.std(intercepts)}
    return results
