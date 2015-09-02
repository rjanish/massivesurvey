"""
Functions for calculating lambda_R
"""

import functools

import numpy as np

def calc_lambda(R,V,sigma,flux,Vnorm='fluxavg'):
    nbins = len(R)
    if not len(V)==nbins and len(sigma)==nbins and len(flux)==nbins:
        raise Exception('u broke it.')
    # sort by radius
    ii = np.argsort(R)
    R = R[ii]
    V = V[ii]
    sigma = sigma[ii]
    flux = flux[ii]
    dt = {'names':['R','Vavg','RVavg','m2avg','Rm2avg','lam',
                   'V','sigma','flux'],
          'formats':9*[np.float64]}
    output = np.zeros(nbins,dtype=dt)
    output['R'] = R
    output['V'] = V
    output['sigma'] = sigma
    output['flux'] = flux
    for i in range(nbins):
        avg = functools.partial(np.average,weights=flux[:i+1])
        output['Vavg'][i] = avg(np.abs(V[:i+1]))
        output['RVavg'][i] = avg(R[:i+1]*np.abs(V[:i+1]))
        output['m2avg'][i] = avg(np.sqrt(V[:i+1]**2+sigma[:i+1]**2))
        output['Rm2avg'][i]=avg(R[:i+1]*np.sqrt(V[:i+1]**2+sigma[:i+1]**2))
        output['lam'][i] = output['RVavg'][i]/output['Rm2avg'][i]
    return output

def calc_sigma(R,sigma,flux):
    """
    Calculate flux-weighted average sigma within R
    """
    output = np.zeros(R.shape,dtype={'names':['R','sig'],
                                     'formats':2*['f8']})
    ii = np.argsort(R)
    output['R'] = R[ii]
    output['sig'] = np.cumsum(sigma[ii]*flux[ii])/np.cumsum(flux[ii])
    return output
