"""
Functions for calculating lambda_R
"""

import functools

import numpy as np

def calc_lambda(R,V,sigma,flux,idiff):
    """
    Calculate lambda at each R.
    Requires input to be sorted.
    Gives both differential and cumulative lambda.
    """
    nbins = len(R)
    if not len(V)==nbins and len(sigma)==nbins and len(flux)==nbins:
        raise Exception('u broke it.')
    dt = {'names':['R','num','denom','lam','diff_lam'],
          'formats':5*[np.float64]}
    output = np.zeros(nbins,dtype=dt)
    for i in range(nbins):
        avg = functools.partial(np.average,weights=flux[:i+1])
        output['num'][i] = avg(R[:i+1]*np.abs(V[:i+1]))
        output['denom'][i] = avg(R[:i+1]*np.sqrt(V[:i+1]**2+sigma[:i+1]**2))
        output['lam'][i] = output['num'][i]/output['denom'][i]
    for i1,i2 in zip([0]+idiff[:-1],idiff):
        avg = functools.partial(np.average,weights=flux[i1:i2+1])
        num = avg(R[i1:i2+1]*np.abs(V[i1:i2+1]))
        denom = avg(R[i1:i2+1]*np.sqrt(V[i1:i2+1]**2+sigma[i1:i2+1]**2))
        output['diff_lam'][i2] = num/denom
    return output

def calc_sigma(R,sigma,flux,idiff):
    """
    Calculate flux-weighted average sigma within R
    """
    output = np.zeros(R.shape,dtype={'names':['R','sig','diff_sig'],
                                     'formats':3*['f8']})
    ii = np.argsort(R)
    output['R'] = R[ii]
    output['sig'] = np.cumsum(sigma[ii]*flux[ii])/np.cumsum(flux[ii])
    for i1,i2 in zip([0]+idiff[:-1],idiff):
        output['diff_sig'][i2]=np.average(sigma[i1:i2+1],weights=flux[i1:i2+1])
    return output
