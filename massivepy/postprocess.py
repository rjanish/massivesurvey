"""
Functions for calculating lambda_R
"""

import functools

import numpy as np

def calc_lambda(R,Vraw,sigma,flux,Vnorm='fluxavg'):
    nbins = len(R)
    if not len(Vraw)==nbins and len(sigma)==nbins and len(flux)==nbins:
        raise Exception('u broke it.')
    if Vnorm=='fluxavg':
        Voffset = np.average(Vraw,weights=flux)
    elif Vnorm=='no_offset':
        Voffset = 0
    elif Vnorm=='median':
        Voffset = np.median(Vraw)
    else:
        raise Exception('u broke it.')
    V = Vraw - Voffset
    # sort by radius
    ii = np.argsort(R)
    R = R[ii]
    V = V[ii]
    Vraw = Vraw[ii]
    sigma = sigma[ii]
    flux = flux[ii]
    dt = {'names':['R','Vavg','RVavg','m2avg','Rm2avg','lam',
                   'V','Vraw','sigma','flux'],
          'formats':10*[np.float64]}
    output = np.zeros(nbins,dtype=dt)
    output['R'] = R
    output['Vraw'] = Vraw
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

'''
def calc_lambda_Jennystyle(R,Vraw,sigma,flux,style='c'):
    nbins = len(R)
    if not len(Vraw)==nbins and len(sigma)==nbins and len(flux)==nbins:
        raise Exception('u broke it.')
    newR = np.arange(int(min(R))+1,int(max(R))+2,1)
    Voffset = np.median(Vraw)
    V = Vraw - Voffset
    ii = np.argsort(R)
    R = R[ii]
    V = V[ii]
    sigma = sigma[ii]
    flux = flux[ii]
    #flux = np.max(flux)*np.ones(flux.shape)
    dt = {'names':['R','Vavg','RVavg','m2avg','Rm2avg','lam',
                   'V','Vraw','sigma','flux'],
          'formats':10*[np.float64]}
    output = np.zeros(len(newR),dtype=dt)
    output['R'] = newR
    output['V'] = np.interp(newR,R,V)
    output['Vraw'] = np.interp(newR,R,Vraw)
    output['sigma'] = np.interp(newR,R,sigma)
    output['flux'] = np.interp(newR,R,flux)
    iused = -1
    for j in range(len(newR)):
        i = np.searchsorted(R,newR[j])
        for thing in ['Vavg','RVavg','m2avg','Rm2avg']:
            output[thing][j] = output[thing][j-1]
        while i > iused:
            if iused==len(R)-1:
                iused += 1
            else:
                output['Vavg'][j] += np.abs(V[iused+1])*flux[iused+1]
                output['RVavg'][j] += (np.abs(V[iused+1])*flux[iused+1]
                                       *R[iused+1])
                output['m2avg'][j] += (np.sqrt(V[iused+1]**2+sigma[iused+1]**2)
                                       *flux[iused+1])
                output['Rm2avg'][j] += (np.sqrt(V[iused+1]**2+sigma[iused+1]**2)
                                        *flux[iused+1]*R[iused+1])
                iused += 1
        output['lam'][j] = output['RVavg'][j]/output['Rm2avg'][j]
    return output
'''

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
