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

def write_rprofiles(path,data,metadata,comments):
    """
    Save radial profiles to file. Arguments are:
    -path: path to save file to
    -data: numpy array with named columns, assuming the first column is bin
    number and second column is "toplot" flag, to format as ints; any number
    of following columns accepted, formatted as float
    -metadata: a dictionary of metadata, keys being descriptive strings and
    values being a single number each, to save in the header
    -comments: a list of strings to be saved after the metadata in the header,
    with any further explanations or notes
    """
    header = ('Columns are as follows:'
              '\n {colnames}'
              '\nMetadata is as follows:'
              '\n {headermeta}'
              '\nAdditional comments:'
              '\n {headercomments}'
              ''.format)
    metalist = ['{spacing}{k}: {v}'.format(spacing=(21-len(k))*' ',k=k,v=v)
                for k,v in metadata.iteritems()]
    header = header(colnames=' '.join(data.dtype.names),
                    headermeta='\n '.join(metalist),
                    headercomments='\n '.join(comments))
    fmt = 2*['%2i'] + (len(data.dtype.names)-2)*['%9.5f']
    np.savetxt(path,data,header=header,fmt=fmt)
    return

def read_rprofiles_header(path):
    """
    Get radial profile metadata from file. See write_rprofiles for details.
    Returns a dictionary with one float per key, except "comments" which
     contains a list of strings.
    """
    metadata = {'comments':[]}
    header = open(path,'r')
    isheader = True
    ismeta = False
    iscomments = False
    while isheader:
        line = header.next()
        if not line[0]=='#':
            isheader=False
        elif line=='# Metadata is as follows:\n':
            ismeta = True
        elif line=='# Additional comments:\n':
            iscomments = True
            ismeta = False
        elif ismeta:
            key, value = line[1:].strip().split(':')
            metadata[key.strip()] = float(value)
        elif iscomments:
            metadata['comments'].append(line[3:-1])
    return metadata
