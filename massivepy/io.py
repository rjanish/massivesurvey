"""
Basic repetiitve io tasks
"""

import os
import re

import numpy as np

import utilities as utl
import massivepy.constants as const

def parse_paramfile_path(path):
    """
    Returns the directory and galaxy name, given a parameter file path.
    Checks that the first 7 characters of the file name match 'NGC####'.
    Also checks that the directory has #### somewhere in it.
    """
    output_dir = os.path.dirname(path)
    gal_name = os.path.basename(path)[0:7]
    if re.match(utl.force_full_match(const.re_ngc),gal_name) is None:
        raise Exception("Invalid galaxy name in parameter file path.")
    re_gal = re.search(const.re_ngc,output_dir)
    if re_gal is None:
        print "\nWarning, your output directory has no ngc name."
        print "Your organization is bad, go fix it.\n"
    elif re_gal.groups()[0] != gal_name[-4:]: #Test that galaxy number matches
        print "\nWarning, your output directory does not match your galaxy."
        print "Putting {} in {} directory\n".format(gal_name,re_gal.groups()[0])
    else:
        pass
    return output_dir, gal_name

def get_friendly_ppxf_output(path):
    """
    Returns a friendly dict- and recarray-based set of ppxf output.
    Only return the data we actually use in plots and want in text files.
    """
    #Anything not used here should probably get dropped from the fits files
    # as well, except in a debug=True type case.
    #Will want to make sure none of this information could come from earlier
    # files like the binned spectra files, because we should avoid too much
    # duplication. E.g. should avoid getting the spectra from here.
    data, headers = utl.fits_quickread(path)
    friendly_data = {}

    # populate basic scalar parameters
    nbins = headers[0]['NAXIS2']
    nmoments = headers[0]['NAXIS1']
    npixels = headers[2]['NAXIS1']
    friendly_data['nmoments'] = nmoments
    friendly_data['nbins'] = nbins
    #friendly_data['npixels'] = npixels
    friendly_data['add_deg'] = headers[0]['ADD_DEG']
    friendly_data['mul_deg'] = headers[0]['MUL_DEG']

    # populate moment stuff
    dt = {'names':['moment','err','scalederr'],'formats':3*[np.float64]}
    friendly_data['gh'] = np.zeros((nbins,nmoments),dtype=dt)
    for ibin in range(nbins):
        for imom in range(nmoments):
            friendly_data['gh'][ibin,imom] = tuple(data[0][:,ibin,imom])

    # populate template stuff

    # populate bin stuff
    dt = {'names':['id','chisq'],'formats':[int,np.float64]}
    friendly_data['bins'] = np.zeros((nbins,),dtype=dt)
    for ibin in range(nbins):
        friendly_data['bins'][ibin] = tuple(data[3][:2,ibin])

    # populate spectrum stuff
    # pretty sure this can go away since its all in the bin output!
    dt = {'names':['spectrum','noise','pixused','bestmodel'],
          'formats':4*['<f8']}
    friendly_data['spec'] = np.zeros((nbins,npixels),dtype=dt)
    for ibin in range(nbins):
        for ipix in range(npixels):
            friendly_data['spec'][ibin,ipix] = tuple(data[2][:4,ibin,ipix])

    # populate waves 
    # pretty sure this can go away since its all in the bin output!
    friendly_data['waves'] = data[6]

    return friendly_data


def get_friendly_ppxf_output_mc(path):
    """
    Returns a friendly dict- and recarray-based set of ppxf mc output.
    Only return the data we actually use in plots and want in text files.
    """
    data, headers = utl.fits_quickread(path)
    friendly_data = {}

    nmoments = headers[0]['NAXIS1']
    nruns = headers[0]['NAXIS2']
    nbins = headers[0]['NAXIS3']
    nthings = headers[0]['NAXIS4']

    friendly_data['nruns'] = nruns

    friendly_data['err'] = np.zeros((nbins,nmoments))
    for ibin in range(nbins):
        for imom in range(nmoments):
            friendly_data['err'][ibin,imom] = np.std(data[0][0,ibin,:,imom])

    return friendly_data
